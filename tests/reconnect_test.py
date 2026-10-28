# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end connection lifecycle: a real ColabWebSocketServer and ColabProxyClient, with a fake Colab tab on the
other end of the websocket. The fake tab is an MCP server that, like any spec-following one, answers nothing before
`initialize`, so a proxy that reuses a stale session against a new tab fails here the way it fails in a browser."""

import asyncio
import contextlib
import json
from unittest.mock import patch

import pytest
import websockets

from colab_mcp import session
from colab_mcp.websocket_server import COLAB, ColabWebSocketServer

TOOLS = [{"name": "run_code_cell", "inputSchema": {"type": "object"}}]


class FakeColabTab:
    def __init__(self, wss):
        self.wss = wss

    async def __aenter__(self):
        self.ws = await websockets.connect(
            f"ws://localhost:{self.wss.port}",
            origin=COLAB,
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {self.wss.token}"},
        )
        self._task = asyncio.create_task(self._serve())
        return self

    async def _serve(self):
        initialized = False
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            async for raw in self.ws:
                msg = json.loads(raw)
                if "id" not in msg:  # a notification
                    continue
                reply = {"jsonrpc": "2.0", "id": msg["id"]}
                if msg["method"] == "initialize":
                    initialized = True
                    reply["result"] = {
                        "protocolVersion": msg["params"]["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake-colab", "version": "0"},
                    }
                elif not initialized:
                    reply["error"] = {"code": -32002, "message": "not initialized"}
                elif msg["method"] == "tools/list":
                    reply["result"] = {"tools": TOOLS}
                else:
                    reply["result"] = {}
                await self.ws.send(json.dumps(reply))

    async def __aexit__(self, *exc):
        await self.ws.close()
        await self._task


async def wait_until(predicate, timeout=5.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def tool_names(client):
    return [t.name for t in await client.proxy_mcp_client.list_tools()]


@pytest.mark.asyncio
async def test_a_tab_that_connects_after_a_timed_out_wait_still_gets_a_session():
    async with ColabWebSocketServer() as wss, session.ColabProxyClient(wss) as client:
        with patch("colab_mcp.session.UI_CONNECTION_TIMEOUT", 0.2):
            await client.await_proxy_connection()  # nobody opened the tab in time
        assert not client.is_connected()

        async with FakeColabTab(wss):
            with patch("colab_mcp.session.UI_CONNECTION_TIMEOUT", 5.0):
                await client.await_proxy_connection()
            assert client.is_connected()
            assert await tool_names(client) == ["run_code_cell"]


@pytest.mark.asyncio
async def test_a_new_tab_after_the_old_one_closes_gets_a_fresh_session():
    async with ColabWebSocketServer() as wss, session.ColabProxyClient(wss) as client:
        async with FakeColabTab(wss):
            await client.await_proxy_connection()
            assert await tool_names(client) == ["run_code_cell"]
        await wait_until(lambda: not client.is_connected())

        async with FakeColabTab(wss):  # e.g. the user reloaded the Colab page
            await client.await_proxy_connection()
            assert client.is_connected()
            assert await tool_names(client) == ["run_code_cell"]


@pytest.mark.asyncio
async def test_the_connect_tool_answers_false_after_a_timeout_instead_of_hanging():
    async with ColabWebSocketServer() as wss, session.ColabProxyClient(wss) as client:
        middleware = session.ColabProxyMiddleware(client)

        class Ctx:
            class message:
                name = session.INJECTED_TOOL_NAME

            class fastmcp_context:
                @staticmethod
                async def report_progress(**kw):
                    pass

        async def call_next(_):
            return "tool ran"

        with patch("colab_mcp.session.UI_CONNECTION_TIMEOUT", 0.2):
            for _ in range(2):  # the second call is the one that used to hang
                async with asyncio.timeout(5):
                    result = await middleware.on_call_tool(Ctx, call_next)
                assert result.structured_content == {"result": False}
