#!/usr/bin/env python3
"""B1 interop gate — official Python MCP SDK **client** speaking to the
langchainrust Streamable HTTP server (examples/streamable_echo_server.rs).

    python py_streamable_client.py --url http://127.0.0.1:PORT/mcp \
        [--bearer TOKEN]

Exits 0 only after the full official-client flow succeeds: initialize
handshake, tools/list, tools/call roundtrip and ping. Prints PY_STREAMABLE_OK
as its final line on success. Requires `pip install mcp` (see requirements.txt).
"""

import argparse
import asyncio
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def run(url: str, bearer: str | None) -> None:
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    async with streamablehttp_client(url, headers=headers) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            session_id = get_session_id()
            print(f"[py] connected; Mcp-Session-Id={session_id}")

            listed = await session.list_tools()
            names = [tool.name for tool in listed.tools]
            print(f"[py] tools/list -> {names}")
            assert "echo" in names, f"expected 'echo' tool, got {names}"

            marker = f"py-sdk-marker-{int(time.time() * 1000)}"
            called = await session.call_tool("echo", {"msg": marker})
            text = "".join(
                block.text for block in called.content if getattr(block, "type", "") == "text"
            )
            print(f"[py] tools/call -> {text!r}")
            assert not called.isError, "echo tool call flagged isError"
            assert marker in text, f"echo payload missing marker {marker}: {text}"

            await session.send_ping()
            print("[py] ping ok")

    print("PY_STREAMABLE_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--bearer")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.url, args.bearer))
    except Exception as exc:  # noqa: BLE001 - interop gate reports any failure
        print(f"[py] INTEROP FAILED: {exc!r}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
