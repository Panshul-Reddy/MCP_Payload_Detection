"""
Raw TCP Traffic Generator — opens TCP connections and exchanges framed
binary/text messages using a custom protocol.

Frame format: [type:1B][length:4B][payload]

Message types:
    0x01 = TEXT
    0x02 = BINARY
    0x03 = PING
    0x04 = PONG

Usage:
    python -m non_mcp_traffic.tcp_traffic --host localhost --port 5002 --connections 3 --messages 10
"""

import argparse
import asyncio
import os
import random
import struct


MSG_TYPE_TEXT = 0x01
MSG_TYPE_BINARY = 0x02
MSG_TYPE_PING = 0x03
MSG_TYPE_PONG = 0x04

_SAMPLE_COMMANDS = [
    "GET /status HTTP/1.1",
    "POST /api/data HTTP/1.1",
    "SELECT * FROM users WHERE id = 1",
    "INSERT INTO logs (msg) VALUES ('test')",
    "ls -la /var/log",
    "cat /etc/hostname",
    "echo 'health check'",
    "ping localhost",
    "netstat -tlnp",
    "curl http://internal-api/health",
    "systemctl status nginx",
    "docker ps --format json",
]


def _frame(msg_type: int, payload: bytes) -> bytes:
    """Encode a message with [type:1B][length:4B][payload] header."""
    return struct.pack("!BI", msg_type, len(payload)) + payload


def _random_text_payload() -> bytes:
    if random.random() < 0.5:
        return random.choice(_SAMPLE_COMMANDS).encode("utf-8")
    else:
        parts = random.sample(_SAMPLE_COMMANDS, k=random.randint(2, 5))
        return "\n".join(parts).encode("utf-8")


def _random_message() -> bytes:
    """Generate a random framed message."""
    choice = random.choices(
        ["text", "binary", "ping"],
        weights=[50, 33, 17],
        k=1,
    )[0]

    if choice == "text":
        return _frame(MSG_TYPE_TEXT, _random_text_payload())
    elif choice == "binary":
        size = random.randint(4, 4096)
        return _frame(MSG_TYPE_BINARY, os.urandom(size))
    else:
        return _frame(MSG_TYPE_PING, b"ping")


async def _tcp_session(
    host: str,
    port: int,
    num_messages: int,
    session_id: int,
) -> None:
    """Send messages over short-lived TCP connections."""
    messages_sent = 0

    while messages_sent < num_messages:
        batch = random.randint(1, min(10, num_messages - messages_sent))
        try:
            reader, writer = await asyncio.open_connection(host, port)

            for _ in range(batch):
                msg = _random_message()
                writer.write(msg)
                await writer.drain()

                # Try to read response (server echoes back)
                try:
                    header = await asyncio.wait_for(reader.read(5), timeout=2.0)
                    if len(header) == 5:
                        _, payload_len = struct.unpack("!BI", header)
                        if payload_len > 0 and payload_len < 65536:
                            await asyncio.wait_for(
                                reader.read(payload_len), timeout=2.0
                            )
                except (asyncio.TimeoutError, Exception):
                    pass

                messages_sent += 1
                await asyncio.sleep(random.uniform(0.02, 0.2))

            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

        except Exception:
            messages_sent += 1
            await asyncio.sleep(0.5)

        await asyncio.sleep(random.uniform(0.1, 0.5))

    print(f"  [TCP S{session_id}] Completed {messages_sent} messages")


async def run_tcp_traffic(
    host: str,
    port: int,
    num_connections: int,
    num_messages: int,
) -> None:
    tasks = [
        _tcp_session(host, port, num_messages, i)
        for i in range(num_connections)
    ]
    await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="TCP Traffic Generator")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--connections", type=int, default=3)
    parser.add_argument("--messages", type=int, default=10)
    args = parser.parse_args()

    print(f"[TCP Client] Connecting to {args.host}:{args.port}")
    asyncio.run(run_tcp_traffic(args.host, args.port, args.connections, args.messages))
    print("[TCP Client] Done")


if __name__ == "__main__":
    main()
