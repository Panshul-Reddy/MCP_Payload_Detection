"""
WebSocket Traffic Generator — connects to the WS server and sends a
realistic mix of ping/echo/time messages with variable content sizes.

Usage:
    python -m non_mcp_traffic.websocket_traffic --url ws://localhost:5001 --sessions 2 --messages 30
"""

import argparse
import asyncio
import json
import random
import time

import websockets


_WS_WORDS = [
    "stream", "socket", "connect", "message", "channel", "subscribe",
    "publish", "event", "data", "binary", "frame", "handshake",
    "protocol", "upgrade", "bidirectional", "realtime", "push",
    "notification", "heartbeat", "keepalive",
]


def _random_ws_text(min_len: int = 10, max_len: int = 500) -> str:
    target = random.randint(min_len, max_len)
    words = []
    current = 0
    while current < target:
        w = random.choice(_WS_WORDS)
        words.append(w)
        current += len(w) + 1
    return " ".join(words)


async def run_ws_session(
    url: str,
    num_messages: int,
    session_id: int,
) -> None:
    """Send messages over short-lived WebSocket connections."""
    messages_sent = 0

    while messages_sent < num_messages:
        batch = random.randint(1, min(10, num_messages - messages_sent))
        try:
            async with websockets.connect(url) as ws:
                for _ in range(batch):
                    msg_type = random.choices(
                        ["ping", "echo", "time"],
                        weights=[20, 60, 20],
                        k=1,
                    )[0]

                    if msg_type == "ping":
                        payload = json.dumps({"type": "ping"})
                    elif msg_type == "time":
                        payload = json.dumps({"type": "time"})
                    else:
                        size = random.choices(
                            ["short", "medium", "long"],
                            weights=[40, 40, 20],
                            k=1,
                        )[0]
                        if size == "short":
                            text = _random_ws_text(5, 30)
                        elif size == "medium":
                            text = _random_ws_text(50, 200)
                        else:
                            text = _random_ws_text(500, 2000)
                        payload = json.dumps({"type": "echo", "data": text})

                    await ws.send(payload)
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        pass
                    messages_sent += 1

                    await asyncio.sleep(random.uniform(0.05, 0.3))

        except Exception as e:
            messages_sent += 1
            await asyncio.sleep(0.5)

        await asyncio.sleep(random.uniform(0.1, 0.5))

    print(f"  [WS S{session_id}] Completed {messages_sent} messages")


async def run_ws_traffic(
    url: str,
    num_sessions: int,
    num_messages: int,
) -> None:
    tasks = [
        run_ws_session(url, num_messages, i)
        for i in range(num_sessions)
    ]
    await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="WebSocket Traffic Generator")
    parser.add_argument("--url", default="ws://localhost:5001")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--messages", type=int, default=30)
    args = parser.parse_args()

    print(f"[WS Client] Connecting to {args.url}")
    asyncio.run(run_ws_traffic(args.url, args.sessions, args.messages))
    print("[WS Client] Done")


if __name__ == "__main__":
    main()
