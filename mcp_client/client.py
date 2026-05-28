"""
MCP Client — generates realistic, randomized tool-call traffic against the
MCP server for traffic classification experiments.

Each session opens one or more short-lived SSE connections and calls
random tools with varying payload sizes to produce diverse network flows.

Usage:
    # Plain HTTP
    python -m mcp_client.client --url http://localhost:8000/sse --sessions 3 --requests 20

    # HTTPS / TLS
    python -m mcp_client.client --url https://localhost:8443/sse --sessions 3 --requests 20 --cert certs/server.crt
"""

import argparse
import asyncio
import random
import ssl
import string

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client


# ---------------------------------------------------------------------------
# Random payload generators
# ---------------------------------------------------------------------------

_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "xray", "yankee", "zulu", "the", "quick",
    "brown", "fox", "jumps", "over", "lazy", "dog", "network",
    "traffic", "analysis", "machine", "learning", "classification",
    "protocol", "encrypted", "payload", "detection", "metadata",
    "flow", "packet", "capture", "feature", "model",
]


def _random_string(min_len: int = 10, max_len: int = 200) -> str:
    """Generate a random text string from the word list."""
    target_len = random.randint(min_len, max_len)
    words = []
    current = 0
    while current < target_len:
        w = random.choice(_WORDS)
        words.append(w)
        current += len(w) + 1
    return " ".join(words)


def _random_calculator_call() -> tuple[str, dict]:
    """Generate a random calculator tool call."""
    ops = [
        ("add", {"a": random.uniform(-1000, 1000), "b": random.uniform(-1000, 1000)}),
        ("subtract", {"a": random.uniform(-1000, 1000), "b": random.uniform(-1000, 1000)}),
        ("multiply", {"a": random.uniform(-100, 100), "b": random.uniform(-100, 100)}),
        ("divide", {"a": random.uniform(-100, 100), "b": random.uniform(-100, 100)}),
        ("power", {"base": random.uniform(1, 10), "exponent": random.uniform(0, 5)}),
        ("sqrt", {"x": random.uniform(0, 10000)}),
    ]
    return random.choice(ops)


def _random_echo_call() -> tuple[str, dict]:
    """Generate a random echo tool call with varying payload sizes."""
    size_tier = random.choices(
        ["short", "medium", "long"], weights=[40, 40, 20], k=1
    )[0]

    if size_tier == "short":
        msg = _random_string(5, 30)
    elif size_tier == "medium":
        msg = _random_string(50, 200)
    else:
        msg = _random_string(500, 2000)

    tool = random.choice(["echo", "echo_upper", "echo_reversed"])
    return tool, {"message": msg}


def _random_weather_call() -> tuple[str, dict]:
    """Generate a random weather tool call."""
    city = random.choice(["New York", "London", "Tokyo", "Sydney", "Mumbai"])
    if random.random() < 0.5:
        return "get_weather", {"city": city}
    else:
        return "get_forecast", {"city": city, "days": random.randint(1, 7)}


def _random_string_call() -> tuple[str, dict]:
    """Generate a random string utility tool call."""
    text = _random_string(20, 500)
    ops = [
        ("count_words", {"text": text}),
        ("count_characters", {"text": text}),
        ("to_title_case", {"text": text}),
        ("replace_substring", {"text": text, "old": random.choice(_WORDS), "new": "REPLACED"}),
        ("split_text", {"text": text, "delimiter": " "}),
    ]
    return random.choice(ops)


def _random_tool_call() -> tuple[str, dict]:
    """Pick a random tool call from any category."""
    category = random.choices(
        [_random_calculator_call, _random_echo_call, _random_weather_call, _random_string_call],
        weights=[30, 30, 20, 20],
        k=1,
    )[0]
    return category()


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

async def run_session(
    url: str,
    session_id: int,
    num_requests: int,
    ca_cert: str | None = None,
) -> None:
    """Run one MCP client session making tool calls over SSE connections."""

    # Create httpx client factory for TLS support
    def make_client(
        headers: dict | None = None,
        timeout: object | None = None,
        auth: object | None = None,
    ) -> httpx.AsyncClient:
        kwargs = {}
        if headers:
            kwargs["headers"] = headers
        if timeout:
            kwargs["timeout"] = timeout
        if auth:
            kwargs["auth"] = auth
        if ca_cert:
            kwargs["verify"] = ca_cert
        return httpx.AsyncClient(**kwargs)

    # Break requests across multiple short-lived connections
    requests_done = 0
    while requests_done < num_requests:
        batch = random.randint(1, min(12, num_requests - requests_done))
        try:
            async with sse_client(
                url,
                timeout=30,
                httpx_client_factory=make_client,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    for i in range(batch):
                        tool_name, tool_args = _random_tool_call()
                        try:
                            result = await session.call_tool(tool_name, tool_args)
                            requests_done += 1
                        except Exception as e:
                            print(f"  [S{session_id}] Tool call failed: {tool_name}: {e}")
                            requests_done += 1

                        # Randomized inter-request delay (bursty + normal + idle)
                        delay_type = random.choices(
                            ["burst", "normal", "idle"],
                            weights=[30, 50, 20],
                            k=1,
                        )[0]
                        if delay_type == "burst":
                            await asyncio.sleep(random.uniform(0.01, 0.1))
                        elif delay_type == "normal":
                            await asyncio.sleep(random.uniform(0.1, 0.5))
                        else:
                            await asyncio.sleep(random.uniform(0.5, 2.0))

        except Exception as e:
            print(f"  [S{session_id}] Connection error: {e}")
            await asyncio.sleep(1.0)
            requests_done += 1

        # Small gap between connections
        await asyncio.sleep(random.uniform(0.1, 0.5))

    print(f"  [S{session_id}] Completed {requests_done} requests")


async def run_all_sessions(
    url: str,
    num_sessions: int,
    num_requests: int,
    ca_cert: str | None = None,
) -> None:
    """Run all sessions concurrently."""
    tasks = [
        run_session(url, i, num_requests, ca_cert)
        for i in range(num_sessions)
    ]
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MCP Traffic Generator Client")
    parser.add_argument("--url", default="http://localhost:8000/sse")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--cert", default=None, help="CA certificate for TLS verification")
    args = parser.parse_args()

    print(f"[MCP Client] Connecting to {args.url} ({args.sessions} sessions, {args.requests} requests each)")
    asyncio.run(run_all_sessions(args.url, args.sessions, args.requests, args.cert))
    print("[MCP Client] Done")


if __name__ == "__main__":
    main()
