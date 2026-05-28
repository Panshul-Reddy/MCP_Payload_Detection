"""
MCP Server — exposes 16 tools across 4 categories via HTTP+SSE transport.

Categories:
    Calculator : add, subtract, multiply, divide, power, sqrt
    Echo       : echo, echo_upper, echo_reversed
    Weather    : get_weather, get_forecast  (simulated)
    String     : count_words, count_characters, to_title_case, replace_substring, split_text

Usage:
    # Plain HTTP (port 8000)
    python -m mcp_server.server --port 8000

    # HTTPS / TLS (port 8443)
    python -m mcp_server.server --port 8443 --tls --cert certs/server.crt --key certs/server.key
"""

import argparse
import math
import random
import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MCP Traffic Lab")

# ---------------------------------------------------------------------------
# Calculator tools
# ---------------------------------------------------------------------------

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        return float("inf")
    return a / b

@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return math.pow(base, exponent)

@mcp.tool()
def sqrt(x: float) -> float:
    """Square root of x."""
    if x < 0:
        return float("nan")
    return math.sqrt(x)

# ---------------------------------------------------------------------------
# Echo tools
# ---------------------------------------------------------------------------

@mcp.tool()
def echo(message: str) -> str:
    """Echo back the message."""
    return message

@mcp.tool()
def echo_upper(message: str) -> str:
    """Echo back the message in uppercase."""
    return message.upper()

@mcp.tool()
def echo_reversed(message: str) -> str:
    """Echo back the message reversed."""
    return message[::-1]

# ---------------------------------------------------------------------------
# Weather tools (simulated data)
# ---------------------------------------------------------------------------

_CITIES = {
    "New York": (40.7, -74.0),
    "London": (51.5, -0.1),
    "Tokyo": (35.7, 139.7),
    "Sydney": (-33.9, 151.2),
    "Mumbai": (19.1, 72.9),
}

@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city (simulated)."""
    if city not in _CITIES:
        return f"Unknown city: {city}. Known cities: {', '.join(_CITIES.keys())}"
    temp = random.randint(-5, 40)
    humidity = random.randint(20, 95)
    conditions = random.choice(["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy"])
    return (
        f"Weather in {city}: {conditions}, "
        f"{temp}°C, humidity {humidity}%, "
        f"coords {_CITIES[city]}"
    )

@mcp.tool()
def get_forecast(city: str, days: int = 3) -> str:
    """Get a multi-day forecast for a city (simulated)."""
    if city not in _CITIES:
        return f"Unknown city: {city}."
    days = min(max(days, 1), 7)
    forecasts = []
    today = datetime.date.today()
    for i in range(days):
        d = today + datetime.timedelta(days=i)
        temp_high = random.randint(10, 40)
        temp_low = temp_high - random.randint(5, 15)
        cond = random.choice(["Sunny", "Cloudy", "Rainy", "Thunderstorms", "Clear"])
        forecasts.append(f"  {d}: {cond}, {temp_low}–{temp_high}°C")
    return f"Forecast for {city}:\n" + "\n".join(forecasts)

# ---------------------------------------------------------------------------
# String utility tools
# ---------------------------------------------------------------------------

@mcp.tool()
def count_words(text: str) -> int:
    """Count the number of words in the text."""
    return len(text.split())

@mcp.tool()
def count_characters(text: str) -> int:
    """Count the number of characters in the text."""
    return len(text)

@mcp.tool()
def to_title_case(text: str) -> str:
    """Convert text to title case."""
    return text.title()

@mcp.tool()
def replace_substring(text: str, old: str, new: str) -> str:
    """Replace all occurrences of 'old' with 'new' in text."""
    return text.replace(old, new)

@mcp.tool()
def split_text(text: str, delimiter: str = " ") -> list[str]:
    """Split text by delimiter."""
    return text.split(delimiter)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MCP Traffic Lab Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tls", action="store_true", help="Enable HTTPS/TLS")
    parser.add_argument("--cert", default="certs/server.crt", help="TLS certificate path")
    parser.add_argument("--key", default="certs/server.key", help="TLS private key path")
    args = parser.parse_args()

    if args.tls:
        import uvicorn

        # Get the SSE ASGI app from FastMCP
        app = mcp.sse_app()

        print(f"[MCP Server] Starting HTTPS on {args.host}:{args.port}")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            ssl_certfile=args.cert,
            ssl_keyfile=args.key,
            log_level="warning",
        )
    else:
        print(f"[MCP Server] Starting HTTP on {args.host}:{args.port}")
        mcp.run(transport="sse")


if __name__ == "__main__":
    main()
