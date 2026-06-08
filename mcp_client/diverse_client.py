"""
Diverse MCP Client — connects to different MCP server types and calls
their tools with realistic patterns to generate varied MCP traffic.

Supports: github, filesystem, fetch, memory, database server types.
Each server type gets a tailored sequence of tool calls that mimics
how a real LLM agent would interact with that type of server.

Usage:
    python -m mcp_client.diverse_client --url https://localhost:9001/sse --type github --requests 20
    python -m mcp_client.diverse_client --url https://localhost:9002/sse --type filesystem --requests 20
"""

import argparse
import asyncio
import json
import random
import ssl
import sys
import time

from mcp import ClientSession
from mcp.client.sse import sse_client


# ---------------------------------------------------------------------------
# Tool call sequences per server type
# ---------------------------------------------------------------------------

GITHUB_CALLS = [
    ("search_repositories", {"query": "machine learning", "language": "python"}),
    ("search_repositories", {"query": "react", "language": "javascript"}),
    ("search_repositories", {"query": "api framework"}),
    ("get_repository", {"owner": "facebook", "repo": "react"}),
    ("get_repository", {"owner": "tensorflow", "repo": "tensorflow"}),
    ("get_repository", {"owner": "tiangolo", "repo": "fastapi"}),
    ("list_issues", {"owner": "facebook", "repo": "react", "state": "open"}),
    ("list_issues", {"owner": "torvalds", "repo": "linux", "state": "closed", "per_page": 5}),
    ("search_code", {"query": "async def", "language": "python"}),
    ("search_code", {"query": "import React", "language": "javascript"}),
    ("get_pull_requests", {"owner": "microsoft", "repo": "vscode", "state": "open"}),
    ("get_pull_requests", {"owner": "facebook", "repo": "react", "state": "closed"}),
]

FILESYSTEM_CALLS = [
    ("list_directory", {"path": "/home/user"}),
    ("list_directory", {"path": "/home/user/project"}),
    ("read_file", {"path": "/home/user/project/main.py"}),
    ("read_file", {"path": "/home/user/project/config.yaml"}),
    ("read_file", {"path": "/home/user/project/README.md"}),
    ("read_file", {"path": "/home/user/data/results.csv"}),
    ("get_file_info", {"path": "/home/user/project/main.py"}),
    ("search_files", {"pattern": "def main", "directory": "/home/user"}),
    ("search_files", {"pattern": "import", "directory": "/home/user/project"}),
    ("write_file", {"path": "/home/user/project/output.txt", "content": "Analysis results:\n" + "data line\n" * 20}),
    ("read_file", {"path": "/home/user/notes.txt"}),
    ("list_directory", {"path": "/home/user/data"}),
]

FETCH_CALLS = [
    ("fetch_url", {"url": "https://example.com"}),
    ("fetch_url", {"url": "https://news.ycombinator.com", "max_length": 3000}),
    ("fetch_url", {"url": "https://httpbin.org/json"}),
    ("fetch_json", {"url": "https://api.github.com"}),
    ("fetch_json", {"url": "https://jsonplaceholder.typicode.com/posts"}),
    ("extract_links", {"url": "https://example.com"}),
    ("extract_links", {"url": "https://news.ycombinator.com"}),
    ("fetch_url", {"url": "https://docs.python.org/3/", "max_length": 5000}),
    ("fetch_json", {"url": "https://api.example.com/data"}),
    ("fetch_url", {"url": "https://en.wikipedia.org/wiki/Machine_learning", "max_length": 2000}),
]

MEMORY_CALLS = [
    ("create_entity", {"name": "Python", "entity_type": "programming_language", "observations": ["General-purpose", "Interpreted", "High-level"]}),
    ("create_entity", {"name": "TensorFlow", "entity_type": "framework", "observations": ["ML framework", "By Google", "Open source"]}),
    ("create_entity", {"name": "FastAPI", "entity_type": "framework", "observations": ["Web framework", "Async", "Type hints"]}),
    ("create_relation", {"from_entity": "TensorFlow", "to_entity": "Python", "relation_type": "written_in"}),
    ("create_relation", {"from_entity": "FastAPI", "to_entity": "Python", "relation_type": "written_in"}),
    ("add_observation", {"entity_name": "Python", "observation": "Most popular for data science"}),
    ("add_observation", {"entity_name": "TensorFlow", "observation": "Supports GPU acceleration"}),
    ("search_entities", {"query": "Python"}),
    ("search_entities", {"query": "framework"}),
    ("get_graph", {}),
    ("create_entity", {"name": "MCP", "entity_type": "protocol", "observations": ["Model Context Protocol", "By Anthropic", "For AI agents"]}),
    ("create_relation", {"from_entity": "MCP", "to_entity": "Python", "relation_type": "implemented_in"}),
]

DATABASE_CALLS = [
    ("list_tables", {}),
    ("describe_table", {"table_name": "users"}),
    ("describe_table", {"table_name": "orders"}),
    ("query", {"sql": "SELECT * FROM users LIMIT 10"}),
    ("query", {"sql": "SELECT * FROM orders WHERE status = 'pending' LIMIT 20"}),
    ("query", {"sql": "SELECT COUNT(*) FROM users"}),
    ("query", {"sql": "SELECT COUNT(*) FROM orders"}),
    ("query", {"sql": "SELECT * FROM logs WHERE level = 'ERROR' LIMIT 15"}),
    ("describe_table", {"table_name": "logs"}),
    ("query", {"sql": "SELECT user_id, SUM(amount) FROM orders GROUP BY user_id LIMIT 10"}),
    ("query", {"sql": "SELECT * FROM users WHERE role = 'admin'"}),
    ("query", {"sql": "SELECT * FROM logs LIMIT 50"}),
]

CALL_SEQUENCES = {
    "github": GITHUB_CALLS,
    "filesystem": FILESYSTEM_CALLS,
    "fetch": FETCH_CALLS,
    "memory": MEMORY_CALLS,
    "database": DATABASE_CALLS,
}


async def run_diverse_session(url: str, server_type: str, num_requests: int, session_id: int = 0):
    """Connect to an MCP server and make tool calls."""
    calls = CALL_SEQUENCES.get(server_type, GITHUB_CALLS)

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    try:
        async with sse_client(url, ssl_context=ssl_ctx) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                print(f"  [S{session_id}][{server_type}] Connected, tools: {tool_names}")

                for i in range(num_requests):
                    tool_name, args = random.choice(calls)

                    if tool_name not in tool_names:
                        tool_name = random.choice(tool_names)
                        args = {}

                    try:
                        result = await session.call_tool(tool_name, args)
                        resp_len = sum(len(c.text) for c in result.content if hasattr(c, 'text'))
                    except Exception as e:
                        resp_len = 0

                    # Bursty timing pattern
                    pattern = random.choices(["burst", "normal", "slow"], weights=[30, 50, 20], k=1)[0]
                    if pattern == "burst":
                        await asyncio.sleep(random.uniform(0.05, 0.15))
                    elif pattern == "normal":
                        await asyncio.sleep(random.uniform(0.3, 1.0))
                    else:
                        await asyncio.sleep(random.uniform(1.0, 3.0))

                print(f"  [S{session_id}][{server_type}] Completed {num_requests} requests")

    except Exception as e:
        print(f"  [S{session_id}][{server_type}] Error: {e}")


async def run_diverse_client(url: str, server_type: str, num_requests: int, sessions: int = 2):
    """Run multiple sessions against a server."""
    tasks = [run_diverse_session(url, server_type, num_requests, i) for i in range(sessions)]
    await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="Diverse MCP Client")
    parser.add_argument("--url", required=True, help="MCP server SSE URL")
    parser.add_argument("--type", choices=list(CALL_SEQUENCES.keys()), required=True)
    parser.add_argument("--requests", type=int, default=15)
    parser.add_argument("--sessions", type=int, default=2)
    args = parser.parse_args()

    print(f"[Diverse Client] Connecting to {args.url} (type={args.type})")
    asyncio.run(run_diverse_client(args.url, args.type, args.requests, args.sessions))
    print(f"[Diverse Client] Done")


if __name__ == "__main__":
    main()
