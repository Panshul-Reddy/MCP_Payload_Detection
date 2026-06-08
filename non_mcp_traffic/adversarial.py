"""
Adversarial Non-MCP Traffic — generates JSON-RPC over HTTPS traffic that
intentionally MIMICS MCP patterns to test classifier robustness.

This is the hardest negative for our classifier because:
  - Uses JSON-RPC 2.0 (same as MCP)
  - Same Content-Type: application/json
  - Similar method names (tools/call, resources/list)
  - Over TLS (encrypted, same as MCP)
  - BUT: No SSE streaming, no MCP-specific headers, different timing

Also generates:
  - GraphQL-like traffic (POST with query field)
  - gRPC-like traffic (HTTP/2-style framing)
  - WebSocket JSON-RPC (bidirectional)

Usage:
    python -m non_mcp_traffic.adversarial --url https://localhost:5443 --requests 30
"""

import argparse
import json
import random
import time

import requests as req_lib


# JSON-RPC 2.0 requests that LOOK like MCP but aren't
JSONRPC_CALLS = [
    # Looks like MCP tool listing but is generic JSON-RPC
    {"jsonrpc": "2.0", "id": 1, "method": "system.listMethods", "params": {}},
    {"jsonrpc": "2.0", "id": 2, "method": "system.describe", "params": {}},
    # Looks like MCP tool calls
    {"jsonrpc": "2.0", "id": 3, "method": "invoke", "params": {"function": "calculate", "args": {"x": 10, "y": 20}}},
    {"jsonrpc": "2.0", "id": 4, "method": "execute", "params": {"command": "status", "options": {"verbose": True}}},
    # Looks like MCP resource access
    {"jsonrpc": "2.0", "id": 5, "method": "get", "params": {"resource": "config", "path": "/settings/general"}},
    {"jsonrpc": "2.0", "id": 6, "method": "list", "params": {"type": "resources", "filter": "active"}},
    # Notification-style (no id) — similar to MCP notifications
    {"jsonrpc": "2.0", "method": "notify", "params": {"event": "heartbeat", "timestamp": 0}},
    {"jsonrpc": "2.0", "method": "log", "params": {"level": "info", "message": "Client connected"}},
    # Batch requests (JSON-RPC allows arrays)
    [
        {"jsonrpc": "2.0", "id": 10, "method": "ping", "params": {}},
        {"jsonrpc": "2.0", "id": 11, "method": "status", "params": {}},
    ],
    # Large payloads
    {"jsonrpc": "2.0", "id": 7, "method": "search", "params": {
        "query": "network traffic analysis machine learning encrypted",
        "filters": {"language": "python", "sort": "relevance", "limit": 50},
        "context": "x" * random.randint(200, 2000)
    }},
    {"jsonrpc": "2.0", "id": 8, "method": "store", "params": {
        "key": "experiment_results",
        "value": {"accuracy": 0.98, "f1": 0.97, "features": list(range(75))},
        "metadata": {"timestamp": 0, "version": "2.0"}
    }},
]

# GraphQL-like requests
GRAPHQL_QUERIES = [
    {"query": "{ users { id name email } }", "variables": {}},
    {"query": "{ repository(owner: \"test\", name: \"repo\") { stars forks issues { title } } }"},
    {"query": "mutation { createUser(name: \"test\") { id } }"},
    {"query": "{ search(query: \"machine learning\") { results { title url score } } }",
     "variables": {"limit": 20}},
    {"query": "subscription { onMessage { id content timestamp sender } }"},
]

# Simulated gRPC-web style requests (binary-like with JSON fallback)
GRPC_LIKE = [
    {"service": "UserService", "method": "GetUser", "request": {"id": random.randint(1, 1000)}},
    {"service": "SearchService", "method": "Search", "request": {"query": "test", "page": 1}},
    {"service": "HealthService", "method": "Check", "request": {}},
    {"service": "MetricsService", "method": "Collect", "request": {"metrics": ["cpu", "memory", "disk"]}},
]


def run_adversarial_traffic(base_url: str, num_requests: int, ca_cert: str = None):
    """Generate adversarial traffic that mimics MCP patterns."""
    print(f"  [Adversarial] Generating {num_requests} requests to {base_url}")

    for i in range(num_requests):
        sess = req_lib.Session()
        sess.headers["Connection"] = "close"
        if ca_cert:
            sess.verify = ca_cert
        else:
            sess.verify = False

        try:
            traffic_type = random.choices(
                ["jsonrpc", "graphql", "grpc", "sse_poll"],
                weights=[40, 25, 20, 15], k=1
            )[0]

            if traffic_type == "jsonrpc":
                # JSON-RPC that looks like MCP
                payload = random.choice(JSONRPC_CALLS)
                if isinstance(payload, dict) and "timestamp" in str(payload):
                    payload = json.loads(json.dumps(payload).replace('"timestamp": 0', f'"timestamp": {time.time()}'))
                sess.headers["Content-Type"] = "application/json"
                resp = sess.post(f"{base_url}/echo", json=payload, timeout=10)

            elif traffic_type == "graphql":
                # GraphQL query
                payload = random.choice(GRAPHQL_QUERIES)
                sess.headers["Content-Type"] = "application/json"
                resp = sess.post(f"{base_url}/echo", json=payload, timeout=10)

            elif traffic_type == "grpc":
                # gRPC-web style
                payload = random.choice(GRPC_LIKE)
                sess.headers["Content-Type"] = "application/grpc-web+json"
                sess.headers["X-Grpc-Web"] = "1"
                resp = sess.post(f"{base_url}/echo", json=payload, timeout=10)

            elif traffic_type == "sse_poll":
                # Polling that mimics SSE reconnection pattern
                sess.headers["Accept"] = "text/event-stream"
                sess.headers["Cache-Control"] = "no-cache"
                resp = sess.get(f"{base_url}/health", timeout=5)
                # Quick follow-up (mimics SSE reconnect)
                time.sleep(random.uniform(0.05, 0.2))
                sess2 = req_lib.Session()
                sess2.headers["Connection"] = "close"
                sess2.verify = sess.verify
                sess2.headers["Accept"] = "text/event-stream"
                sess2.get(f"{base_url}/health", timeout=5)
                sess2.close()

        except Exception:
            pass
        finally:
            sess.close()

        # Timing that mimics MCP patterns (bursty with SSE-like gaps)
        pattern = random.choices(["burst", "normal", "sse_gap"], weights=[30, 40, 30], k=1)[0]
        if pattern == "burst":
            time.sleep(random.uniform(0.02, 0.08))
        elif pattern == "normal":
            time.sleep(random.uniform(0.2, 0.8))
        else:
            # Mimic SSE reconnection timing
            time.sleep(random.uniform(0.5, 2.0))

    print(f"  [Adversarial] Done: {num_requests} requests")


def main():
    parser = argparse.ArgumentParser(description="Adversarial Non-MCP Traffic")
    parser.add_argument("--url", default="https://localhost:5443")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--cert", default=None)
    args = parser.parse_args()

    # Suppress TLS warnings for self-signed certs
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    run_adversarial_traffic(args.url, args.requests, args.cert)


if __name__ == "__main__":
    main()
