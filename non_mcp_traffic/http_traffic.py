"""
HTTP Traffic Generator — sends randomized REST requests to produce
non-MCP HTTP/HTTPS traffic for classification experiments.

Generates diverse payload sizes and bursty timing patterns.
Creates a NEW TCP connection per request (Connection: close) so each
request produces a distinct network flow.

Usage:
    python -m non_mcp_traffic.http_traffic --url http://localhost:5000 --requests 50
    python -m non_mcp_traffic.http_traffic --url https://localhost:5443 --requests 50 --cert certs/server.crt
"""

import argparse
import asyncio
import random
import time

import requests as req_lib


_SAMPLE_TEXTS = [
    "hello world", "testing 123", "quick brown fox",
    "network traffic analysis", "machine learning model",
    "encrypted payload detection", "flow metadata features",
    "inter-arrival time patterns", "packet size distribution",
    "classification accuracy metrics",
]


def _random_payload() -> dict:
    """Generate a random JSON payload with varying sizes."""
    tier = random.choices(
        ["tiny", "small", "medium", "large"],
        weights=[20, 35, 30, 15],
        k=1,
    )[0]

    if tier == "tiny":
        return {"msg": random.choice(_SAMPLE_TEXTS)}
    elif tier == "small":
        return {
            "name": f"item_{random.randint(1, 10000)}",
            "value": random.random() * 100,
            "tags": random.sample(_SAMPLE_TEXTS, k=random.randint(1, 3)),
        }
    elif tier == "medium":
        return {
            "title": f"Document {random.randint(1, 1000)}",
            "content": " ".join(random.choices(_SAMPLE_TEXTS, k=random.randint(5, 15))),
            "metadata": {
                "author": f"user_{random.randint(1, 100)}",
                "version": random.randint(1, 10),
                "timestamp": time.time(),
            },
        }
    else:  # large
        return {
            "batch": [
                {
                    "id": i,
                    "data": " ".join(random.choices(_SAMPLE_TEXTS, k=random.randint(3, 8))),
                    "priority": random.choice(["low", "medium", "high"]),
                }
                for i in range(random.randint(5, 20))
            ],
            "summary": " ".join(random.choices(_SAMPLE_TEXTS, k=10)),
        }


def run_http_traffic(
    base_url: str,
    num_requests: int,
    delay: float = 0.2,
    ca_cert: str | None = None,
) -> None:
    """Generate HTTP traffic with bursty timing patterns."""
    item_ids = []

    for i in range(num_requests):
        # New session per request → new TCP flow
        sess = req_lib.Session()
        sess.headers["Connection"] = "close"
        if ca_cert:
            sess.verify = ca_cert

        try:
            # Choose action
            action = random.choices(
                ["get_health", "list", "create", "get", "update", "delete", "echo"],
                weights=[10, 15, 25, 15, 15, 10, 10],
                k=1,
            )[0]

            if action == "get_health":
                sess.get(f"{base_url}/health", timeout=10)
            elif action == "list":
                sess.get(f"{base_url}/items", timeout=10)
            elif action == "create":
                payload = _random_payload()
                resp = sess.post(f"{base_url}/items", json=payload, timeout=10)
                if resp.status_code == 201:
                    try:
                        item_ids.append(resp.json()["id"])
                    except Exception:
                        pass
            elif action == "get" and item_ids:
                iid = random.choice(item_ids)
                sess.get(f"{base_url}/items/{iid}", timeout=10)
            elif action == "update" and item_ids:
                iid = random.choice(item_ids)
                payload = _random_payload()
                sess.put(f"{base_url}/items/{iid}", json=payload, timeout=10)
            elif action == "delete" and item_ids:
                iid = item_ids.pop(random.randint(0, len(item_ids) - 1))
                sess.delete(f"{base_url}/items/{iid}", timeout=10)
            elif action == "echo":
                payload = _random_payload()
                sess.post(f"{base_url}/echo", json=payload, timeout=10)
            else:
                # Fallback: create
                payload = _random_payload()
                resp = sess.post(f"{base_url}/items", json=payload, timeout=10)
                if resp.status_code == 201:
                    try:
                        item_ids.append(resp.json()["id"])
                    except Exception:
                        pass
        except Exception as e:
            pass  # Silently continue on errors
        finally:
            sess.close()

        # Bursty timing: burst / normal / idle gap
        pattern = random.choices(["burst", "normal", "idle"], weights=[30, 50, 20], k=1)[0]
        if pattern == "burst":
            time.sleep(random.uniform(0.01, 0.05))
        elif pattern == "normal":
            time.sleep(random.uniform(0.1, 0.4))
        else:
            time.sleep(random.uniform(0.5, 1.5))

    print(f"  [HTTP Traffic] Completed {num_requests} requests")


def main():
    parser = argparse.ArgumentParser(description="HTTP Traffic Generator")
    parser.add_argument("--url", default="http://localhost:5000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--cert", default=None, help="CA cert for TLS")
    args = parser.parse_args()

    run_http_traffic(args.url, args.requests, ca_cert=args.cert)


if __name__ == "__main__":
    main()
