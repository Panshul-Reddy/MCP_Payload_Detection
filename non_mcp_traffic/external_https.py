"""
External HTTPS Traffic Generator -- makes real HTTPS requests to public
REST APIs to produce realistic non-MCP encrypted traffic.

These are "hard negatives" for the classifier: real TLS-encrypted web
traffic that the model must distinguish from encrypted MCP traffic.

APIs used (all public, no auth required):
  - httpbin.org     -- echo service with various endpoints
  - jsonplaceholder.typicode.com -- fake REST API
  - api.publicapis.org -- public API directory
  - catfact.ninja   -- random facts API
  - official-joke-api.appspot.com -- joke API

Usage:
    python -m non_mcp_traffic.external_https --requests 30 --output-dir data/pcap_external
"""

import argparse
import json
import os
import random
import time

import requests


# Public APIs that need no authentication
ENDPOINTS = [
    # jsonplaceholder -- fake REST API (most diverse)
    {"url": "https://jsonplaceholder.typicode.com/posts", "method": "GET"},
    {"url": "https://jsonplaceholder.typicode.com/posts/1", "method": "GET"},
    {"url": "https://jsonplaceholder.typicode.com/posts/1/comments", "method": "GET"},
    {"url": "https://jsonplaceholder.typicode.com/comments", "method": "GET"},
    {"url": "https://jsonplaceholder.typicode.com/users", "method": "GET"},
    {"url": "https://jsonplaceholder.typicode.com/todos", "method": "GET"},
    {"url": "https://jsonplaceholder.typicode.com/albums", "method": "GET"},
    {"url": "https://jsonplaceholder.typicode.com/photos", "method": "GET"},
    {"url": "https://jsonplaceholder.typicode.com/posts", "method": "POST", "json": True},
    {"url": "https://jsonplaceholder.typicode.com/posts/1", "method": "PUT", "json": True},
    # dummyjson -- rich fake REST API
    {"url": "https://dummyjson.com/products", "method": "GET"},
    {"url": "https://dummyjson.com/products/1", "method": "GET"},
    {"url": "https://dummyjson.com/products/search?q=phone", "method": "GET"},
    {"url": "https://dummyjson.com/users", "method": "GET"},
    {"url": "https://dummyjson.com/users/1", "method": "GET"},
    {"url": "https://dummyjson.com/carts", "method": "GET"},
    {"url": "https://dummyjson.com/quotes", "method": "GET"},
    {"url": "https://dummyjson.com/recipes", "method": "GET"},
    {"url": "https://dummyjson.com/posts", "method": "GET"},
    {"url": "https://dummyjson.com/comments", "method": "GET"},
    {"url": "https://dummyjson.com/products/add", "method": "POST", "json": True},
    # catfact.ninja
    {"url": "https://catfact.ninja/fact", "method": "GET"},
    {"url": "https://catfact.ninja/facts", "method": "GET"},
    {"url": "https://catfact.ninja/breeds", "method": "GET"},
    # Dog API
    {"url": "https://dog.ceo/api/breeds/list/all", "method": "GET"},
    {"url": "https://dog.ceo/api/breeds/image/random", "method": "GET"},
    {"url": "https://dog.ceo/api/breed/hound/images/random", "method": "GET"},
    # IP / utility APIs
    {"url": "https://api.ipify.org?format=json", "method": "GET"},
]

SAMPLE_DATA = [
    {"title": "Test post", "body": "This is a test", "userId": 1},
    {"title": "Analysis report", "body": "Network traffic classification results", "userId": 2},
    {"title": "ML experiment", "body": "Feature extraction from encrypted flows", "userId": 3},
    {"name": "test_item", "value": random.random(), "category": "network"},
    {"query": "encrypted traffic detection", "filters": {"protocol": "TLS", "version": "1.3"}},
]


def run_external_https(num_requests: int = 30, delay_range: tuple = (0.2, 1.5)) -> dict:
    """
    Make real HTTPS requests to public APIs.
    Returns stats about successful/failed requests.
    """
    stats = {"total": 0, "success": 0, "failed": 0, "endpoints_hit": set()}

    for i in range(num_requests):
        endpoint = random.choice(ENDPOINTS)
        url = endpoint["url"]
        method = endpoint["method"]

        try:
            # New session per request AND Connection: close -> separate TCP flow per request
            sess = requests.Session()
            sess.headers["Connection"] = "close"
            sess.headers["User-Agent"] = random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "MCP-Traffic-Analyzer/1.0",
                "Python-Requests/2.31",
                "curl/8.0",
            ])

            if method == "GET":
                resp = sess.get(url, timeout=10)
            elif method == "POST":
                data = random.choice(SAMPLE_DATA) if endpoint.get("json") else None
                resp = sess.post(url, json=data, timeout=10)
            elif method == "PUT":
                data = random.choice(SAMPLE_DATA) if endpoint.get("json") else None
                resp = sess.put(url, json=data, timeout=10)
            elif method == "DELETE":
                resp = sess.delete(url, timeout=10)
            else:
                resp = sess.get(url, timeout=10)

            stats["success"] += 1
            stats["endpoints_hit"].add(url.split("/")[2])  # domain
            sess.close()

        except Exception as e:
            stats["failed"] += 1

        stats["total"] += 1

        # Bursty timing
        pattern = random.choices(["burst", "normal", "slow"], weights=[25, 50, 25], k=1)[0]
        if pattern == "burst":
            time.sleep(random.uniform(0.05, 0.15))
        elif pattern == "normal":
            time.sleep(random.uniform(delay_range[0], delay_range[1]))
        else:
            time.sleep(random.uniform(1.0, 3.0))

        if (i + 1) % 10 == 0:
            print(f"  [External HTTPS] {i+1}/{num_requests} requests done")

    stats["endpoints_hit"] = list(stats["endpoints_hit"])
    print(f"  [External HTTPS] Done: {stats['success']}/{stats['total']} succeeded, "
          f"domains={stats['endpoints_hit']}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="External HTTPS Traffic Generator")
    parser.add_argument("--requests", type=int, default=30)
    args = parser.parse_args()

    print("[External HTTPS] Starting real-world HTTPS traffic generation...")
    stats = run_external_https(args.requests)
    print(f"[External HTTPS] Complete. Stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
