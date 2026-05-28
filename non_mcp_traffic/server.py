"""
Non-MCP Traffic Server — Flask HTTP REST API + WebSocket server.

Provides endpoints for generating non-MCP traffic that serves as the
negative class for the traffic classifier.

HTTP endpoints: /health, /items (CRUD), /echo
WebSocket: bidirectional message handler (ping/echo/time)

Usage:
    # Plain HTTP
    python -m non_mcp_traffic.server --http-port 5000 --ws-port 5001

    # HTTPS / TLS
    python -m non_mcp_traffic.server --http-port 5443 --ws-port 5001 --tls --cert certs/server.crt --key certs/server.key
"""

import argparse
import asyncio
import json
import ssl
import threading
import time
import uuid

from flask import Flask, jsonify, request

try:
    import websockets
except ImportError:
    websockets = None

app = Flask(__name__)

# In-memory data store
_items: dict[int, dict] = {}
_next_id = 1
_lock = threading.Lock()


def _new_item(data: dict) -> dict:
    """Create a new item with an auto-incrementing ID."""
    global _next_id
    with _lock:
        item_id = _next_id
        _next_id += 1
    item = {"id": item_id, **data, "created_at": time.time()}
    _items[item_id] = item
    return item


# ---------------------------------------------------------------------------
# HTTP REST endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": time.time()})


@app.route("/items", methods=["GET"])
def list_items():
    return jsonify({"items": list(_items.values()), "count": len(_items)})


@app.route("/items", methods=["POST"])
def create_item():
    data = request.get_json(force=True, silent=True) or {}
    item = _new_item(data)
    return jsonify(item), 201


@app.route("/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    item = _items.get(item_id)
    if item is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(item)


@app.route("/items/<int:item_id>", methods=["PUT"])
def update_item(item_id):
    item = _items.get(item_id)
    if item is None:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(force=True, silent=True) or {}
    item.update(data)
    item["updated_at"] = time.time()
    return jsonify(item)


@app.route("/items/<int:item_id>", methods=["DELETE"])
def delete_item(item_id):
    if item_id in _items:
        del _items[item_id]
        return jsonify({"deleted": item_id})
    return jsonify({"error": "not found"}), 404


@app.route("/echo", methods=["POST"])
def echo():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify({"echo": data, "timestamp": time.time()})


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def _ws_handler(websocket):
    """Handle WebSocket messages: ping→pong, echo→echo, time→timestamp."""
    async for message in websocket:
        try:
            data = json.loads(message)
            msg_type = data.get("type", "echo")

            if msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong", "ts": time.time()}))
            elif msg_type == "time":
                await websocket.send(json.dumps({
                    "type": "time",
                    "timestamp": time.time(),
                    "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }))
            else:
                await websocket.send(json.dumps({
                    "type": "echo",
                    "data": data.get("data", message),
                    "ts": time.time(),
                }))
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"type": "echo", "data": message}))


def _run_ws_server(host: str, port: int):
    """Run the WebSocket server in a new event loop (for threading)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def serve():
        async with websockets.serve(_ws_handler, host, port):
            print(f"[Non-MCP WS] Listening on ws://{host}:{port}")
            await asyncio.Future()  # run forever

    loop.run_until_complete(serve())


def _run_http_server(host: str, port: int, certfile: str = None, keyfile: str = None):
    """Run the Flask HTTP server, optionally with TLS."""
    if certfile and keyfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
        print(f"[Non-MCP HTTP] Listening on https://{host}:{port}")
        app.run(host=host, port=port, ssl_context=ctx, threaded=True, use_reloader=False)
    else:
        print(f"[Non-MCP HTTP] Listening on http://{host}:{port}")
        app.run(host=host, port=port, threaded=True, use_reloader=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Non-MCP Traffic Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=5000)
    parser.add_argument("--ws-port", type=int, default=5001)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--cert", default="certs/server.crt")
    parser.add_argument("--key", default="certs/server.key")
    args = parser.parse_args()

    # Start WebSocket server in a daemon thread
    if websockets:
        ws_thread = threading.Thread(
            target=_run_ws_server,
            args=(args.host, args.ws_port),
            daemon=True,
        )
        ws_thread.start()
    else:
        print("[Non-MCP WS] websockets not installed, skipping WS server")

    # Start HTTP server in the main thread
    certfile = args.cert if args.tls else None
    keyfile = args.key if args.tls else None
    _run_http_server(args.host, args.http_port, certfile, keyfile)


if __name__ == "__main__":
    main()
