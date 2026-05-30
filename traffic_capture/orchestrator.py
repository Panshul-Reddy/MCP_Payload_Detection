"""
Pipeline Orchestrator — end-to-end coordinator that starts all servers,
runs traffic generators, captures packets, and shuts everything down.

Usage:
    # Plain HTTP
    python -m traffic_capture.orchestrator --duration 60 --requests 50 --output-dir data/pcap

    # HTTPS / TLS
    python -m traffic_capture.orchestrator --duration 60 --requests 50 --tls --output-dir data/pcap
"""

import argparse
import asyncio
import os
import platform
import random
import signal
import socket
import struct
import subprocess
import sys
import time

from scapy.all import get_if_list


def _find_free_port(start: int, end: int, exclude: set) -> int:
    """Find a free TCP port in the given range, excluding specified ports."""
    candidates = list(set(range(start, end)) - exclude)
    random.shuffle(candidates)
    for port in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------

def _python_exe():
    """Return the path to the current Python interpreter."""
    return sys.executable


def _start(args: list[str], label: str = "") -> subprocess.Popen:
    """Start a subprocess, creating a process group on Unix."""
    kwargs = {}
    if platform.system() != "Windows":
        kwargs["preexec_fn"] = os.setsid
    else:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    print(f"[Orchestrator] Started: {' '.join(args[:6])}{'...' if len(args) > 6 else ''} (PID {proc.pid})")
    return proc


def _stop(proc: subprocess.Popen, name: str = "") -> None:
    """Gracefully stop a subprocess."""
    if proc.poll() is not None:
        return
    try:
        if platform.system() == "Windows":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            pass
    print(f"[Orchestrator] Stopped: {name} (PID {proc.pid})")


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    """Wait until a TCP port is accepting connections."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


# ---------------------------------------------------------------------------
# Inline TCP echo server
# ---------------------------------------------------------------------------

def _tcp_echo_handler(reader, writer):
    """Handle one TCP connection: read framed messages, echo them back."""
    async def handle():
        try:
            while True:
                header = await asyncio.wait_for(reader.read(5), timeout=30.0)
                if len(header) < 5:
                    break
                msg_type, payload_len = struct.unpack("!BI", header)
                if payload_len > 65536:
                    break
                payload = await asyncio.wait_for(
                    reader.read(payload_len), timeout=10.0
                )
                # Echo back with PONG type if it was a PING
                resp_type = 0x04 if msg_type == 0x03 else msg_type
                resp = struct.pack("!BI", resp_type, len(payload)) + payload
                writer.write(resp)
                await writer.drain()
        except (asyncio.TimeoutError, ConnectionResetError, Exception):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    return handle()


def _run_tcp_echo_server(host: str, port: int):
    """Run the TCP echo server (blocking, for subprocess)."""
    async def serve():
        server = await asyncio.start_server(
            lambda r, w: _tcp_echo_handler(r, w),
            host, port,
        )
        print(f"[TCP Echo] Listening on {host}:{port}")
        async with server:
            await server.serve_forever()

    asyncio.run(serve())


def _start_tcp_echo_server(host: str, port: int) -> subprocess.Popen:
    """Start the TCP echo server as a subprocess."""
    return _start([
        _python_exe(), "-c",
        f"import sys; sys.path.insert(0, '.'); "
        f"from traffic_capture.orchestrator import _run_tcp_echo_server; "
        f"_run_tcp_echo_server('{host}', {port})",
    ], "TCP Echo Server")


# ---------------------------------------------------------------------------
# Loopback interface detection
# ---------------------------------------------------------------------------

def _default_loopback_interface() -> str:
    """Auto-detect the loopback interface."""
    system = platform.system()
    if system == "Darwin":
        return "lo0"
    elif system == "Linux":
        return "lo"
    else:  # Windows
        ifaces = get_if_list()
        for iface in ifaces:
            if "loopback" in iface.lower() or "NPF_Loopback" in iface:
                return iface
        return "\\Device\\NPF_Loopback"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    duration: int = 60,
    num_requests: int = 50,
    mcp_sessions: int = 3,
    ws_sessions: int = 3,
    tcp_connections: int = 3,
    interface: str | None = None,
    output_dir: str = "data/pcap",
    capture: bool = True,
    tls: bool = False,
    cert: str = "certs/server.crt",
    key: str = "certs/server.key",
    randomize_ports: bool = True,
) -> None:
    """Run the full traffic generation + capture pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    interface = interface or _default_loopback_interface()
    py = _python_exe()

    # Determine ports -- randomize to prevent port-based leakage
    used_ports = set()
    if randomize_ports:
        mcp_port = _find_free_port(7000, 9500, used_ports)
        used_ports.add(mcp_port)
        http_port = _find_free_port(7000, 9500, used_ports)
        used_ports.add(http_port)
        ws_port = _find_free_port(7000, 9500, used_ports)
        used_ports.add(ws_port)
        tcp_port = _find_free_port(7000, 9500, used_ports)
        used_ports.add(tcp_port)
        print(f"[Orchestrator] Randomized ports: MCP={mcp_port}, HTTP={http_port}, WS={ws_port}, TCP={tcp_port}")
    else:
        if tls:
            mcp_port = 8443
            http_port = 5443
        else:
            mcp_port = 8000
            http_port = 5000
        ws_port = 5001
        tcp_port = 5002

    procs = []

    try:
        # --- Start servers ---
        print("[Orchestrator] Starting servers...")

        # MCP server
        mcp_server_args = [py, "-m", "mcp_server.server", "--port", str(mcp_port)]
        if tls:
            mcp_server_args += ["--tls", "--cert", cert, "--key", key]
        mcp_proc = _start(mcp_server_args, "MCP Server")
        procs.append((mcp_proc, "MCP Server"))

        # Non-MCP HTTP + WS server
        non_mcp_args = [
            py, "-m", "non_mcp_traffic.server",
            "--http-port", str(http_port),
            "--ws-port", str(ws_port),
        ]
        if tls:
            non_mcp_args += ["--tls", "--cert", cert, "--key", key]
        non_mcp_proc = _start(non_mcp_args, "Non-MCP Server")
        procs.append((non_mcp_proc, "Non-MCP Server"))

        # TCP echo server
        tcp_proc = _start_tcp_echo_server("0.0.0.0", tcp_port)
        procs.append((tcp_proc, "TCP Echo Server"))

        # --- Wait for servers ---
        print("[Orchestrator] Waiting for servers...")
        for port_check, name in [(mcp_port, "MCP"), (http_port, "HTTP"), (tcp_port, "TCP")]:
            if _wait_for_port("localhost", port_check):
                print(f"  [+] {name} server ready on port {port_check}")
            else:
                print(f"  [!] {name} server NOT ready on port {port_check} (continuing anyway)")

        # Extra wait for stability
        time.sleep(2)

        # --- Start capture ---
        mcp_ports = [mcp_port]
        non_mcp_ports = [http_port, ws_port, tcp_port]

        if capture:
            from traffic_capture.capture import capture_traffic
            import threading

            capture_result = [None, None]

            def do_capture():
                result = capture_traffic(
                    interface=interface,
                    mcp_ports=mcp_ports,
                    non_mcp_ports=non_mcp_ports,
                    duration=duration + 5,  # Extra buffer
                    output_dir=output_dir,
                )
                capture_result[0], capture_result[1] = result

            capture_thread = threading.Thread(target=do_capture, daemon=True)
            capture_thread.start()

            # Small delay so capture starts before traffic
            time.sleep(2)

        # --- Generate traffic ---
        print(f"[Orchestrator] Generating traffic for {duration}s...")

        protocol = "https" if tls else "http"
        cert_args = ["--cert", cert] if tls else []

        generators = []

        # MCP client
        mcp_client_args = [
            py, "-m", "mcp_client.client",
            "--url", f"{protocol}://localhost:{mcp_port}/sse",
            "--sessions", str(mcp_sessions),
            "--requests", str(num_requests),
        ] + cert_args
        generators.append((_start(mcp_client_args, "MCP Client"), "MCP Client"))

        # HTTP traffic
        http_url = f"{protocol}://localhost:{http_port}"
        http_args = [
            py, "-m", "non_mcp_traffic.http_traffic",
            "--url", http_url,
            "--requests", str(num_requests),
        ] + cert_args
        generators.append((_start(http_args, "HTTP Traffic"), "HTTP Traffic"))

        # WebSocket traffic
        ws_args = [
            py, "-m", "non_mcp_traffic.websocket_traffic",
            "--url", f"ws://localhost:{ws_port}",
            "--sessions", str(ws_sessions),
            "--messages", str(num_requests),
        ]
        generators.append((_start(ws_args, "WS Traffic"), "WS Traffic"))

        # TCP traffic
        tcp_args = [
            py, "-m", "non_mcp_traffic.tcp_traffic",
            "--host", "localhost",
            "--port", str(tcp_port),
            "--connections", str(tcp_connections),
            "--messages", str(num_requests),
        ]
        generators.append((_start(tcp_args, "TCP Traffic"), "TCP Traffic"))

        # Wait for generators to finish (up to duration + buffer)
        deadline = time.time() + duration + 30
        for gen_proc, gen_name in generators:
            remaining = max(0, deadline - time.time())
            try:
                gen_proc.wait(timeout=remaining)
                print(f"[Orchestrator] {gen_name} finished")
            except subprocess.TimeoutExpired:
                print(f"[Orchestrator] {gen_name} timed out, killing")
                _stop(gen_proc, gen_name)

        # Wait for capture to finish
        if capture:
            print("[Orchestrator] Waiting for capture to complete...")
            capture_thread.join(timeout=duration + 15)

        print("[Orchestrator] Pipeline complete")

    finally:
        # --- Cleanup ---
        print("[Orchestrator] Stopping servers...")
        for proc, name in procs:
            _stop(proc, name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Traffic Pipeline Orchestrator")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--mcp-sessions", type=int, default=3)
    parser.add_argument("--ws-sessions", type=int, default=3)
    parser.add_argument("--tcp-connections", type=int, default=3)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--output-dir", default="data/pcap")
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--cert", default="certs/server.crt")
    parser.add_argument("--key", default="certs/server.key")
    args = parser.parse_args()

    run_pipeline(
        duration=args.duration,
        num_requests=args.requests,
        mcp_sessions=args.mcp_sessions,
        ws_sessions=args.ws_sessions,
        tcp_connections=args.tcp_connections,
        interface=args.interface,
        output_dir=args.output_dir,
        capture=not args.no_capture,
        tls=args.tls,
        cert=args.cert,
        key=args.key,
    )


if __name__ == "__main__":
    main()
