"""
Diverse MCP Servers — multiple FastMCP server variants that simulate
real-world MCP server types with different tool signatures, response
sizes, and streaming patterns.

Each server type produces DIFFERENT network traffic patterns:
  - GitHub-like:     Large JSON responses (repo data, issues, PRs)
  - Filesystem-like: Variable responses (small ls, large file reads)
  - Fetch-like:      Web content retrieval (medium-large HTML/text)
  - Memory-like:     Knowledge graph (small structured responses)
  - Database-like:   SQL query results (tabular data, varying sizes)

Usage:
    python -m mcp_server.diverse_servers --type github --port 9001 --tls
    python -m mcp_server.diverse_servers --type filesystem --port 9002 --tls
    python -m mcp_server.diverse_servers --type all --base-port 9001 --tls
"""

import argparse
import json
import os
import random
import string
import sys
import time

from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------------
# GitHub-like MCP Server
# ---------------------------------------------------------------------------

def create_github_server(name: str = "github-mcp") -> FastMCP:
    """Simulates GitHub MCP server with repos, issues, PRs, code search."""
    server = FastMCP(name)

    REPOS = [
        {"name": "react", "owner": "facebook", "stars": 220000, "language": "JavaScript",
         "description": "A declarative, efficient, and flexible JavaScript library for building user interfaces."},
        {"name": "tensorflow", "owner": "tensorflow", "stars": 183000, "language": "Python",
         "description": "An Open Source Machine Learning Framework for Everyone"},
        {"name": "linux", "owner": "torvalds", "stars": 175000, "language": "C",
         "description": "Linux kernel source tree"},
        {"name": "vscode", "owner": "microsoft", "stars": 160000, "language": "TypeScript",
         "description": "Visual Studio Code - Open Source"},
        {"name": "fastapi", "owner": "tiangolo", "stars": 75000, "language": "Python",
         "description": "FastAPI framework, high performance, easy to learn"},
    ]

    ISSUES = [
        {"id": i, "title": f"Bug: {random.choice(['crash on','slow','error in','fix'])} {random.choice(['login','search','api','ui','build'])}",
         "state": random.choice(["open", "closed"]), "labels": random.sample(["bug", "enhancement", "help wanted", "good first issue", "documentation"], k=random.randint(1,3)),
         "body": "Lorem ipsum " * random.randint(10, 50), "comments": random.randint(0, 30)}
        for i in range(1, 21)
    ]

    @server.tool()
    def search_repositories(query: str, language: str = "", sort: str = "stars") -> str:
        """Search GitHub repositories by query and optional language filter."""
        results = [r for r in REPOS if query.lower() in r["name"].lower() or query.lower() in r["description"].lower()]
        if language:
            results = [r for r in results if r["language"].lower() == language.lower()]
        return json.dumps({"total_count": len(results), "items": results}, indent=2)

    @server.tool()
    def get_repository(owner: str, repo: str) -> str:
        """Get detailed information about a specific repository."""
        for r in REPOS:
            if r["owner"] == owner and r["name"] == repo:
                detail = {**r, "forks": r["stars"] // 10, "open_issues": random.randint(100, 5000),
                          "created_at": "2013-05-24T16:15:54Z", "updated_at": "2024-01-15T10:30:00Z",
                          "default_branch": "main", "topics": ["framework", "web", r["language"].lower()],
                          "readme": f"# {r['name']}\n\n{r['description']}\n\n## Installation\n\n```\npip install {r['name']}\n```\n\n" + "Documentation " * 50}
                return json.dumps(detail, indent=2)
        return json.dumps({"error": "Repository not found"})

    @server.tool()
    def list_issues(owner: str, repo: str, state: str = "open", per_page: int = 10) -> str:
        """List issues for a repository."""
        filtered = [i for i in ISSUES if i["state"] == state][:per_page]
        return json.dumps({"total_count": len(filtered), "issues": filtered}, indent=2)

    @server.tool()
    def search_code(query: str, language: str = "python") -> str:
        """Search code across repositories."""
        results = [{"file": f"src/{random.choice(['utils','main','config','api','models'])}.{language[:2]}",
                     "repo": random.choice(REPOS)["name"], "line": random.randint(1, 500),
                     "content": f"def {query.replace(' ','_')}({', '.join(random.sample(['x','y','data','config','opts'], k=2))}):\n    " + "pass\n" * random.randint(1,5)}
                    for _ in range(random.randint(3, 8))]
        return json.dumps({"total_count": len(results), "items": results}, indent=2)

    @server.tool()
    def get_pull_requests(owner: str, repo: str, state: str = "open") -> str:
        """List pull requests for a repository."""
        prs = [{"number": random.randint(1000, 9999), "title": f"{random.choice(['Fix','Add','Update','Refactor','Improve'])} {random.choice(['docs','tests','ci','api','core'])}",
                "state": state, "user": f"dev{random.randint(1,100)}", "additions": random.randint(1, 500),
                "deletions": random.randint(0, 200), "changed_files": random.randint(1, 20),
                "body": "This PR " + "changes things " * random.randint(5, 30)}
               for _ in range(random.randint(3, 10))]
        return json.dumps({"total_count": len(prs), "items": prs}, indent=2)

    return server


# ---------------------------------------------------------------------------
# Filesystem-like MCP Server
# ---------------------------------------------------------------------------

def create_filesystem_server(name: str = "filesystem-mcp") -> FastMCP:
    """Simulates Filesystem MCP server with file operations."""
    server = FastMCP(name)

    FILES = {
        "/home/user/project/main.py": "#!/usr/bin/env python3\n" + "import os\nimport sys\n\n" + "def main():\n    " + "print('hello')\n    " * 20 + "\n\nif __name__ == '__main__':\n    main()\n",
        "/home/user/project/config.yaml": "server:\n  host: 0.0.0.0\n  port: 8080\n  debug: false\n\ndatabase:\n  url: postgresql://localhost/mydb\n  pool_size: 10\n\nlogging:\n  level: INFO\n  file: /var/log/app.log\n",
        "/home/user/project/README.md": "# My Project\n\n" + "This is a sample project. " * 30 + "\n\n## Features\n\n- Feature 1\n- Feature 2\n- Feature 3\n",
        "/home/user/data/results.csv": "id,name,value,timestamp\n" + "\n".join([f"{i},item_{i},{random.random():.4f},{int(time.time())+i}" for i in range(100)]),
        "/home/user/notes.txt": "Meeting notes:\n" + "- Discussed project timeline\n" * 15,
    }

    @server.tool()
    def read_file(path: str) -> str:
        """Read the contents of a file at the given path."""
        if path in FILES:
            return FILES[path]
        return f"Error: File not found: {path}"

    @server.tool()
    def list_directory(path: str) -> str:
        """List files and directories at the given path."""
        entries = []
        for fpath in FILES:
            if fpath.startswith(path):
                rel = fpath[len(path):].lstrip("/")
                parts = rel.split("/")
                entry = parts[0]
                is_dir = len(parts) > 1
                entries.append({"name": entry, "type": "directory" if is_dir else "file",
                                "size": len(FILES.get(fpath, "")) if not is_dir else None})
        return json.dumps({"path": path, "entries": list({e["name"]: e for e in entries}.values())}, indent=2)

    @server.tool()
    def write_file(path: str, content: str) -> str:
        """Write content to a file at the given path."""
        FILES[path] = content
        return json.dumps({"status": "success", "path": path, "bytes_written": len(content)})

    @server.tool()
    def search_files(pattern: str, directory: str = "/home/user") -> str:
        """Search for files matching a pattern."""
        matches = [{"path": p, "size": len(c), "matches": c.count(pattern)}
                   for p, c in FILES.items() if pattern.lower() in c.lower() and p.startswith(directory)]
        return json.dumps({"pattern": pattern, "matches": matches}, indent=2)

    @server.tool()
    def get_file_info(path: str) -> str:
        """Get metadata about a file."""
        if path in FILES:
            content = FILES[path]
            return json.dumps({"path": path, "size": len(content), "lines": content.count("\n") + 1,
                                "modified": "2024-01-15T10:30:00Z", "permissions": "rw-r--r--"})
        return json.dumps({"error": f"File not found: {path}"})

    return server


# ---------------------------------------------------------------------------
# Fetch-like MCP Server
# ---------------------------------------------------------------------------

def create_fetch_server(name: str = "fetch-mcp") -> FastMCP:
    """Simulates web Fetch MCP server that retrieves URL content."""
    server = FastMCP(name)

    PAGES = {
        "https://example.com": "<html><head><title>Example</title></head><body><h1>Example Domain</h1>" + "<p>This domain is for use in illustrative examples.</p>" * 10 + "</body></html>",
        "https://api.github.com": json.dumps({"current_user_url": "https://api.github.com/user", "repository_url": "https://api.github.com/repos/{owner}/{repo}"}, indent=2),
        "https://httpbin.org/json": json.dumps({"slideshow": {"title": "Sample Slide Show", "author": "Test", "slides": [{"title": f"Slide {i}", "items": [f"Item {j}" for j in range(5)]} for i in range(10)]}}, indent=2),
        "https://news.ycombinator.com": "<html><body>" + "".join([f"<tr><td>{i}.</td><td><a href='https://example.com/{i}'>Article Title {i}: {'Lorem ipsum ' * random.randint(3,8)}</a><span class='score'>{random.randint(10,500)} points</span></td></tr>" for i in range(30)]) + "</body></html>",
    }

    @server.tool()
    def fetch_url(url: str, max_length: int = 5000) -> str:
        """Fetch the contents of a URL and return as text."""
        for page_url, content in PAGES.items():
            if url.startswith(page_url):
                return content[:max_length]
        # Generate plausible content for unknown URLs
        return f"<html><head><title>Page at {url}</title></head><body>" + "<p>Content paragraph. " * random.randint(20, 100) + "</body></html>"

    @server.tool()
    def fetch_json(url: str) -> str:
        """Fetch a URL and parse the JSON response."""
        data = {"url": url, "status": 200, "headers": {"content-type": "application/json"},
                "data": {"items": [{"id": i, "value": ''.join(random.choices(string.ascii_lowercase, k=20))} for i in range(random.randint(5, 20))]}}
        return json.dumps(data, indent=2)

    @server.tool()
    def extract_links(url: str) -> str:
        """Extract all links from a web page."""
        links = [{"href": f"https://example.com/{random.choice(['about','contact','blog','docs','api'])}/{i}",
                  "text": f"Link {i}: {'Sample ' * random.randint(1,5)}text"} for i in range(random.randint(5, 25))]
        return json.dumps({"url": url, "links": links}, indent=2)

    return server


# ---------------------------------------------------------------------------
# Memory/Knowledge-Graph-like MCP Server
# ---------------------------------------------------------------------------

def create_memory_server(name: str = "memory-mcp") -> FastMCP:
    """Simulates Memory/Knowledge Graph MCP server."""
    server = FastMCP(name)

    ENTITIES = {}
    RELATIONS = []

    @server.tool()
    def create_entity(name: str, entity_type: str, observations: list[str] = []) -> str:
        """Create a new entity in the knowledge graph."""
        ENTITIES[name] = {"type": entity_type, "observations": observations or [], "created": time.time()}
        return json.dumps({"status": "created", "entity": name, "type": entity_type})

    @server.tool()
    def add_observation(entity_name: str, observation: str) -> str:
        """Add an observation to an existing entity."""
        if entity_name in ENTITIES:
            ENTITIES[entity_name]["observations"].append(observation)
            return json.dumps({"status": "added", "entity": entity_name, "total_observations": len(ENTITIES[entity_name]["observations"])})
        return json.dumps({"error": f"Entity not found: {entity_name}"})

    @server.tool()
    def create_relation(from_entity: str, to_entity: str, relation_type: str) -> str:
        """Create a relation between two entities."""
        RELATIONS.append({"from": from_entity, "to": to_entity, "type": relation_type})
        return json.dumps({"status": "created", "relation": f"{from_entity} --{relation_type}--> {to_entity}"})

    @server.tool()
    def search_entities(query: str) -> str:
        """Search for entities matching a query."""
        matches = {k: v for k, v in ENTITIES.items() if query.lower() in k.lower() or any(query.lower() in o.lower() for o in v.get("observations", []))}
        return json.dumps({"query": query, "results": matches}, indent=2)

    @server.tool()
    def get_graph() -> str:
        """Get the entire knowledge graph."""
        return json.dumps({"entities": ENTITIES, "relations": RELATIONS, "stats": {"entity_count": len(ENTITIES), "relation_count": len(RELATIONS)}}, indent=2)

    return server


# ---------------------------------------------------------------------------
# Database-like MCP Server
# ---------------------------------------------------------------------------

def create_database_server(name: str = "database-mcp") -> FastMCP:
    """Simulates SQL Database MCP server."""
    server = FastMCP(name)

    TABLES = {
        "users": [{"id": i, "name": f"user_{i}", "email": f"user{i}@example.com", "role": random.choice(["admin","user","moderator"]), "created": "2024-01-01"} for i in range(1, 51)],
        "orders": [{"id": i, "user_id": random.randint(1,50), "product": random.choice(["laptop","phone","tablet","headphones","monitor"]), "amount": round(random.uniform(10,2000),2), "status": random.choice(["pending","shipped","delivered"])} for i in range(1, 101)],
        "logs": [{"id": i, "timestamp": f"2024-01-{random.randint(1,31):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:00Z", "level": random.choice(["INFO","WARN","ERROR"]), "message": f"Event {i}: " + "log entry " * random.randint(2,10)} for i in range(1, 201)],
    }

    @server.tool()
    def query(sql: str) -> str:
        """Execute a SQL query and return results."""
        sql_lower = sql.lower().strip()
        for table_name, data in TABLES.items():
            if table_name in sql_lower:
                if "count" in sql_lower:
                    return json.dumps({"result": [{"count": len(data)}], "rows_affected": 1})
                limit = 20
                if "limit" in sql_lower:
                    try:
                        limit = int(sql_lower.split("limit")[-1].strip().split()[0])
                    except Exception:
                        pass
                return json.dumps({"result": data[:limit], "rows_affected": min(limit, len(data)), "columns": list(data[0].keys())}, indent=2)
        return json.dumps({"error": "Table not found", "available_tables": list(TABLES.keys())})

    @server.tool()
    def list_tables() -> str:
        """List all available database tables."""
        info = {name: {"row_count": len(data), "columns": list(data[0].keys())} for name, data in TABLES.items()}
        return json.dumps(info, indent=2)

    @server.tool()
    def describe_table(table_name: str) -> str:
        """Get schema information for a table."""
        if table_name in TABLES:
            sample = TABLES[table_name][0]
            schema = [{"column": k, "type": type(v).__name__, "sample": str(v)} for k, v in sample.items()]
            return json.dumps({"table": table_name, "schema": schema, "row_count": len(TABLES[table_name])}, indent=2)
        return json.dumps({"error": f"Table not found: {table_name}"})

    return server


# ---------------------------------------------------------------------------
# Server factory and main
# ---------------------------------------------------------------------------

SERVER_TYPES = {
    "github": create_github_server,
    "filesystem": create_filesystem_server,
    "fetch": create_fetch_server,
    "memory": create_memory_server,
    "database": create_database_server,
}


def run_server(server_type: str, port: int, tls: bool = True, cert: str = "certs/server.crt", key: str = "certs/server.key"):
    """Run a single MCP server."""
    creator = SERVER_TYPES.get(server_type)
    if not creator:
        print(f"Unknown server type: {server_type}. Available: {list(SERVER_TYPES.keys())}")
        sys.exit(1)

    server = creator()

    transport = "sse"
    kwargs = {"transport": transport, "host": "0.0.0.0", "port": port}
    if tls:
        kwargs["ssl_certfile"] = cert
        kwargs["ssl_keyfile"] = key

    print(f"[{server_type.upper()} MCP Server] Starting on port {port} (TLS={tls})")
    server.run(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Diverse MCP Servers")
    parser.add_argument("--type", choices=list(SERVER_TYPES.keys()) + ["all"], default="github")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--tls", action="store_true", default=True)
    parser.add_argument("--no-tls", dest="tls", action="store_false")
    parser.add_argument("--cert", default="certs/server.crt")
    parser.add_argument("--key", default="certs/server.key")
    args = parser.parse_args()

    run_server(args.type, args.port, args.tls, args.cert, args.key)


if __name__ == "__main__":
    main()
