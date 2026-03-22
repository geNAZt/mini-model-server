# Mini Model Server

A lightweight, OpenVINO-powered model server with MCP integration and multi-model support.

## Features

- **Multi-Model Support (Poly):** Load and switch between multiple LLM/VLM models dynamically.
- **MCP Integration:** Exposes model capabilities (list, load, generate) as MCP tools for AI assistants.
- **OpenAI Compatible API:** Standard `/v1/chat/completions` endpoint.
- **Built-in Download Manager:** Auto-download models from Hugging Face.

## Setup

1. Create and activate a virtual environment (requires Python 3.10+):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure models in `models.json` (optional). Default includes `gemma-3-4b-it-int4-ov`.

## Usage

### Start the Server

Ensure your virtual environment is activated:
```bash
python main.py
```
The server runs on `http://0.0.0.0:8000`.

### Download Models

You can download models via the API or CLI:

**CLI:**
```bash
python download_models.py --list
python download_models.py --model gemma-3-4b-it-int4-ov
```

**API:**
POST `/admin/download/gemma-3-4b-it-int4-ov`

### API Endpoints

- **Chat Completions:** `POST /v1/chat/completions`
  - Body: `{"model": "gemma-3-4b-it-int4-ov", "messages": [...]}`
- **List Models:** `GET /v1/models`
- **MCP Endpoint:** `GET /mcp` (or SSE endpoint depending on client)

### MCP Client Configuration (e.g., Claude Desktop)

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mini-model-server": {
      "command": "uv",
      "args": [
        "run",
        "--with", "mcp",
        "mcp", "connect", "http://localhost:8000/mcp/sse"
      ]
    }
  }
}
```
*Note: Adjust the connection command based on how your `mcp` client connects to HTTP SSE.*
