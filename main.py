import json
import logging
import queue
import threading
from typing import Any, Dict, List, Optional

import anyio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mcp.server.fastmcp import FastMCP
from starlette.concurrency import run_in_threadpool

from model_manager import ModelManager
from model_runner import ModelRunner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize managers
model_manager = ModelManager()
model_runner = ModelRunner()

# Initialize FastAPI
app = FastAPI(title="Mini Model Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MCP
mcp = FastMCP("MiniModelServer")


async def get_mcp_tools_description(mcp_instance: FastMCP) -> str:
    """Dynamically retrieves descriptions for all registered tools."""
    try:
        # FastMCP.list_tools() returns the registered tool objects
        tools = await mcp_instance.list_tools()
        if not tools:
            return "No tools currently registered."

        descriptions = []
        for tool in tools:
            desc = f"- {tool.name}: {tool.description or 'No description provided.'}"
            descriptions.append(desc)
        return "\n".join(descriptions)
    except Exception as e:
        logger.error(f"Error dynamically building tool list: {e}")
        return "Error retrieving tool list."


@app.on_event("startup")
async def startup_event():
    """Download all configured models on startup without blocking the loop."""

    async def download_all():
        logger.info("Checking for model updates...")
        models = model_manager.list_available_models()
        for model_id in models:
            logger.info(f"Ensuring model {model_id} is downloaded...")
            try:
                await run_in_threadpool(model_manager.download_model, model_id)
            except Exception as e:
                logger.error(f"Failed to download {model_id} on startup: {e}")

    # Kick off downloads in background task group
    # We use a simple background task instead of full task group if we want simpler
    # But this is safe
    async def bg_runner():
        async with anyio.create_task_group() as tg:
            tg.start_soon(download_all)

    threading.Thread(target=anyio.run, args=(bg_runner,), daemon=True).start()


# --- MCP Tools ---


@mcp.tool()
async def list_available_models() -> Dict[str, Any]:
    """Lists all models configured in the server."""
    return model_manager.list_available_models()


@mcp.tool()
async def list_loaded_models() -> List[str]:
    """Lists models currently loaded in memory."""
    return model_runner.get_loaded_models()


@mcp.tool()
async def unload_model(model_id: str) -> str:
    """Unloads a model from memory to free up RAM."""
    if not model_runner.is_loaded(model_id):
        return f"Model {model_id} is not loaded."
    try:
        # Unloading is fast but still better in thread if we want to be safe
        await run_in_threadpool(model_runner.unload_model, model_id)
        return f"Model {model_id} unloaded successfully."
    except Exception as e:
        return f"Error unloading model: {str(e)}"


@mcp.tool()
async def load_model(model_id: str, device: str = "GPU") -> str:
    """Loads a specific model into memory. Device can be 'GPU' or 'CPU'."""
    if not model_manager.is_model_downloaded(model_id):
        return f"Error: Model {model_id} not found locally. Use download_model first."

    config = model_manager.list_available_models()[model_id]
    model_path = config["local_path"]
    model_type = config.get("type", "vlm")

    try:
        await run_in_threadpool(
            model_runner.load_model, model_id, model_path, device, model_type
        )
        return f"Model {model_id} loaded successfully on {device}."
    except Exception as e:
        return f"Error loading model: {str(e)}"


@mcp.tool()
async def download_model_tool(model_id: str) -> str:
    """Downloads a model by ID."""
    try:
        await run_in_threadpool(model_manager.download_model, model_id)
        return f"Model {model_id} downloaded successfully."
    except Exception as e:
        return f"Error downloading model: {str(e)}"


@mcp.tool()
async def generate_text(model_id: str, prompt: str, max_tokens: int = 100) -> str:
    """Generates text using a loaded model."""
    if not model_runner.is_loaded(model_id):
        return f"Error: Model {model_id} is not loaded."

    result_text = []

    def streamer(subword):
        result_text.append(subword)
        return False

    try:
        await run_in_threadpool(
            model_runner.generate, model_id, prompt, None, streamer, max_tokens
        )
        return "".join(result_text)
    except Exception as e:
        return f"Error generating text: {str(e)}"


# Mount MCP
try:
    app.mount("/mcp", mcp)
except Exception as e:
    logger.error(f"Failed to mount MCP directly: {e}")
    if hasattr(mcp, "app"):
        app.mount("/mcp", mcp.app)
    elif hasattr(mcp, "http_app"):
        app.mount("/mcp", mcp.http_app)

# --- Standard API Endpoints ---


@app.get("/v1/models")
async def list_models_api():
    """OpenAI-compatible models list."""
    models = []
    config = model_manager.list_available_models()
    for mid, mdata in config.items():
        models.append(
            {"id": mid, "object": "model", "created": 0, "owned_by": "openvino"}
        )
    return {"data": models}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    """OpenAI-compatible chat completion."""
    body = await request.json()
    model_id = body.get("model", "gemma-3-4b-it-int4-ov")
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 4096)

    # Context awareness system prompt
    tools_list = await get_mcp_tools_description(mcp)
    system_prompt = f"""You are the AI assistant for the 'Mini Model Server'.
Your underlying system supports the Model Context Protocol (MCP).
When asked about your tools or MCP, you should refer to the following available tools:
{tools_list}

Do not confuse MCP with 'Master Control Program' or other legacy systems. You are a modern AI assistant."""

    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": system_prompt})

    config = model_manager.list_available_models()
    if model_id not in config:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    if not model_runner.is_loaded(model_id):
        path = config[model_id]["local_path"]
        mtype = config[model_id].get("type", "vlm")
        if not model_manager.is_model_downloaded(model_id):
            raise HTTPException(
                status_code=400, detail=f"Model {model_id} not downloaded"
            )

        try:
            logger.info(f"Auto-loading model {model_id} in background thread...")
            await run_in_threadpool(
                model_runner.load_model, model_id, path, "GPU", mtype
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    prompt, image = model_runner.extract_image_and_text(messages)
    logger.info(f"--- FULL PROMPT DEBUG ---\n{prompt}\n--- END PROMPT DEBUG ---")

    def event_generator():
        q = queue.Queue()

        def streamer(subword):
            q.put(subword)
            return False

        def run_generation():
            try:
                model_runner.generate(model_id, prompt, image, streamer, max_tokens)
            except Exception as e:
                logger.error(f"Generation error: {e}")
                q.put(f"\n[Error: {str(e)}]")
            q.put(None)

        threading.Thread(target=run_generation).start()
        while True:
            token = q.get()
            if token is None:
                yield "data: [DONE]\n\n"
                break
            chunk = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model_id,
                "choices": [
                    {"delta": {"content": token}, "index": 0, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
