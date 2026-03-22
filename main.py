import time

print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Starting imports...")
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

print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Imports complete. Loading managers...")
from model_manager import ModelManager
from model_runner import ModelRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Initializing ModelManager...")
model_manager = ModelManager()
print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Initializing ModelRunner...")
model_runner = ModelRunner()

print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Creating FastAPI app...")
app = FastAPI(title="Mini Model Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Initializing FastMCP...")
mcp = FastMCP("MiniModelServer")


async def get_mcp_tools_description(mcp_instance: FastMCP) -> str:
    logger.info("DEBUG: Entering get_mcp_tools_description")
    try:
        tools = await mcp_instance.list_tools()
        logger.info(f"DEBUG: Found {len(tools)} tools")
        descriptions = [
            f"- {t.name}: {t.description or 'No description'}" for t in tools
        ]
        return "\n".join(descriptions)
    except Exception as e:
        logger.error(f"DEBUG: Error in get_mcp_tools_description: {e}")
        return "Error retrieving tool list."


@app.on_event("startup")
async def startup_event():
    logger.info("EVENT: Startup triggered")

    def background_download():
        logger.info("THREAD: Background download thread started")
        try:
            models = model_manager.list_available_models()
            for model_id in models:
                logger.info(f"THREAD: Checking/Downloading {model_id}...")
                model_manager.download_model(model_id)
                logger.info(f"THREAD: Done with {model_id}")
        except Exception as e:
            logger.error(f"THREAD ERROR: {e}")
        logger.info("THREAD: Background download thread finished")

    # Use standard threading to be 100% sure we don't block the async loop
    download_thread = threading.Thread(target=background_download, daemon=True)
    download_thread.start()
    logger.info("EVENT: Startup logic complete (downloads running in background)")


# --- MCP Tools ---


@mcp.tool()
async def list_available_models() -> Dict[str, Any]:
    return model_manager.list_available_models()


@mcp.tool()
async def list_loaded_models() -> List[str]:
    return model_runner.get_loaded_models()


@mcp.tool()
async def unload_model(model_id: str) -> str:
    logger.info(f"TOOL: Unloading {model_id}")
    await run_in_threadpool(model_runner.unload_model, model_id)
    return f"Model {model_id} unloaded."


@mcp.tool()
async def load_model(model_id: str, device: str = "GPU") -> str:
    logger.info(f"TOOL: Loading {model_id} on {device}")
    config = model_manager.list_available_models().get(model_id)
    if not config:
        return "Model not found"
    await run_in_threadpool(
        model_runner.load_model,
        model_id,
        config["local_path"],
        device,
        config.get("type", "vlm"),
    )
    return f"Loaded {model_id}"


@mcp.tool()
async def download_model_tool(model_id: str) -> str:
    logger.info(f"TOOL: Downloading {model_id}")
    await run_in_threadpool(model_manager.download_model, model_id)
    return f"Downloaded {model_id}"


@mcp.tool()
async def generate_text(model_id: str, prompt: str, max_tokens: int = 100) -> str:
    logger.info(f"TOOL: Generating text with {model_id}")
    result_text = []

    def streamer(t):
        result_text.append(t)
        return False

    await run_in_threadpool(
        model_runner.generate, model_id, prompt, None, streamer, max_tokens
    )
    return "".join(result_text)


print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Mounting MCP...")
try:
    app.mount("/mcp", mcp)
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: MCP mounted successfully.")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ERROR: Failed to mount MCP: {e}")


@app.get("/v1/models")
async def list_models_api():
    config = model_manager.list_available_models()
    return {"data": [{"id": mid, "object": "model"} for mid in config]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    logger.info("API: Chat completion request received")
    body = await request.json()
    model_id = body.get("model", "gemma-3-4b-it-int4-ov")
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 4096)

    logger.info("API: Building system prompt...")
    tools_list = await get_mcp_tools_description(mcp)
    system_prompt = f"You are Mini Model Server.\nTools:\n{tools_list}"

    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": system_prompt})

    config = model_manager.list_available_models()
    if model_id not in config:
        raise HTTPException(status_code=404, detail="Model not found")

    if not model_runner.is_loaded(model_id):
        logger.info(f"API: Model {model_id} not loaded. Triggering load...")
        path = config[model_id]["local_path"]
        mtype = config[model_id].get("type", "vlm")
        await run_in_threadpool(model_runner.load_model, model_id, path, "GPU", mtype)
        logger.info(f"API: Model {model_id} load complete")

    prompt, image = model_runner.extract_image_and_text(messages)
    logger.info(f"--- PROMPT DEBUG ---\n{prompt}\n--- END ---")

    def event_generator():
        q = queue.Queue()

        def streamer(t):
            q.put(t)
            return False

        def run_gen():
            try:
                logger.info("GEN: Starting generation...")
                model_runner.generate(model_id, prompt, image, streamer, max_tokens)
                logger.info("GEN: Generation finished")
            except Exception as e:
                logger.error(f"GEN ERROR: {e}")
                q.put(f"[Error: {e}]")
            q.put(None)

        threading.Thread(target=run_gen).start()
        while True:
            token = q.get()
            if token is None:
                yield "data: [DONE]\n\n"
                break
            yield f"data: {json.dumps({'choices': [{'delta': {'content': token}, 'index': 0}]})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Launching Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
