import json
import logging
import queue
import threading
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mcp.server.fastmcp import FastMCP

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


@app.on_event("startup")
async def startup_event():
    """Download all configured models on startup."""
    logger.info("Checking for model updates...")
    models = model_manager.list_available_models()
    for model_id in models:
        logger.info(f"Ensuring model {model_id} is downloaded...")
        try:
            model_manager.download_model(model_id)
        except Exception as e:
            logger.error(f"Failed to download {model_id} on startup: {e}")


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
        model_runner.unload_model(model_id)
        return f"Model {model_id} unloaded successfully."
    except Exception as e:
        return f"Error unloading model: {str(e)}"


@mcp.tool()
async def load_model(model_id: str, device: str = "GPU") -> str:
    """Loads a specific model into memory. Device can be 'GPU' or 'CPU'."""
    if not model_manager.is_model_downloaded(model_id):
        # Auto-download? For now, fail or trigger download if we want.
        # Let's try to download if not present?
        # Maybe too slow for a tool call without feedback.
        # Let's check if we can download.
        return f"Error: Model {model_id} not found locally. Use download_model first."

    config = model_manager.list_available_models()[model_id]
    model_path = config["local_path"]
    model_type = config.get("type", "vlm")

    try:
        model_runner.load_model(model_id, model_path, device, model_type)
        return f"Model {model_id} loaded successfully on {device}."
    except Exception as e:
        return f"Error loading model: {str(e)}"


@mcp.tool()
async def download_model_tool(model_id: str) -> str:
    """Downloads a model by ID."""
    try:
        model_manager.download_model(model_id)
        return f"Model {model_id} downloaded successfully."
    except Exception as e:
        return f"Error downloading model: {str(e)}"


@mcp.tool()
async def generate_text(model_id: str, prompt: str, max_tokens: int = 100) -> str:
    """Generates text using a loaded model."""
    if not model_runner.is_loaded(model_id):
        return f"Error: Model {model_id} is not loaded."

    # We use a queue to capture the stream for the tool output
    # This is a blocking call for the tool, essentially.
    result_text = []

    def streamer(subword):
        result_text.append(subword)
        return False

    try:
        # Run in a thread to allow streaming logic to work (though here we collect it)
        # Actually generate is blocking unless we thread it, but here we want to wait for result.
        # ModelRunner.generate is synchronous (blocking) in the current implementation
        # except for the threading used in the API endpoint.
        # Wait, ModelRunner.generate calls pipeline.generate which IS blocking.
        # The API endpoint uses a thread to run it and yields from a queue.
        # Here we can just run it.
        model_runner.generate(
            model_id, prompt, streamer=streamer, max_new_tokens=max_tokens
        )
        return "".join(result_text)
    except Exception as e:
        return f"Error generating text: {str(e)}"


# Mount MCP
# We assume FastMCP object is an ASGI app or has a mount method.
# If FastMCP is an app, we can mount it.
try:
    app.mount("/mcp", mcp)
except Exception as e:
    logger.error(f"Failed to mount MCP directly, trying .app or .http_app: {e}")
    # Fallback strategies if the library version differs
    if hasattr(mcp, "app"):
        app.mount("/mcp", mcp.app)
    elif hasattr(mcp, "http_app"):
        app.mount("/mcp", mcp.http_app)
    else:
        logger.error("Could not mount MCP. Please check mcp library version.")


# --- Standard API Endpoints ---


@app.get("/v1/models")
async def list_models_api():
    """OpenAI-compatible models list."""
    models = []
    config = model_manager.list_available_models()
    for mid, mdata in config.items():
        models.append(
            {
                "id": mid,
                "object": "model",
                "created": 0,
                "owned_by": "openvino",
                "permission": [],
                "root": mid,
                "parent": None,
            }
        )
    return {"data": models}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    """OpenAI-compatible chat completion."""
    body = await request.json()
    model_id = body.get("model", "gemma-3-4b-it-int4-ov")  # Default
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 1024)

    # Check if model is known
    config = model_manager.list_available_models()
    if model_id not in config:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Check if loaded, if not, try to load
    if not model_runner.is_loaded(model_id):
        # Auto-load logic
        path = config[model_id]["local_path"]
        mtype = config[model_id].get("type", "vlm")
        # Ensure downloaded
        if not model_manager.is_model_downloaded(model_id):
            raise HTTPException(
                status_code=400, detail=f"Model {model_id} not downloaded"
            )

        try:
            model_runner.load_model(model_id, path, device="GPU", model_type=mtype)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    prompt, image = model_runner.extract_image_and_text(messages)

    # Formatting logic specific to Gemma 3 or general?
    # For now, we apply the Gemma 3 hack if it's that model and has an image.
    # Ideally this should be in ModelRunner or a template manager.
    formatted_prompt = prompt
    if "gemma-3" in model_id and image is not None:
        formatted_prompt = f"user\n<image>\n{prompt}\nassistant\n"

    def event_generator():
        q = queue.Queue()

        def streamer(subword):
            q.put(subword)
            return False

        def run_generation():
            try:
                model_runner.generate(
                    model_id,
                    formatted_prompt,
                    image=image,
                    streamer=streamer,
                    max_new_tokens=max_tokens,
                )
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


# --- Admin / Management Endpoints ---


@app.post("/admin/download/{model_id}")
async def trigger_download(model_id: str):
    if model_id not in model_manager.list_available_models():
        raise HTTPException(status_code=404, detail="Model ID not found in config")

    # Run in background? For simplicity, synchronous for now or thread.
    # A proper job queue would be better.
    def download_task():
        model_manager.download_model(model_id)

    threading.Thread(target=download_task).start()
    return {"status": "Download started in background"}


@app.post("/admin/models")
async def add_model_config(request: Request):
    """Add a new model to configuration."""
    body = await request.json()
    model_id = body.get("model_id")
    repo_id = body.get("repo_id")
    local_path = body.get("local_path")
    model_type = body.get("type", "llm")

    if not all([model_id, repo_id, local_path]):
        raise HTTPException(status_code=400, detail="Missing fields")

    model_manager.add_model_config(model_id, repo_id, local_path, model_type)
    return {"status": "Model added"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
