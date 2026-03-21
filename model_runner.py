import base64
import io
import logging
import queue
import threading

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(self):
        self.pipelines = {}  # model_id -> pipeline
        self.configs = {}  # model_id -> config
        self.lock = threading.Lock()

    def load_model(
        self,
        model_id: str,
        model_path: str,
        device: str = "GPU",
        model_type: str = "vlm",
    ):
        with self.lock:
            if model_id in self.pipelines:
                return

            logger.info(
                f"Loading {model_type} model '{model_id}' from {model_path} on {device}..."
            )
            try:
                if model_type == "vlm":
                    pipeline = ov_genai.VLMPipeline(model_path, device)
                elif model_type == "llm":
                    pipeline = ov_genai.LLMPipeline(model_path, device)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                self.pipelines[model_id] = pipeline
                self.configs[model_id] = {
                    "type": model_type,
                    "path": model_path,
                    "device": device,
                }
                logger.info(f"Model '{model_id}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model '{model_id}': {e}")
                raise e

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self.pipelines

    def unload_model(self, model_id: str):
        with self.lock:
            if model_id in self.pipelines:
                logger.info(f"Unloading model '{model_id}'...")
                del self.pipelines[model_id]
                del self.configs[model_id]
                import gc

                gc.collect()
                logger.info(f"Model '{model_id}' unloaded.")

    def get_loaded_models(self):
        return list(self.pipelines.keys())

    def generate(
        self,
        model_id: str,
        prompt: str,
        image: ov.Tensor = None,
        streamer=None,
        max_new_tokens=1024,
    ):
        if model_id not in self.pipelines:
            raise ValueError(f"Model '{model_id}' is not loaded.")

        pipeline = self.pipelines[model_id]
        model_type = self.configs[model_id]["type"]

        # Generate logic based on type
        if model_type == "vlm":
            if image is not None:
                pipeline.generate(
                    prompt,
                    image=image,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                )
            else:
                pipeline.generate(
                    prompt, max_new_tokens=max_new_tokens, streamer=streamer
                )
        else:
            pipeline.generate(prompt, max_new_tokens=max_new_tokens, streamer=streamer)

    def extract_image_and_text(self, messages):
        """Helper to extract text and image from OpenAI chat format"""
        prompt = ""
        ov_image = None

        last_msg = messages[-1]
        content = last_msg.get("content")

        if isinstance(content, list):
            for item in content:
                if item["type"] == "text":
                    prompt = item["text"]
                elif item["type"] == "image_url":
                    # Base64 decode
                    base_url = item["image_url"]["url"]
                    base64_data = (
                        base_url.split(",")[1] if "," in base_url else base_url
                    )
                    img_data = base64.b64decode(base64_data)
                    pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    image_array = np.array(pil_img)
                    ov_image = ov.Tensor(image_array)
        else:
            prompt = content

        return prompt, ov_image
