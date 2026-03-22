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
        """Helper to extract conversation text and image from OpenAI chat format"""
        full_conversation = []
        ov_image = None
        image_found = False

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Text extraction
            msg_text = ""
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        msg_text = item["text"]
                    elif item["type"] == "image_url" and ov_image is None:
                        # Extract image if not already found
                        base_url = item["image_url"]["url"]
                        base64_data = (
                            base_url.split(",")[1] if "," in base_url else base_url
                        )
                        img_data = base64.b64decode(base64_data)
                        pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                        image_array = np.array(pil_img)
                        ov_image = ov.Tensor(image_array)
                        image_found = True
            else:
                msg_text = content

            # Format message for the prompt (simple concatenation for now)
            # For Gemma-3 VLMs, the <image> token should be placed in the user turn.
            if role == "system":
                full_conversation.append(f"system\n{msg_text}")
            elif role == "user":
                # Inject <image> token if it's the first time we find one in a user message
                if image_found:
                    full_conversation.append(f"user\n<image>\n{msg_text}")
                    image_found = False  # Only add once
                else:
                    full_conversation.append(f"user\n{msg_text}")
            elif role == "assistant":
                full_conversation.append(f"assistant\n{msg_text}")

        # Join conversation with role markers
        prompt = "\n".join(full_conversation)

        # Ensure we end with 'assistant\n' for models to start generating
        if not prompt.strip().endswith("assistant"):
            prompt += "\nassistant\n"

        return prompt, ov_image
