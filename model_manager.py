import json
import logging
import os
import shutil
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_CONFIG_FILE = "models.json"


class ModelManager:
    def __init__(self, config_file: str = MODELS_CONFIG_FILE):
        self.config_file = config_file
        self.models_config = self._load_config()
        self.loaded_models = {}

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_file):
            logger.warning(
                f"Config file {self.config_file} not found. Creating default."
            )
            return {}
        with open(self.config_file, "r") as f:
            return json.load(f)

    def save_config(self):
        with open(self.config_file, "w") as f:
            json.dump(self.models_config, f, indent=4)

    def list_available_models(self) -> Dict[str, Any]:
        return self.models_config

    def is_model_downloaded(self, model_id: str) -> bool:
        if model_id not in self.models_config:
            return False
        local_path = self.models_config[model_id].get("local_path")
        if not local_path:
            return False
        # Simple check: if directory exists and is not empty
        return os.path.isdir(local_path) and len(os.listdir(local_path)) > 0

    def download_model(self, model_id: str, force: bool = False):
        if model_id not in self.models_config:
            raise ValueError(f"Model {model_id} not found in configuration.")

        model_info = self.models_config[model_id]
        repo_id = model_info["repo_id"]
        local_path = model_info["local_path"]

        logger.info(
            f"Starting download for {model_id} from {repo_id} to {local_path}..."
        )

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                force_download=force,
                # local_dir_use_symlinks=False # Often better for direct usage
            )
            logger.info(f"Successfully downloaded {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            raise e

    def add_model_config(
        self,
        model_id: str,
        repo_id: str,
        local_path: str,
        model_type: str = "llm",
        description: str = "",
    ):
        self.models_config[model_id] = {
            "repo_id": repo_id,
            "local_path": local_path,
            "type": model_type,
            "description": description,
        }
        self.save_config()

    def get_model_path(self, model_id: str) -> Optional[str]:
        if model_id in self.models_config:
            return self.models_config[model_id]["local_path"]
        return None
