from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

load_dotenv()

repo_id = "OpenVINO/gemma-3-12b-it-int4-ov"
local_dir = "/opt/ai/models/gemma12b/1"

print(f"Starte Download von {repo_id}...")

os.makedirs(local_dir, exist_ok=True)

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        library_name="openvino",
        token=os.getenv('HF_TOKEN')
    )
    print("\n✅ Download erfolgreich abgeschlossen!")
except Exception as e:
    print(f"\n❌ Fehler beim Download: {e}")
