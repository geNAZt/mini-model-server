import threading
import queue
import json
import base64
import io
import numpy as np
import openvino as ov  # Wir importieren das Core-Modul
import openvino_genai as ov_genai
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()
MODEL_PATH = "/opt/ai/models/gemma/1"

print("⏳ Lade VLM-Modell auf GPU (UHD 730)...")
# Pipeline laden
pipe = ov_genai.VLMPipeline(MODEL_PATH, "GPU")
print("✅ Auge & Gehirn auf GPU bereit!")

def extract_image_and_text(messages):
    """Extrahiert Text und das letzte Bild aus der OpenAI-Struktur."""
    prompt = ""
    ov_image = None
    
    last_msg = messages[-1]
    content = last_msg.get("content")
    
    if isinstance(content, list):
        for item in content:
            if item["type"] == "text":
                prompt = item["text"]
            elif item["type"] == "image_url":
                # Base64 Bild extrahieren
                base_url = item["image_url"]["url"]
                base64_data = base_url.split(",")[1] if "," in base_url else base_url
                img_data = base64.b64decode(base64_data)
                pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # FIX: Wir nutzen einen OpenVINO Core Tensor statt ov_genai.Image
                image_array = np.array(pil_img)
                ov_image = ov.Tensor(image_array)
    else:
        prompt = content
        
    return prompt, ov_image

@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    prompt, image = extract_image_and_text(messages)

    def event_generator():
        q = queue.Queue()
        def streamer(subword):
            q.put(subword)
            return False 

        def run_generation():
            try:
                # Wir bauen einen sauberen Prompt ohne Altlasten
                # Gemma 3 braucht bei Bildern oft eine klare Struktur
                if image is not None:
                    # Wir senden nur den aktuellen Prompt mit dem Bild
                    # Das verhindert den History-Fehler
                    formatted_prompt = f"user\n<image>\n{prompt}\nassistant\n"
                    pipe.generate(formatted_prompt, image=image, max_new_tokens=1024, streamer=streamer)
                else:
                    pipe.generate(prompt, max_new_tokens=1024, streamer=streamer)
            except Exception as e:
                print(f"Fehler Details: {e}")
                q.put(f"\n[Ein Fehler ist aufgetreten: {str(e)} - Versuche die Seite neu zu laden oder den Chat zu leeren.]")
            q.put(None)

        threading.Thread(target=run_generation).start()
        while True:
            token = q.get()
            if token is None:
                yield "data: [DONE]\n\n"; break
            chunk = {"choices": [{"delta": {"content": token}, "index": 0}]}
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "gemma3", "object": "model"}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
