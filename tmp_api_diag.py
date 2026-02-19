import json
import os
import sys

import replicate
import requests


def mask_prefix(value: str) -> str:
    if not value:
        return "<missing>"
    return f"{value[:8]}***"


openai_key = os.environ.get("OPENAI_API_KEY", "")
replicate_key = os.environ.get("REPLICATE_API_TOKEN", "")

print("python_executable=", sys.executable)
print("conda_env=", os.environ.get("CONDA_DEFAULT_ENV"))
print("openai_prefix=", mask_prefix(openai_key))
print("replicate_prefix=", mask_prefix(replicate_key))
print("openai_key_matches_getenv=", openai_key == os.getenv("OPENAI_API_KEY"))
print(
    "replicate_key_matches_getenv=",
    replicate_key == os.getenv("REPLICATE_API_TOKEN"),
)

endpoint = "https://api.openai.com/v1/chat/completions"
model = "gpt-5.2"
payload = {"model": model, "messages": [{"role": "user", "content": "ping"}]}
resp = requests.post(
    endpoint,
    headers={
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=60,
)
print("openai_endpoint=", endpoint)
print("openai_model=", model)
print("openai_status=", resp.status_code)
try:
    print("openai_raw_json=", json.dumps(resp.json(), ensure_ascii=False)[:4000])
except Exception:
    print("openai_raw_text=", (resp.text or "")[:4000])

try:
    output = replicate.run(
        "thomasmol/whisper-diarization:1495a9cddc83b2203b0d8d3516e38b80fd1572ebc4bc5700ac1da56a9b3ed886",
        input={
            "file": open("audio/test recording.wav", "rb"),
            "output": "json",
            "group_segments": True,
            "num_speakers": 2,
        },
    )
    print("replicate_status=", 200)
    print("replicate_response=", json.dumps(output, ensure_ascii=False)[:4000])
except Exception as exc:
    print("replicate_error=", str(exc)[:4000])
