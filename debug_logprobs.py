import json
import requests
import math

# Configuration
BASE_URL = "https://curtate-unkeeled-sam.ngrok-free.dev"

payload = {
    "model": "qwen3-coder",
    "messages": [{"role": "user", "content": "Say 'Alice, Bob'."}],
    "max_tokens": 20,
    "temperature": 0.0,
    "logprobs": True,
    "top_logprobs": 1,
    "stream": True,
}

print(f"Connecting to {BASE_URL}...")
try:
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=30
    )
    response.raise_for_status()

    with open("raw_chunks.txt", "w", encoding="utf-8") as f:
        for i, line in enumerate(response.iter_lines()):
            if not line: continue
            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "): continue
            if "[DONE]" in line_str: 
                f.write(f"\n--- [DONE] ---\n")
                break
            
            data_str = line_str[6:]
            try:
                data = json.loads(data_str)
                f.write(f"\n--- Chunk {i} ---\n")
                f.write(json.dumps(data, indent=2))
                f.write("\n")
            except:
                f.write(f"\n--- Chunk {i} (Error decoding) ---\n")
                f.write(data_str + "\n")
            
            if i > 50: break

    print("Successfully wrote chunks to raw_chunks.txt")
except Exception as e:
    print(f"Error: {e}")
