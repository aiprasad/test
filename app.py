import torch
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

@app.post("/generate")
async def generate(data: dict):
    prompt = data["prompt"]
    max_length = data.get("max_length", 100)

    output = generate_text(prompt, max_length=max_length)
    generated_text = output[0]["generated_text"]

    return {"text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

