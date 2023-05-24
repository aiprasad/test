
import torch
from fastapi import FastAPI
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.bfloat16)

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data["prompt"]
    max_length = data.get("max_length", 100)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = generate_text.generate(input_ids, max_length=max_length)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"text": generated_text})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
