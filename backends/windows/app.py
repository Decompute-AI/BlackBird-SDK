from flask import Flask, request, jsonify
from unsloth import FastLanguageModel

# Initialize Flask app
app = Flask(__name__)

# Load Unsloth model & tokenizer once at startup
MODEL_ID = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=2048,
    dtype="float16",
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

@app.route("/generate", methods=["POST"])
def generate():
    """
    Expects JSON payload:
      {
        "prompt": "Your prompt text",
        "max_new_tokens": 64        # optional
      }
    Returns:
      {
        "generated_text": "..."
      }
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", 64)

    # Tokenize input and move tensors to model device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Run generation
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({
        "generated_text": text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
