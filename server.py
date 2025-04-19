import torch
import json
import os
from flask import Flask, request, jsonify, Response, stream_with_context
from functools import wraps
from utils.utils import get_model, generate_stream, create_prompt
from utils.retriever import Retrieval

with open("data/api_aut.json", "r") as f:
    data_key = json.load(f)

VALID_API_KEYS = data_key[0]['API_KEY']

raw_doc_folder     = "data/pdf"
encoded_doc_folder = "data/embedded"

# Load model và tokenizer
tokenizer, model = get_model()
retriever = Retrieval(
    raw_doc_folder=raw_doc_folder,
    encoded_doc_folder=encoded_doc_folder
)

app = Flask(__name__)

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in VALID_API_KEYS:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/update_doc_list", methods=["GET"])
@require_api_key
def update_doc_list():
    folder_path = raw_doc_folder
    
    if not os.path.isdir(folder_path):
        return jsonify({"error": "Invalid folder path"}), 400

    bin_files = [os.path.splitext(f)[0] for f in os.listdir(folder_path)
                 if f.endswith('.pdf') and os.path.isfile(os.path.join(folder_path, f))]

    return jsonify({"files": bin_files})


@app.route("/generate", methods=["POST"])
@require_api_key
def generate():
    try:
        data = request.get_json()
        prompt = create_prompt(data, tokenizer, retriever)
        print(prompt)
        print("#########################################")
        print()
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        messages = [
            {"role": "system", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        def generate():
            try:
                for token in generate_stream(text, tokenizer, model):
                    yield token
            except Exception as e:
                yield f"Error: {str(e)}"

        return Response(
            stream_with_context(generate()),
            mimetype="text/plain",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)