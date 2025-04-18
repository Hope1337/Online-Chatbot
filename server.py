import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from flask import Flask, request, jsonify, Response, stream_with_context
import json
from queue import Queue

app = Flask(__name__)

# Cấu hình quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

# Load model và tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

def generate_stream(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Tạo queue để lưu trữ các token
    token_queue = Queue()
    
    # Custom streamer để gửi từng token
    class CustomStreamer(TextStreamer):
        def __init__(self, tokenizer, queue, **kwargs):
            super().__init__(tokenizer, **kwargs)
            self.queue = queue
        
        def on_finalized_text(self, text: str, stream_end: bool = False):
            # Gửi text không rỗng vào queue
            if text.strip():
                self.queue.put(text)
            if stream_end:
                self.queue.put("")  # Kết thúc stream

    streamer = CustomStreamer(tokenizer, token_queue, skip_prompt=True, skip_special_tokens=True)
    
    # Chạy model.generate trong một thread riêng để tránh block
    from threading import Thread
    
    def run_generate():
        try:
            model.generate(
                **inputs,
                max_length=32768,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                streamer=streamer,
            )
        except Exception as e:
            token_queue.put(f"Error: {str(e)}")
    
    # Bắt đầu thread để chạy model.generate
    generate_thread = Thread(target=run_generate)
    generate_thread.start()
    
    # Yield các token từ queue
    while True:
        token = token_queue.get()
        yield token
        if token == "":  # Kết thúc stream
            break
    
    generate_thread.join()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        
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
                for token in generate_stream(text):
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