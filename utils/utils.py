from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from queue import Queue
from .prompts import guide_prompt

def get_model(model_name = "Qwen/Qwen2.5-7B-Instruct", bnb_config = BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=6.0,)):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model

def generate_stream(prompt, tokenizer, model):
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
    
    # Yield các token từ queuejson={ "characteristic": None, "chat_history": chat_history, "prompt": prompt, "doc_file_name": "" },  xong tôi có code 
    while True:
        token = token_queue.get()
        yield token
        if token == "":  # Kết thúc stream
            break
    
    generate_thread.join()


def create_context_window(chat_history, tokenizer, max_context_length=20000):
    # Initialize variables
    current_tokens = 0
    context_window = []
    
    # Process chat history in reverse to prioritize recent messages
    for message in reversed(chat_history):
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            continue
            
        # Tokenize the message content
        tokens = tokenizer.encode(message["content"], add_special_tokens=False)
        token_count = len(tokens)
        
        # Check if adding this message would exceed max context length
        if current_tokens + token_count > max_context_length:
            break
            
        # Add message to context window (in original order)
        context_window.insert(0, message)
        current_tokens += token_count
    
    return context_window

def build_context_window(strings, tokenizer, max_length=5000):

    if not strings:
        return ""

    # Nếu reverse, đảo ngược danh sách để ưu tiên chuỗi mới nhất
    working_strings = strings

    selected_strings = []
    current_token_count = 0

    for s in working_strings:
        # Mã hóa chuỗi để đếm số token
        tokens = tokenizer.encode(s, add_special_tokens=False)
        token_count = len(tokens)

        # Kiểm tra nếu thêm chuỗi này có vượt quá max_length không
        if current_token_count + token_count <= max_length:
            selected_strings.append(s)
            current_token_count += token_count
        else:
            # Nếu vượt quá, thử cắt chuỗi hiện tại để vừa với max_length
            remaining_tokens = max_length - current_token_count
            if remaining_tokens > 0:
                # Giải mã lại một phần chuỗi để vừa với số token còn lại
                partial_tokens = tokens[:remaining_tokens]
                partial_string = tokenizer.decode(partial_tokens, skip_special_tokens=True)
                if partial_string.strip():  # Chỉ thêm nếu chuỗi không rỗng sau khi cắt
                    selected_strings.append(partial_string)
            break

    print("asdflkjasdflkjasd")
    for st in strings:
        print(st)
    # Nối các chuỗi đã chọn thành một context window
    return "\n".join(selected_strings)


def create_prompt(data, tokenizer, retriever):
    chat_history   = create_context_window(data.get("chat_history", "Assistant"), tokenizer)
    characteristic = data.get("characteristic", "You are my assistant")
    prompt         = data.get("prompt")
    related_doc    = data.get("doc_file_name")
    print(related_doc)

    chat_history   = "\n".join(f" {k} :{v}" for message in chat_history for k, v in message.items())

    if related_doc != "None":
        retriever.encode_document(related_doc)
        #print(prompt)
        doc_list = retriever.query_document(prompt, related_doc, top_k=10, similarity_threshold=0.001, max_length=512)
        #print(doc_list)
        #print(doc_list)
        related_doc = build_context_window(doc_list, tokenizer)

    prompt         = guide_prompt.format(characteristic, chat_history, related_doc, prompt)

    return prompt


