import streamlit as st
import requests

# Cấu hình API endpoint
API_URL = "http://localhost:5000/predict"

# Khởi tạo session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {"Session 1": []}
if "current_session" not in st.session_state:
    st.session_state.current_session = "Session 1"
if "documents" not in st.session_state:
    st.session_state.documents = ["Document 1", "Document 2", "Document 3"]
if "selected_document" not in st.session_state:
    st.session_state.selected_document = None
if "current_response" not in st.session_state:
    st.session_state.current_response = ""
if "edit_session_name" not in st.session_state:
    st.session_state.edit_session_name = False  # Trạng thái hiển thị ô chỉnh sửa tên

# Giao diện
st.title("Chatbot NLP")

# Sidebar: Quản lý session và tài liệu
with st.sidebar:
    st.header("Chat Sessions")
    
    # Nút tạo session mới
    if st.button("Create New Chat Session"):
        # Tạo tên mặc định
        session_count = len(st.session_state.chat_history) + 1
        new_session_name = f"Session {session_count}"
        st.session_state.chat_history[new_session_name] = []
        st.session_state.current_session = new_session_name  # Tự động chuyển sang session mới

    # Dropdown chọn session
    session_choice = st.selectbox(
        "Select Session",
        list(st.session_state.chat_history.keys()),
        index=list(st.session_state.chat_history.keys()).index(st.session_state.current_session)
    )
    if session_choice:
        st.session_state.current_session = session_choice

    st.header("Document Selection")
    st.session_state.selected_document = st.selectbox("Select Document for RAG", ["None"] + st.session_state.documents)

# Khu vực tiêu đề và tên session
col1, col2 = st.columns([3, 1])
with col1:
    if st.session_state.edit_session_name:
        # Hiển thị ô nhập liệu để chỉnh sửa tên
        new_name = st.text_input(
            "Edit session name:",
            value=st.session_state.current_session,
            key="edit_session_name_input"
        )
    else:
        # Hiển thị tên session dưới dạng văn bản
        st.markdown(f"**Chat - {st.session_state.current_session}**")
with col2:
    if st.session_state.edit_session_name:
        if st.button("Save"):
            if new_name and new_name != st.session_state.current_session:
                # Cập nhật tên session trong chat_history
                st.session_state.chat_history[new_name] = st.session_state.chat_history.pop(st.session_state.current_session)
                st.session_state.current_session = new_name
            st.session_state.edit_session_name = False  # Ẩn ô chỉnh sửa
    else:
        if st.button("Edit"):
            st.session_state.edit_session_name = True  # Hiển thị ô chỉnh sửa

# Khu vực chat
chat_container = st.container()

# Hiển thị lịch sử chat
with chat_container:
    for message in st.session_state.chat_history.get(st.session_state.current_session, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Xử lý form và gửi prompt
with st.form(key="chat_form", clear_on_submit=True):
    prompt = st.text_input("Enter your question:", key="prompt_input")
    submit_button = st.form_submit_button("Send")

if submit_button and prompt:
    # Thêm câu hỏi vào lịch sử và hiển thị ngay
    st.session_state.chat_history[st.session_state.current_session].append({"role": "user", "content": prompt})
    print(st.session_state.chat_history[st.session_state.current_session])
    
    # Hiển thị tin nhắn người dùng ngay lập tức
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Tạo message container cho response của chatbot
    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Gửi request tới API và xử lý streaming
            try:
                headers = {
                    'Accept': 'text/plain',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                }
                response = requests.post(API_URL, json={"prompt": prompt}, stream=True, headers=headers, timeout=60)
                
                if response.status_code == 200:
                    # Đọc từng chunk từ stream
                    st.session_state.current_response = ""
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            st.session_state.current_response += chunk
                            message_placeholder.markdown(st.session_state.current_response + "▌")
                    
                    # Lưu response hoàn chỉnh vào lịch sử
                    if st.session_state.current_response:
                        st.session_state.chat_history[st.session_state.current_session].append(
                            {"role": "assistant", "content": st.session_state.current_response}
                        )
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
