import streamlit as st
import requests
import json

with open("data/api_aut.json", "r") as f:
    data_key = json.load(f)

# Cấu hình API endpoints
API_URL        = data_key[0]['GENERATE']
DOC_LIST_URL   = data_key[0]['UPDATE_DOCS_LIST']
API_KEY        = data_key[0]['API_KEY']

# Khởi tạo session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {"Session 1": []}
if "session_prompts" not in st.session_state:
    st.session_state.session_prompts = {"Session 1": ""}  # Lưu prompting cho mỗi session
if "current_session" not in st.session_state:
    st.session_state.current_session = "Session 1"
if "documents" not in st.session_state:
    st.session_state.documents = []
if "selected_document" not in st.session_state:
    st.session_state.selected_document = None
if "current_response" not in st.session_state:
    st.session_state.current_response = ""
if "edit_session_name" not in st.session_state:
    st.session_state.edit_session_name = False

# Giao diện
st.title("Chatbot NLP")

# Sidebar: Quản lý session, tài liệu và prompting
with st.sidebar:
    st.header("Chat Sessions")
    
    # Nút tạo session mới
    if st.button("Create New Chat Session"):
        session_count = len(st.session_state.chat_history) + 1
        new_session_name = f"Session {session_count}"
        # Copy lịch sử chat và prompting từ session hiện tại
        st.session_state.chat_history[new_session_name] = []
        st.session_state.session_prompts[new_session_name] = st.session_state.session_prompts.get(
            st.session_state.current_session, ""
        )
        st.session_state.current_session = new_session_name

    # Dropdown chọn session
    session_choice = st.selectbox(
        "Select Session",
        list(st.session_state.chat_history.keys()),
        index=list(st.session_state.chat_history.keys()).index(st.session_state.current_session)
    )
    if session_choice:
        st.session_state.current_session = session_choice

    st.header("Document Selection")
    # Sử dụng columns để đặt dropdown và nút reset cạnh nhau
    col1, col2 = st.columns([3, 1])
    with col1:
        options = ["None"] + st.session_state.documents
        if st.session_state.selected_document is None or st.session_state.selected_document not in st.session_state.documents:
            dropdown_index = 0
        else:
            dropdown_index = st.session_state.documents.index(st.session_state.selected_document) + 1
        
        st.session_state.selected_document = st.selectbox(
            "Select Document for RAG",
            options,
            index=dropdown_index,
            key="doc_select"
        )
    with col2:
        # Nút reset danh sách tài liệu
        if st.button("Reset", key="reset_doc_button"):
            try:
                response = requests.get(DOC_LIST_URL, headers={"X-API-Key": API_KEY}, timeout=10)
                if response.status_code == 200:
                    st.session_state.documents = response.json().get("files", [])
                    if st.session_state.selected_document not in st.session_state.documents:
                        st.session_state.selected_document = None
                else:
                    st.error(f"Error fetching document list: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")

    # Prompting box cho session hiện tại
    st.header("Session Prompt")
    col_prompt, col_button = st.columns([4, 1])  # Điều chỉnh tỷ lệ để cân đối hơn
    with col_prompt:
        current_prompt = st.text_area(
            "Enter session prompt:",
            value=st.session_state.session_prompts.get(st.session_state.current_session, ""),
            key=f"prompt_{st.session_state.current_session}",
            height=150
        )
    with col_button:
        # Đặt nút OK với nhãn ngắn gọn và căn chỉnh
        st.markdown(
            """
            <style>
            div[data-testid="column"]:nth-child(2) .stButton > button {
                margin-top: 30px;  /* Căn chỉnh nút theo chiều dọc */
                width: 100%;       /* Đảm bảo nút chiếm toàn bộ chiều rộng cột */
                padding: 8px;      /* Giảm padding để nút gọn hơn */
                font-size: 14px;   /* Giảm kích thước chữ */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        if st.button("OK", key=f"apply_prompt_{st.session_state.current_session}"):
            # Cập nhật prompting cho session hiện tại khi nhấn OK
            st.session_state.session_prompts[st.session_state.current_session] = current_prompt

# Khu vực tiêu đề và tên session
col1, col2 = st.columns([3, 1])
with col1:
    if st.session_state.edit_session_name:
        new_name = st.text_input(
            "Edit session name:",
            value=st.session_state.current_session,
            key="edit_session_name_input"
        )
    else:
        st.markdown(f"**Chat - {st.session_state.current_session}**")
with col2:
    if st.session_state.edit_session_name:
        if st.button("Save"):
            if new_name and new_name != st.session_state.current_session:
                st.session_state.chat_history[new_name] = st.session_state.chat_history.pop(st.session_state.current_session)
                st.session_state.session_prompts[new_name] = st.session_state.session_prompts.pop(st.session_state.current_session)
                st.session_state.current_session = new_name
            st.session_state.edit_session_name = False
    else:
        if st.button("Edit"):
            st.session_state.edit_session_name = True

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
    chat_history = st.session_state.chat_history[st.session_state.current_session].copy()

    # Thêm câu hỏi vào lịch sử và hiển thị ngay
    st.session_state.chat_history[st.session_state.current_session].append({"role": "user", "content": prompt})
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                headers = {
                    'Accept': 'text/plain',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-API-Key': API_KEY,
                }
                response = requests.post(
                    API_URL, 
                    json={
                        "characteristic": st.session_state.session_prompts.get(st.session_state.current_session, ""),
                        "chat_history": chat_history, 
                        "prompt": prompt,
                        "doc_file_name": st.session_state.selected_document,
                    }, 
                    stream=True, 
                    headers=headers, 
                    timeout=60
                )
                
                if response.status_code == 200:
                    st.session_state.current_response = ""
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            st.session_state.current_response += chunk
                            message_placeholder.markdown(st.session_state.current_response + "▌")
                    
                    if st.session_state.current_response:
                        st.session_state.chat_history[st.session_state.current_session].append(
                            {"role": "assistant", "content": st.session_state.current_response}
                        )
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")