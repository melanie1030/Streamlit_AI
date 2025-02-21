import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF
import pdfplumber  # 提取表格
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from openai import OpenAI  # 更新後的 OpenAI 調用方式

# 設定 Tesseract OCR 的路徑（Windows 用戶請修改路徑）
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# 設定 Poppler 的路徑（用戶自訂）
POPPLER_PATH = r"C:\\Program Files\\poppler-24.08.0\\Library\\bin"

# 將文本分段以避免 Token 超限
def split_text(text, max_tokens=8000):
    paragraphs = text.split("\n\n")
    current_chunk = ""
    chunks = []

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_tokens:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk)
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def save_text_as_txt(text, filename="ocr_result.txt"):
    """ 將字串存成 TXT 檔案並提供下載 """
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    with open(temp_path, "w", encoding="utf-8") as file:
        file.write(text)

    with open(temp_path, "rb") as file:
        st.download_button(
            label="📥 下載 OCR 解析結果",
            data=file,
            file_name=filename,
            mime="text/plain"
        )

def extract_text_with_ocr(pdf_path):
    """ 混合 PDF 文字提取與 OCR，並優化表格識別 """
    doc = fitz.open(pdf_path)
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            tables = page.extract_tables()

            if text:
                full_text += f"--- 第 {i+1} 頁 ---\n{text}\n\n"

            if tables:
                for table in tables:
                    formatted_table = "\n".join(
                        [" | ".join(str(cell) if cell is not None else "" for cell in row) for row in table]
                    )
                    full_text += f"--- 表格（第 {i+1} 頁）---\n{formatted_table}\n\n"

    for page in doc:
        if not page.get_text("text").strip():
            images = convert_from_path(
                pdf_path, first_page=page.number+1, last_page=page.number+1, poppler_path=POPPLER_PATH
            )
            for img in images:
                ocr_text = pytesseract.image_to_string(
                    img, lang="eng+chi_tra", config="--psm 6"
                )
                full_text += f"--- OCR 解析（第 {page.number+1} 頁）---\n{ocr_text}\n\n"

    return full_text

def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        all_docs = []
        combined_text = ""

        for uploaded_file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                extracted_text = extract_text_with_ocr(temp_file_path)
                os.remove(temp_file_path)

                combined_text += extracted_text + "\n\n"
                all_docs.append(extracted_text)
            except Exception as e:
                st.error(f"處理文件 {uploaded_file.name} 時出錯: {e}")
                continue

        st.session_state.vectors = combined_text
        save_text_as_txt(combined_text)

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

st.sidebar.header("Settings")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thinking_protocol" not in st.session_state:
    st.session_state.thinking_protocol = None

OPENAI_MODELS = [
    "gpt-4-turbo", 
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4o"
]

st.sidebar.subheader("選擇模型")
selected_model = st.sidebar.selectbox("請選擇模型", OPENAI_MODELS, index=0)

st.session_state.selected_model = selected_model

if st.sidebar.button("🗑️ 清除記憶"):
    st.session_state.memory = []
    st.session_state.messages = []
    st.session_state.thinking_protocol = None
    st.sidebar.success("記憶已清除！")

st.sidebar.subheader("🧠 記憶狀態")
if st.session_state.messages:
    memory_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    st.sidebar.text_area("當前記憶", value=memory_content, height=200)
else:
    st.sidebar.text_area("當前記憶", value="尚無訊息。", height=200)

st.sidebar.subheader("🧠 上傳 Thinking Protocol")
uploaded_thinking_protocol = st.sidebar.file_uploader(
    "選擇 thinking_protocol.md 檔案:", type=["md"], key="thinking_protocol_uploader"
)
if uploaded_thinking_protocol:
    try:
        thinking_protocol_content = uploaded_thinking_protocol.read().decode("utf-8")
        st.session_state.thinking_protocol = thinking_protocol_content
        st.session_state.messages.append({"role": "user", "content": f"📄 Thinking Protocol:\n\n{thinking_protocol_content}"})
        st.sidebar.success("Thinking Protocol 上傳成功！")
    except Exception as e:
        st.sidebar.error(f"讀取 Thinking Protocol 時出錯: {e}")

st.sidebar.write("🔑 **API Key Required**:\n- Get your OpenAI API key from [OpenAI API Key Page](https://platform.openai.com/)")

st.session_state.api_key = st.sidebar.text_input(
    "Enter your OpenAI API key:", type="password", value=st.session_state.api_key
)
if st.session_state.api_key:
    client = OpenAI(api_key=st.session_state.api_key)

    uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if st.sidebar.button("Process Documents"):
            with st.spinner("Processing documents... Please wait."):
                vector_embedding(uploaded_files)
                st.sidebar.write("✅ Documents processed successfully!")

st.title("📊 財報分析 AI（OCR & 表格提取）")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if human_input := st.chat_input("請輸入財報相關問題（例如：公司的流動比率是多少？）"):
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        text_chunks = split_text(st.session_state.vectors)
        assistant_response = ""

        for idx, chunk in enumerate(text_chunks):
            response = client.chat.completions.create(
                model=st.session_state.selected_model,
                messages=[
                    {"role": "system", "content": "你是一名財務分析師，負責解讀 PDF 財報數據。"},
                    {"role": "user", "content": f"**提供的財報內容（分段 {idx+1}/{len(text_chunks)}）**:\n{chunk}\n\n**使用者問題**:\n{human_input}"}
                ]
            )
            assistant_response += response.choices[0].message.content + "\n"
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

    else:
        st.error("請先上傳並處理財報 PDF！")
