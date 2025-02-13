import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF
import pdfplumber  # 新增庫來提取表格
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 設定 Tesseract OCR 的路徑（Windows 用戶請修改路徑）
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 設定 Poppler 的路徑（用戶自訂）
POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"

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

    # 使用 pdfplumber 提取表格 & 文字
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            tables = page.extract_tables()

            # 如果有直接可用的文字，則使用 pdfplumber
            if text:
                full_text += f"--- 第 {i+1} 頁 ---\n{text}\n\n"

            # 如果有表格，格式化並加入文本
            if tables:
                for table in tables:
                    formatted_table = "\n".join(
                        [" | ".join(str(cell) if cell is not None else "" for cell in row) for row in table]
                    )
                    full_text += f"--- 表格（第 {i+1} 頁）---\n{formatted_table}\n\n"

    # OCR 提取圖片中的文本（針對掃描版 PDF）
    for page in doc:
        if not page.get_text("text").strip():  # 如果 PDF 內文字無法提取，則使用 OCR
            images = convert_from_path(
                pdf_path, first_page=page.number+1, last_page=page.number+1, poppler_path=POPPLER_PATH
            )
            for img in images:
                ocr_text = pytesseract.image_to_string(
                    img, lang="eng+chi_tra", config="--psm 6"
                )
                full_text += f"--- OCR 解析（第 {page.number+1} 頁）---\n{ocr_text}\n\n"

    return full_text


# 初始化 API Key
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

st.sidebar.header("Settings")

# 初始化 session state 變數
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thinking_protocol" not in st.session_state:
    st.session_state.thinking_protocol = None

# --- 模型選擇 ---
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

# --- 清除記憶功能 ---
if st.sidebar.button("🗑️ 清除記憶"):
    # 清除所有記憶變數
    st.session_state.memory = []
    st.session_state.messages = []
    st.session_state.thinking_protocol = None  # 清除 Thinking Protocol

    # 顯示成功提示
    st.sidebar.success("記憶已清除！")

# --- 記憶狀態顯示 ---
st.sidebar.subheader("🧠 記憶狀態")
if st.session_state.messages:
    memory_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    st.sidebar.text_area("當前記憶", value=memory_content, height=200)
else:
    st.sidebar.text_area("當前記憶", value="尚無訊息。", height=200)

# --- Thinking Protocol 上傳 ---
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
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key

    # 初始化 ChatOpenAI
    llm = ChatOpenAI(api_key=st.session_state.api_key, model_name=st.session_state.selected_model)

    # 財報分析 Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        你是一名財務分析師，負責解讀 PDF 財報數據。

        **Step 1: 檢索關鍵數據**
        - 你要根據以下「提供的財報內容」來回答問題。
        - 優先查找「現金流量表」、「資產負債表」、「損益表」中的數據。
        - 若問題涉及財務比率，則需要找到對應的數字進行計算。

        **Step 2: 嘗試計算財務比率（如果適用）**
        - **流動比率 = 流動資產 ÷ 流動負債**
        - **負債比率 = 總負債 ÷ 總資產**
        - **自由現金流 = 營業現金流 - 資本支出**

        **Step 3: 給出回答**
        - 若有數據，則提供計算結果
        - 若數據不足，則建議使用者查看特定財報部分

        **提供的財報內容**：
        {context}

        **使用者問題**：
        {input}
        """
    )

    # 上傳 PDF 文件
    uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    
    
    if uploaded_files:
        if st.sidebar.button("Process Documents"):
            with st.spinner("Processing documents... Please wait."):
                
                def vector_embedding(uploaded_files):
                    if "vectors" not in st.session_state:
                        st.session_state.embeddings = OpenAIEmbeddings()
                        all_docs = []
                        combined_text = ""

                        for uploaded_file in uploaded_files:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                                temp_file.write(uploaded_file.read())
                                temp_file_path = temp_file.name

                            extracted_text = extract_text_with_ocr(temp_file_path)
                            combined_text += extracted_text + "\n\n"

                            os.remove(temp_file_path)
                            all_docs.append(extracted_text)

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        final_documents = text_splitter.create_documents(all_docs)

                        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

                        save_text_as_txt(combined_text)

                vector_embedding(uploaded_files)
                st.sidebar.write("✅ Documents processed successfully!")

# 主要聊天介面
st.title("📊 財報分析 AI（OCR & 表格提取）")

# 聊天記錄
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示聊天歷史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 使用者輸入問題
if human_input := st.chat_input("請輸入財報相關問題（例如：公司的流動比率是多少？）"):
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({"input": human_input, "context": ""})
        assistant_response = response["answer"]

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

    else:
        st.error("請先上傳並處理財報 PDF！")
