import streamlit as st
import pandas as pd
import os
import io
import time
import dotenv
from PIL import Image
import numpy as np

# --- Plotly 和 Gemini/Langchain/OpenAI 等核心套件 ---
import plotly.express as px
import google.generativeai as genai
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 初始化與常數定義 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)

# --- 基礎輔助函數 ---
def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    return file_path

def add_user_image_to_main_chat(uploaded_file):
    try:
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.pending_image_for_main_gemini = Image.open(file_path)
        st.image(st.session_state.pending_image_for_main_gemini, caption="圖片已上傳，將隨下一條文字訊息發送。", use_container_width=True)
    except Exception as e: st.error(f"處理上傳圖片時出錯: {e}")

# --- RAG 核心函式 ---
@st.cache_resource
def create_lc_retriever(file_path: str, openai_api_key: str):
    with st.status("正在建立 RAG 知識庫...", expanded=True) as status:
        try:
            status.update(label="步驟 1/3：載入與切割文件...")
            loader = CSVLoader(file_path=file_path, encoding='utf-8')
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            status.update(label=f"步驟 1/3 完成！已切割成 {len(docs)} 個區塊。")
            status.update(label="步驟 2/3：呼叫 OpenAI API 生成向量嵌入...")
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vector_store = LangChainFAISS.from_documents(docs, embeddings)
            status.update(label="步驟 2/3 完成！向量嵌入已生成。")
            status.update(label="步驟 3/3：檢索器準備完成！", state="complete", expanded=False)
            return vector_store.as_retriever(search_kwargs={'k': 5})
        except Exception as e:
            st.error(f"建立知識庫過程中發生嚴重錯誤: {e}")
            status.update(label="建立失敗", state="error")
            return None

# --- Gemini API 相關函式 ---
def get_gemini_client(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash-latest")

def get_gemini_response_with_history(client, history, user_prompt):
    gemini_history = []
    if not isinstance(history, list):
        history = []
    for msg in history:
        role = "user" if msg["role"] == "human" else "model"
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        gemini_history.append({"role": role, "parts": [content]})

    chat = client.start_chat(history=gemini_history)
    response = chat.send_message(user_prompt)
    return response.text
    
def get_gemini_response_for_image(api_key, user_prompt, image_pil):
    if not api_key: return "錯誤：未設定 Gemini API Key。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content([user_prompt, image_pil])
        st.session_state.pending_image_for_main_gemini = None
        return response.text
    except Exception as e: return f"錯誤: {e}"

def get_gemini_executive_analysis(api_key, executive_role_name, full_prompt):
    if not api_key: return f"錯誤：專業經理人 ({executive_role_name}) 未能獲取 Gemini API Key。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e: return f"錯誤: {e}"
    
def generate_data_profile(df, is_simple=False):
    if df is None or df.empty: return "沒有資料可供分析。"
    if is_simple:
        preview_rows = min(5, df.shape[0])
        return f"資料共有 {df.shape[0]} 行, {df.shape[1]} 個欄位。\n前 {preview_rows} 筆資料預覽:\n{df.head(preview_rows).to_string()}"
    buffer = io.StringIO()
    df.info(buf=buffer)
    profile_parts = [f"資料形狀: {df.shape}", f"欄位資訊:\n{buffer.getvalue()}"]
    try: profile_parts.append(f"\n數值欄位統計:\n{df.describe(include='number').to_string()}")
    except: pass
    try: profile_parts.append(f"\n類別欄位統計:\n{df.describe(include=['object', 'category']).to_string()}")
    except: pass
    profile_parts.append(f"\n前 5 筆資料:\n{df.head().to_string()}")
    return "\n".join(profile_parts)

# --- 資料探索器核心函數 ---
@st.cache_data
def get_overview_metrics(df):
    if df is None or df.empty: return 0, 0, 0, 0, 0
    num_rows, num_cols = df.shape
    missing_percentage = (df.isnull().sum().sum() / (num_rows * num_cols)) * 100 if (num_rows * num_cols) > 0 else 0
    numeric_cols_count = len(df.select_dtypes(include=np.number).columns)
    duplicate_rows = df.duplicated().sum()
    return num_rows, num_cols, missing_percentage, numeric_cols_count, duplicate_rows

@st.cache_data
def get_column_quality_assessment(df):
    if df is None or df.empty: return pd.DataFrame()
    quality_data = [{"欄位": col, "資料類型": str(df[col].dtype), "缺失值比例 (%)": (df[col].isnull().sum() / len(df)) * 100 if len(df) > 0 else 0, "唯一值數量": df[col].nunique()} for col in df.columns]
    return pd.DataFrame(quality_data)

def display_simple_data_explorer(df):
    st.subheader("互動式資料探索")
    st.markdown("---")
    st.markdown("##### 關鍵指標")
    num_rows, num_cols, missing_percentage, numeric_cols_count, duplicate_rows = get_overview_metrics(df)
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("總行數", f"{num_rows:,}")
    kpi_cols[1].metric("總列數", f"{num_cols:,}")
    kpi_cols[2].metric("缺失值比例", f"{missing_percentage:.2f}%")
    kpi_cols[3].metric("數值型欄位", f"{numeric_cols_count}")
    kpi_cols[4].metric("重複行數", f"{duplicate_rows:,}")
    st.markdown("##### 欄位品質評估")
    st.dataframe(get_column_quality_assessment(df), use_container_width=True)
    st.markdown("---")
    st.markdown("##### 欄位資料分佈")
    plot_col1, plot_col2 = st.columns(2)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    with plot_col1:
        if numeric_cols:
            selected_numeric = st.selectbox("選擇一個數值型欄位查看分佈:", numeric_cols, key="explorer_numeric")
            if selected_numeric:
                st.plotly_chart(px.histogram(df, x=selected_numeric, title=f"'{selected_numeric}' 的分佈", marginal="box"), use_container_width=True)
        else: st.info("無數值型欄位可供分析。")
    with plot_col2:
        if categorical_cols:
            selected_categorical = st.selectbox("選擇一個類別型欄位查看分佈:", categorical_cols, key="explorer_categorical")
            if selected_categorical:
                top_n = st.slider("顯示前 N 個類別", 5, 20, 10, key="explorer_top_n")
                counts = df[selected_categorical].value_counts().nlargest(top_n)
                st.plotly_chart(px.bar(counts, x=counts.index, y=counts.values, title=f"'{selected_categorical}' 的前 {top_n} 個類別分佈", labels={'index':selected_categorical, 'y':'數量'}), use_container_width=True)
        else: st.info("無類別型欄位可供分析。")
    st.markdown("##### 數值欄位相關性熱力圖")
    if len(numeric_cols) > 1:
        st.plotly_chart(px.imshow(df[numeric_cols].corr(numeric_only=True), text_auto=True, aspect="auto", title="數值欄位相關性熱力圖", color_continuous_scale='RdBu_r'), use_container_width=True)
    else: st.info("需要至少兩個數值型欄位才能計算相關性。")
    
# --- 主應用入口 ---
def main():
    st.set_page_config(page_title="Gemini Multi-Function Bot", page_icon="✨", layout="wide")
    st.title("✨ Gemini 多功能 AI 助理 ")

    executive_session_id = "executive_chat"
    keys_to_init = {
        "use_rag": False, "use_multi_stage_workflow": False, "use_simple_explorer": False,
        "retriever_chain": None, "uploaded_file_path": None, "last_uploaded_filename": None,
        "pending_image_for_main_gemini": None, "chat_histories": {},
        "executive_workflow_stage": "idle", "executive_user_query": "",
        "executive_data_profile_str": "", "executive_rag_context": "", "cfo_analysis_text": "",
        "coo_analysis_text": "", "ceo_summary_text": "",
        "sp_workflow_stage": "idle", "sp_user_query": "", "sp_final_report": "",
        "follow_up_query": "", "follow_up_stage": "idle", 
        "follow_up_cfo_analysis": "", "follow_up_coo_analysis": "", "follow_up_ceo_analysis": "",
    }
    for key, default_value in keys_to_init.items():
        if key not in st.session_state: st.session_state[key] = default_value
    if executive_session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[executive_session_id] = []

    with st.sidebar:
        st.header("⚙️ 功能與模式設定")
        st.session_state.use_rag = st.checkbox("啟用 RAG 知識庫 (需 OpenAI Key)", value=st.session_state.use_rag)
        st.session_state.use_multi_stage_workflow = st.checkbox("啟用階段式工作流 (多重記憶)", value=st.session_state.use_multi_stage_workflow, help="預設(不勾選): AI 一次完成所有角色分析 (單一記憶)。勾選: AI 依序完成各角色分析 (多重記憶)，開銷較大。")
        st.session_state.use_simple_explorer = st.checkbox("啟用簡易資料探索器", value=st.session_state.use_simple_explorer, help="勾選後，將在工作流的統計摘要區塊顯示互動式圖表。")
        st.divider()
        st.header("🔑 API 金鑰")
        st.text_input("請輸入您的 Google Gemini API Key", type="password", key="gemini_api_key_input")
        if st.session_state.use_rag:
            st.text_input("請輸入您的 OpenAI API Key (RAG 功能需要)", type="password", key="openai_api_key_input")
        st.divider()
        st.header("📁 資料上傳")
        uploaded_file = st.file_uploader("上傳 CSV 檔案", type=["csv"])
        if uploaded_file:
            if uploaded_file.name != st.session_state.get("last_uploaded_filename"):
                st.session_state.last_uploaded_filename = uploaded_file.name
                file_path = save_uploaded_file(uploaded_file)
                st.session_state.uploaded_file_path = file_path
                st.success(f"檔案 '{uploaded_file.name}' 上傳成功！")
                st.session_state.retriever_chain = None 
                if st.session_state.use_rag:
                    openai_api_key = st.session_state.get("openai_api_key_input") or os.environ.get("OPENAI_API_KEY")
                    if not openai_api_key: st.error("RAG 功能已啟用，請在上方輸入您的 OpenAI API Key！")
                    else: st.session_state.retriever_chain = create_lc_retriever(file_path, openai_api_key)
        if st.session_state.retriever_chain: st.success("✅ RAG 知識庫已啟用！")
        st.header("🖼️ 圖片分析")
        uploaded_image = st.file_uploader("上傳圖片", type=["png", "jpg", "jpeg"])
        if uploaded_image: add_user_image_to_main_chat(uploaded_image)
        st.divider()
        if st.button("🗑️ 清除所有對話與資料"):
            settings = {k: st.session_state.get(k) for k in ['gemini_api_key_input', 'openai_api_key_input', 'use_rag', 'use_multi_stage_workflow', 'use_simple_explorer']}
            st.session_state.clear()
            for key, value in settings.items(): st.session_state[key] = value
            st.cache_resource.clear()
            st.success("所有對話、Session 記憶和快取已清除！")
            st.rerun()

    tab_titles = ["💬 主要聊天室", "💼 專業經理人"]
    tabs = st.tabs(tab_titles)
    gemini_api_key = st.session_state.get("gemini_api_key_input") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        st.warning("請在側邊欄輸入您的 Google Gemini API Key 以啟動聊天功能。")
        st.stop()
    gemini_client = get_gemini_client(gemini_api_key)

    with tabs[0]:
        st.header("💬 主要聊天室")
        st.caption("可進行一般對話、圖片分析。RAG 問答功能可由側邊欄開關啟用。")
        session_id = "main_chat"
        if session_id not in st.session_state.chat_histories: st.session_state.chat_histories[session_id] = []
        for msg in st.session_state.chat_histories[session_id]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if user_input := st.chat_input("請對數據、圖片提問或開始對話..."):
            st.session_state.chat_histories[session_id].append({"role": "human", "content": user_input})
            with st.chat_message("human"): st.markdown(user_input)
            with st.chat_message("ai"):
                with st.spinner("正在思考中..."):
                    prompt_context = ""
                    if st.session_state.use_rag and st.session_state.retriever_chain:
                        context = "\n---\n".join([doc.page_content for doc in st.session_state.retriever_chain.invoke(user_input)])
                        prompt_context = f"請根據以下上下文回答問題。\n\n[上下文]:\n{context}\n\n"
                    elif not st.session_state.use_rag and st.session_state.get("uploaded_file_path"):
                        try:
                            df = pd.read_csv(st.session_state.uploaded_file_path)
                            prompt_context = f"請參考以下資料摘要來回答問題。\n\n[資料摘要]:\n{generate_data_profile(df.head(), is_simple=True)}\n\n"
                        except Exception as e: st.warning(f"讀取 CSV 檔案時發生錯誤: {e}")
                    if st.session_state.pending_image_for_main_gemini:
                        response = get_gemini_response_for_image(gemini_api_key, f"{prompt_context} [問題]:\n{user_input}", st.session_state.pending_image_for_main_gemini)
                    else:
                        response = get_gemini_response_with_history(gemini_client, st.session_state.chat_histories[session_id][:-1], f"{prompt_context}[問題]:\n{user_input}")
                    st.markdown(response)
                    st.session_state.chat_histories[session_id].append({"role": "ai", "content": response})

    with tabs[1]:
        st.header("💼 專業經理人")
        st.caption(f"目前模式：{'階段式 (多重記憶)' if st.session_state.use_multi_stage_workflow else '整合式 (單一記憶)'} | RAG：{'啟用' if st.session_state.use_rag else '停用'} | 簡易探索器：{'啟用' if st.session_state.use_simple_explorer else '停用'}")

        # --- ▼▼▼ 全新、修正後的控制流程 ▼▼▼ ---
        # 模式一：階段式工作流
        if st.session_state.use_multi_stage_workflow:
            # Step 1: 處理用戶輸入與啟動
            st.info("**方法說明**：此流程將依序（CFO->COO->CEO）進行分析，每一步完成後會立刻顯示結果，並自動觸發下一步。")
            st.session_state.executive_user_query = st.text_area("請輸入商業問題以啟動分析:", value=st.session_state.get("executive_user_query", ""), height=100, key="original_workflow_query")
            can_start = bool(st.session_state.get("uploaded_file_path") and st.session_state.get("executive_user_query"))
            if st.button("🚀 啟動階段式分析", disabled=not can_start or st.session_state.executive_workflow_stage != "idle", key="exec_flow_button"):
                st.session_state.chat_histories[executive_session_id] = []
                st.session_state.executive_workflow_stage = "cfo_analysis_pending"
                st.session_state.cfo_analysis_text, st.session_state.coo_analysis_text, st.session_state.ceo_summary_text, st.session_state.executive_rag_context = "", "", "", ""
                st.rerun()

            # Step 2: 根據當前狀態，執行對應的分析 (核心修正)
            # 這個區塊的程式碼只做計算和狀態轉換，不做顯示
            stage = st.session_state.executive_workflow_stage
            if stage == "cfo_analysis_pending":
                with st.spinner("CFO 正在分析中..."):
                    df = pd.read_csv(st.session_state.uploaded_file_path)
                    st.session_state.executive_data_profile_str = generate_data_profile(df)
                    query = st.session_state.executive_user_query
                    rag_context_for_prompt = ""
                    if st.session_state.use_rag and st.session_state.retriever_chain:
                        rag_context = "\n---\n".join([doc.page_content for doc in st.session_state.retriever_chain.invoke(query)])
                        st.session_state.executive_rag_context = rag_context
                        rag_context_for_prompt = f"\n\n[RAG 檢索出的相關數據]:\n{rag_context}"
                    cfo_prompt = f"作為財務長(CFO)，請基於你的專業知識，並嚴格參考以下提供的資料，為商業問題提供財務角度的簡潔分析。\n\n[商業問題]:\n{query}\n\n[統計摘要]:\n{st.session_state.executive_data_profile_str}{rag_context_for_prompt}"
                    st.session_state.cfo_analysis_text = get_gemini_executive_analysis(gemini_api_key, "CFO", cfo_prompt)
                    st.session_state.executive_workflow_stage = "coo_analysis_pending" # 設定下一階段
            
            elif stage == "coo_analysis_pending":
                with st.spinner("COO 正在分析中..."):
                    rag_context_for_prompt = f"\n\n[RAG 檢索出的相關數據]:\n{st.session_state.executive_rag_context}" if st.session_state.executive_rag_context else ""
                    coo_prompt = f"作為營運長(COO)，請基於商業問題、統計摘要、CFO 的財務分析以及相關數據，提供營運層面的策略與潛在風險。\n\n[商業問題]:\n{st.session_state.executive_user_query}\n\n[CFO 的財務分析]:\n{st.session_state.cfo_analysis_text}\n\n[統計摘要]:\n{st.session_state.executive_data_profile_str}{rag_context_for_prompt}"
                    st.session_state.coo_analysis_text = get_gemini_executive_analysis(gemini_api_key, "COO", coo_prompt)
                    st.session_state.executive_workflow_stage = "ceo_summary_pending" # 設定下一階段
            
            elif stage == "ceo_summary_pending":
                with st.spinner("CEO 正在總結中..."):
                    rag_context_for_prompt = f"\n\n[RAG 檢索出的相關數據]:\n{st.session_state.executive_rag_context}" if st.session_state.executive_rag_context else ""
                    ceo_prompt = f"作為執行長(CEO)，請整合所有資訊，提供一個高層次的決策總結。\n\n[商業問題]:\n{st.session_state.executive_user_query}\n\n[CFO 的財務分析]:\n{st.session_state.cfo_analysis_text}\n\n[COO 的營運分析]:\n{st.session_state.coo_analysis_text}{rag_context_for_prompt}"
                    st.session_state.ceo_summary_text = get_gemini_executive_analysis(gemini_api_key, "CEO", ceo_prompt)
                    st.session_state.executive_workflow_stage = "completed" # 標記完成
                    full_report = f"### 📊 財務長 (CFO) 分析\n{st.session_state.cfo_analysis_text}\n\n### 🏭 營運長 (COO) 分析\n{st.session_state.coo_analysis_text}\n\n### 👑 執行長 (CEO) 最終決策\n{st.session_state.ceo_summary_text}"
                    st.session_state.chat_histories[executive_session_id].append({"role": "ai", "content": full_report})

        # 模式二：整合式工作流
        else:
            st.info("**方法說明**：此為預設流程。模擬一個全能的 AI 專業經理人團隊，只發送**一次**請求，AI 在一次生成中完成所有角色思考。")
            st.session_state.sp_user_query = st.text_area("請輸入商業問題以啟動分析:", value=st.session_state.get("sp_user_query", ""), height=100, key="sp_workflow_query")
            can_start_sp = bool(st.session_state.get("uploaded_file_path") and st.session_state.get("sp_user_query"))
            if st.button("🚀 啟動整合分析", disabled=not can_start_sp, key="sp_flow_button"):
                st.session_state.chat_histories[executive_session_id] = []
                st.session_state.sp_workflow_stage = "running"
                with st.spinner("AI 專業經理人團隊正在進行全面分析..."):
                    df = pd.read_csv(st.session_state.uploaded_file_path)
                    data_profile = generate_data_profile(df)
                    st.session_state.executive_data_profile_str = data_profile
                    query = st.session_state.sp_user_query
                    rag_context_str = ""
                    if st.session_state.use_rag and st.session_state.retriever_chain:
                        rag_context = "\n---\n".join([doc.page_content for doc in st.session_state.retriever_chain.invoke(query)])
                        st.session_state.executive_rag_context = rag_context
                        rag_context_str = f"\n\n**[RAG 檢索出的相關數據]:**\n{rag_context}"
                    
                    mega_prompt = f"""你是一個頂尖的 AI 商業分析團隊，能夠在一次思考中扮演多個專業經理人角色。你的任務是針對給定的商業問題和數據，生成一份包含三個部分的完整分析報告。

請嚴格按照以下結構和要求進行輸出，使用 Markdown 標題來區分每個部分：
---
### 📊 財務長 (CFO) 分析
在此部分，請完全以財務長的角度思考。專注於財務指標、成本效益、投資回報率、毛利率、潛在的財務風險等。你的分析必須完全基於提供的數據。

### 🏭 營運長 (COO) 分析
在此部分，轉換為營運長的角色。你需要思考，在CFO會提出的財務考量下，營運上是否可行？分析潛在的流程、供應鏈、人力資源或執行風險。你的分析需要務實且著重於可執行性。

### 👑 執行長 (CEO) 最終決策
在此部分，作為CEO，請綜合上述的財務(CFO)和營運(COO)分析。不要重複細節，而是提供一個高層次的戰略總結。最終，給出一個明確、果斷的**決策**（例如：批准、駁回、需要更多資料），並列出 2-3 個最重要的**後續行動建議**。
---
現在，請根據以下資訊開始分析：

**[商業問題]:**
{query}

**[資料統計摘要]:**
{data_profile}{rag_context_str}
"""
                    response = get_gemini_executive_analysis(gemini_api_key, "IntegratedTeam", mega_prompt)
                    st.session_state.sp_final_report = response
                    st.session_state.sp_workflow_stage = "completed"
                    st.session_state.chat_histories[executive_session_id].append({"role": "ai", "content": response})
                    st.rerun()

        # Step 3: 統一的顯示邏輯
        workflow_has_started = (st.session_state.executive_workflow_stage != "idle" or st.session_state.sp_workflow_stage != 'idle')
        if workflow_has_started:
            if st.session_state.get('executive_data_profile_str'):
                with st.expander("查看統計摘要與資料探索" if st.session_state.use_simple_explorer else "查看統計摘要", expanded=False):
                    st.subheader("純文字統計摘要")
                    st.text(st.session_state.executive_data_profile_str)
                    if st.session_state.use_simple_explorer and st.session_state.get("uploaded_file_path"):
                        st.divider(); display_simple_data_explorer(pd.read_csv(st.session_state.uploaded_file_path))
            if st.session_state.use_rag and st.session_state.get('executive_rag_context'):
                with st.expander("查看 RAG 檢索出的相關資料"):
                    st.markdown(st.session_state.executive_rag_context)

            st.divider()
            st.subheader("分析報告與後續對話")
            user_query = st.session_state.get("executive_user_query") or st.session_state.get("sp_user_query")
            if user_query:
                with st.chat_message("human"): st.markdown(user_query)

            if st.session_state.cfo_analysis_text:
                with st.chat_message("ai"): st.markdown(f"### 📊 財務長 (CFO) 分析\n{st.session_state.cfo_analysis_text}")
            if st.session_state.coo_analysis_text:
                with st.chat_message("ai"): st.markdown(f"### 🏭 營運長 (COO) 分析\n{st.session_state.coo_analysis_text}")
            if st.session_state.ceo_summary_text:
                with st.chat_message("ai"): st.markdown(f"### 👑 執行長 (CEO) 最終決策\n{st.session_state.ceo_summary_text}")
            elif st.session_state.sp_final_report:
                with st.chat_message("ai"): st.markdown(st.session_state.sp_final_report)
            
            # Step 4: 自動刷新機制
            if st.session_state.executive_workflow_stage in ["coo_analysis_pending", "ceo_summary_pending"]:
                time.sleep(1) 
                st.rerun()

            # Step 5: 後續追問的對話邏輯 (流程完成後才啟用)
            workflow_completed = (st.session_state.executive_workflow_stage == "completed" or st.session_state.sp_workflow_stage == 'completed')
            if workflow_completed:
                history = st.session_state.chat_histories[executive_session_id]
                if len(history) > 1:
                    for msg in history[1:]:
                        with st.chat_message(msg["role"]): st.markdown(msg["content"])

                if st.session_state.follow_up_stage != "idle":
                    with st.chat_message("ai"):
                        if st.session_state.follow_up_cfo_analysis: st.markdown(f"#### 📊 財務長 (CFO) 分析\n{st.session_state.follow_up_cfo_analysis}")
                        if st.session_state.follow_up_coo_analysis: st.markdown(f"#### 🏭 營運長 (COO) 分析\n{st.session_state.follow_up_coo_analysis}")
                        if st.session_state.follow_up_ceo_analysis: st.markdown(f"#### 👑 執行長 (CEO) 最終決策\n{st.session_state.follow_up_ceo_analysis}")
                        if st.session_state.follow_up_stage in ["cfo_pending", "coo_pending", "ceo_pending"]: st.spinner("專業團隊正在分析您的追問...")
                
                if st.session_state.follow_up_stage == "idle":
                    if user_input := st.chat_input("針對以上報告進行提問..."):
                        st.session_state.chat_histories[executive_session_id].append({"role": "human", "content": user_input})
                        st.session_state.follow_up_query = user_input
                        st.session_state.follow_up_stage = "cfo_pending" 
                        st.session_state.follow_up_cfo_analysis = ""
                        st.session_state.follow_up_coo_analysis = ""
                        st.session_state.follow_up_ceo_analysis = ""
                        st.rerun()

        # 處理追問的後端邏輯
        if st.session_state.follow_up_stage != "idle":
            history_context = "\n\n".join([f"**{msg['role']}:**\n{msg['content']}" for msg in st.session_state.chat_histories[executive_session_id]])
            
            if st.session_state.follow_up_stage == "cfo_pending":
                cfo_prompt = f"""作為財務長(CFO)，請針對使用者提出的最新問題，並根據完整的對話歷史，提供財務角度的專業分析。\n\n[完整的對話歷史]:\n{history_context}\n\n[使用者最新提出的問題]:\n{st.session_state.follow_up_query}\n\n請僅提供你作為 CFO 的分析內容，力求簡潔精確。"""
                st.session_state.follow_up_cfo_analysis = get_gemini_executive_analysis(gemini_api_key, "CFO-FollowUp", cfo_prompt)
                st.session_state.follow_up_stage = "coo_pending"
                st.rerun()

            elif st.session_state.follow_up_stage == "coo_pending":
                coo_prompt = f"""作為營運長(COO)，請針對使用者提出的最新問題，並根據完整的對話歷史，以及剛剛CFO提供的最新財務分析，提供營運角度的策略與風險評估。\n\n[完整的對話歷史]:\n{history_context}\n\n[CFO 對此問題的最新分析]:\n{st.session_state.follow_up_cfo_analysis}\n\n[使用者最新提出的問題]:\n{st.session_state.follow_up_query}\n\n請僅提供你作為 COO 的分析內容，重點在於可行性與執行層面。"""
                st.session_state.follow_up_coo_analysis = get_gemini_executive_analysis(gemini_api_key, "COO-FollowUp", coo_prompt)
                st.session_state.follow_up_stage = "ceo_pending"
                st.rerun()

            elif st.session_state.follow_up_stage == "ceo_pending":
                ceo_prompt = f"""作為執行長(CEO)，請整合所有資訊，包括完整的對話歷史、CFO和COO對最新問題的分析，為使用者的問題提供一個高層次的戰略總結與最終決策。\n\n[完整的對話歷史]:\n{history_context}\n\n[CFO 對此問題的最新分析]:\n{st.session_state.follow_up_cfo_analysis}\n\n[COO 對此問題的最新分析]:\n{st.session_state.follow_up_coo_analysis}\n\n[使用者最新提出的問題]:\n{st.session_state.follow_up_query}\n\n請提供一個簡潔、高層次的總結，並給出明確的後續行動建議。"""
                st.session_state.follow_up_ceo_analysis = get_gemini_executive_analysis(gemini_api_key, "CEO-FollowUp", ceo_prompt)
                full_follow_up_response = f"""### 📊 財務長 (CFO) 分析\n{st.session_state.follow_up_cfo_analysis}\n\n### 🏭 營運長 (COO) 分析\n{st.session_state.follow_up_coo_analysis}\n\n### 👑 執行長 (CEO) 最終決策\n{st.session_state.follow_up_ceo_analysis}"""
                st.session_state.chat_histories[executive_session_id].append({"role": "ai", "content": full_follow_up_response})
                st.session_state.follow_up_stage = "idle" 
                st.rerun()

if __name__ == "__main__":
    main()
