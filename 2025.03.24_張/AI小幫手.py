import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback
import re
import os
import dotenv
import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image
import google.generativeai as genai  # 新增Gemini依赖
from streamlit_ace import st_ace
import time
import matplotlib.font_manager as fm
import matplotlib

import random
import tempfile
from reportlab.lib.pagesizes import letter
from io import BytesIO
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
# --- 初始化设置 ---

font_path = "C:/fonts/msjh.ttc"
fm.fontManager.addfont(font_path)
matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
matplotlib.rcParams['axes.unicode_minus'] = False


dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
LLM_MODELS = [  # 修改后的模型列表
    "gpt-4o",
    "gpt-3.5-turbo-16k",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "models/gemini-2.0-flash"
]

MAX_MESSAGES = 10  # Limit message history

def initialize_client(api_key):
    return OpenAI(api_key=api_key) if api_key else None

def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        st.session_state.debug_logs.append(f"**DEBUG LOG:** {msg}")
        st.write(msg)
        print(msg)

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        st.session_state.debug_errors.append(f"**DEBUG ERROR:** {msg}")
        print(msg)

def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    debug_log(f"Saving file to {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    debug_log(f"Files in {UPLOAD_DIR}: {os.listdir(UPLOAD_DIR)}")
    return file_path

def load_image_base64(image):
    """Convert image to Base64 encoding."""
    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")  # Use PNG for consistency
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        debug_error(f"Error converting image to base64: {e}")
        return ""

def append_message(role, content):
    """Append a message and ensure the total number of messages does not exceed MAX_MESSAGES."""
    st.session_state.messages.append({"role": role, "content": content})
    if len(st.session_state.messages) > MAX_MESSAGES:
        # Remove the oldest messages except the system prompt
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-(MAX_MESSAGES - 1):]
        debug_log("Message history trimmed to maintain token limits.")

def add_user_image(uploaded_file):
    """添加用戶圖片消息到session state"""
    try:
        st.session_state["last_uploaded_filename"] = uploaded_file.name
        current_model = st.session_state.get("selected_model", "").lower()
        use_base64 = "gpt" in current_model  
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.uploaded_image_path = file_path
        
        if use_base64:
            # 為OpenAI模型生成Base64 URL
            image = Image.open(file_path)
            image_base64 = load_image_base64(image)
            image_url = f"data:image/{file_path.split('.')[-1]};base64,{image_base64}"
        else:
            image_url = file_path
        
        image_msg = {
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": "auto"
            }
        }
        
        if use_base64:
            append_message("user", [image_msg])
        else:
            return image_url
        debug_log(f"圖片消息已添加：{image_url[:50]}...")
        
        st.session_state.image_base64 = image_base64 if use_base64 else None
        st.rerun()
        
    except Exception as e:
        st.write(f"添加圖片消息失敗：{str(e)}")
        st.error("圖片處理異常，請檢查日誌")

def reset_session_messages():
    """Clear conversation history from the session."""
    if "messages" in st.session_state:
        st.session_state.pop("messages")
        st.success("Memory cleared!")
        debug_log("Conversation history cleared.")

def execute_code(code, global_vars=None):
    try:
        exec_globals = global_vars if global_vars else {}
        debug_log("Ready to execute the following code:")
        if st.session_state.get("debug_mode", False):
            st.session_state.debug_logs.append(f"```python\n{code}\n```")
        debug_log(f"Executing code with global_vars: {list(exec_globals.keys())}")
        exec(code, exec_globals)
        output = exec_globals.get("output", "(No output returned)")
        debug_log(f"Execution output: {output}")
        return f"Code executed successfully. Output: {output}"
    except Exception as e:
        error_msg = f"Error executing code:\n{traceback.format_exc()}"
        debug_log(f"Execution error: {error_msg}")
        if st.session_state.get("debug_mode", False):
            return error_msg
        else:
            return "Error executing code (hidden in non-debug mode)."

def extract_json_block(response: str) -> str:
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        debug_log(f"Extracted JSON block: {json_str}")
        return json_str
    else:
        debug_log("No JSON block found in response.")
        return response.strip()

# ------------------------------
# 舊版本的 OpenAI 與 Gemini 回覆方法保留（供參考）
# ------------------------------

def get_gemini_response(model_params, max_retries=3):
    """
    整合新版 Gemini 請求方法，支援先讀取圖片 (generate_content) 再進行完整對話 (send_message)。
    流程：
      1) 偵測是否有圖片 (最後一則 user 訊息)
      2) 如果有，先用 generate_content() 取得一段回覆並加到 messages
      3) 最後用 send_message() 把整個 messages 發送給 Gemini，取得最終回覆
    """
    api_key = st.session_state.get("gemini_api_key_input", "")
    debug_log(f"gemini api key: {api_key}")
    if not api_key:
        st.error("未設定 Gemini API 金鑰")
        return ""
    genai.configure(api_key=api_key)
    model_name = model_params.get("model", "gemini-1.5-flash")
    debug_log(f"gemini model: {model_name}")
    model = genai.GenerativeModel(model_name)
    st.session_state.gemini_chat = model.start_chat(history=[])
    debug_log("Gemini chat session created.")

    gemini_system_prompt = {
        "role": "system",
        "content": "請以繁體中文回答，並且所有回覆必須以 #zh-tw 回覆還有回覆時不用在開頭加上#zh-tw。"
    }
    st.session_state.messages.insert(0, gemini_system_prompt)
    debug_log("Gemini system prompt for #zh-t added.")

    if st.session_state.uploaded_image_path:
        debug_log("Detected user message with image, using generate_content() first...")
        gen_content = True
    else:
        gen_content = False

    last_user_msg_with_image = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    last_user_msg_with_image = msg
                    break
        if last_user_msg_with_image:
            break

    if gen_content:
        debug_log("Detected user message with image, using generate_content() first... by gen_content")
        text_parts = []
        image_data = st.session_state.uploaded_image_path
        debug_log("Using generate_content() first... by gen_content")
        try:
            debug_log("entering generate_content()")
            retries = 0
            while retries < max_retries:
                try:
                    debug_log("entering generate_content() try block")
                    imageee = genai.upload_file(path=image_data, display_name="Image")
                    debug_log(f"imageee: {imageee}")
                    response_gc = model.generate_content(["請你繁體中文解讀圖片", imageee])
                    debug_log(f"response_gc: {response_gc.text}")
                    generate_content_reply = response_gc.text
                    debug_log(f"Gemini generate_content reply: {generate_content_reply}")
                    append_message("assistant", generate_content_reply)
                    break
                except genai.GenerationError as e:
                    debug_error(f"generate_content() 失敗: {e}")
                    retries += 1
                    time.sleep(5 * retries)
                except Exception as e:
                    debug_error(f"generate_content() 其他錯誤: {e}")
                    return "generate_content Error"
        except Exception as e:
            debug_error(f"generate_content() 其他錯誤: {e}")

    converted_history = []
    for msg in st.session_state.messages:
        role = msg.get("role")
        parts = []
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    parts.append(f"[Image included, base64 size={len(item['image_url']['url'])} chars]")
                else:
                    parts.append(str(item))
        else:
            parts.append(str(msg["content"]))
        converted_history.append({"role": role, "parts": parts})
    converted_history_json = json.dumps(converted_history, ensure_ascii=False)
    debug_log(f"converted history (json) => {converted_history_json}")
    retries = 0
    wait_time = 5
    while retries < max_retries:
        try:
            debug_log("Calling send_message with entire conversation...")
            response_sm = st.session_state.gemini_chat.send_message(converted_history_json)
            final_reply = response_sm.text.strip()
            debug_log(f"Gemini send_message final reply => {final_reply}")
            return final_reply
        except genai.GenerationError as e:
            debug_error(f"send_message() 生成錯誤: {e}")
            st.warning(f"Gemini 生成錯誤，{wait_time}秒後重試...")
            time.sleep(wait_time)
            retries += 1
            wait_time *= 2
        except Exception as e:
            debug_error(f"send_message() API請求異常: {e}")
            st.error(f"Gemini API請求異常: {e}")
            return ""
    return "請求失敗次數過多，請稍後重試"

def get_openai_response(client, model_params, max_retries=3):
    """处理OpenAI API请求"""
    retries = 0
    wait_time = 5
    model_name = model_params.get("model", "gpt-4-turbo")
    while retries < max_retries:
        try:
            request_params = {
                "model": model_name,
                "messages": st.session_state.messages,
                "temperature": model_params.get("temperature", 0.3),
                "max_tokens": model_params.get("max_tokens", 4096),
                "stream": False
            }
            if any(msg.get("content") and isinstance(msg["content"], list) for msg in st.session_state.messages):
                request_params["max_tokens"] = 4096
                debug_log("Detected multimodal input, adjusting max_tokens")
            response = client.chat.completions.create(**request_params)
            response_content = response.choices[0].message.content.strip()
            debug_log(f"OpenAI原始响应：\n{response_content}")
            return response_content
        except Exception as e:
            if 'rate limit' in str(e).lower() or '429' in str(e):
                debug_error(f"速率限制错误（尝试 {retries+1}/{max_retries}）：{e}")
                st.warning(f"请求过于频繁，{wait_time}秒后重试...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2
            elif 'invalid api key' in str(e).lower():
                debug_error(f"API密钥无效：{e}")
                st.error("OpenAI API密钥无效，请检查后重试")
                return ""
            else:
                debug_error(f"OpenAI请求异常：{str(e)}")
                st.error(f"请求发生错误：{str(e)}")
                return ""
    debug_error(f"超过最大重试次数（{max_retries}次）")
    st.error("请求失败次数过多，请稍后再试")
    return ""



def get_llm_response(client, model_params, max_retries=3):
    """獲取LLM模型回覆（支持OpenAI和Gemini）"""
    model_name = model_params.get("model", "gpt-4-turbo")
    debug_log(f"starting to get llm response...{model_name}")
    if "gpt" in model_name:
        debug_log("GPT")
        return get_openai_response(client, model_params, max_retries)
    elif "gemini" in model_name:
        debug_log("Gemini")
        return get_gemini_response(model_params=model_params, max_retries=max_retries)
    else:
        st.error(f"不支持的模型類型: {model_name}")
        return ""

# ------------------------------
# 新增多模型交叉驗證函數
# ------------------------------

def get_cross_validated_response(model_params_gemini, max_retries=3):
    """
    多模型交叉驗證（僅使用 Gemini 模型驗證）：
    1. 在記憶流中添加一則系統提示，要求 Gemini 使用全部對話記憶進行交叉驗證，
       清楚說明其任務：檢查先前回答的正確性、指出潛在錯誤並提供數據或具體理由支持，
       並對比不同模型的優缺點（若適用）。
    2. 呼叫 Gemini 模型 (例如 gemini-1.5-flash 或 models/gemini-2.0-flash) 獲取回答。
    3. 移除該系統提示後返回 Gemini 的回應結果。
    
    注意：此版本不再向 OpenAI 發送請求。
    """
    # 為 Gemini 模型添加更明確的系統提示，說明其任務內容
    cross_validation_prompt = {
        "role": "system",
        "content": (
            "請仔細閱讀以下全部對話記憶，對先前模型的回答進行交叉驗證。"
            "你的任務是檢查回答的正確性，指出其中可能存在的錯誤或不足，"
            "並提供具體的數據、理由或例子來支持你的分析。"
            "請務必使用繁體中文回答。"
            "在回答時請回答的詳細，內容需要你盡可能的多。"
            "並且越漂亮越好"
        )
    }
    st.session_state.messages.insert(0, cross_validation_prompt)
    
    # 呼叫 Gemini 模型，內部會將完整記憶流作為輸入
    response_gemini = get_gemini_response(model_params_gemini, max_retries)
    
    # 移除剛剛添加的系統提示，以免影響後續對話
    st.session_state.messages.pop(0)
    
    final_response = {
        "gemini_response": response_gemini
    }
    return final_response

def simulate_system_message_addition(final_response, report_type="交叉验证报告"):
    """将最终响应模拟为系统生成消息注入记忆流"""
    if not final_response or not isinstance(final_response, dict):
        debug_error("无效的final_response格式")
        return

    # 提取响应内容
    response_content = final_response.get("gemini_response", "")
    if not response_content:
        debug_error("final_response中缺少gemini_response键")
        return

    # 构造系统消息格式
    formatted_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": f"# 🔍 自动生成{report_type}\n{response_content}"
            }
        ]
    }

    # 调用现有消息追加机制
    append_message(formatted_message["role"], formatted_message["content"])
    debug_log(f"已注入系统消息: {str(formatted_message)[:100]}...")
    
    # 新增：渲染圖表
    if "charts_data" in final_response:
        for chart in final_response["charts_data"]:
            chart_id = chart["id"]
            if chart_id in st.session_state.chart_mapping:
                real_url = st.session_state.chart_mapping[chart_id]
                try:
                    st.image(
                        real_url,
                        caption=f"圖表 {chart_id}",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"圖表 {chart_id} 渲染失敗: {str(e)}")

def _render_integrated_report(report_data):
    """私有函數：渲染報告內容與控制項（完整修正版）"""
    
    # ===================================================================
    # 1. 基本驗證與錯誤處理
    # ===================================================================
    if not isinstance(report_data, dict):
        st.error("❌ 無效的報告數據格式")
        debug_error(f"無效報告數據類型: {type(report_data)}")
        return

    # ===================================================================
    # 2. 文字報告渲染
    # ===================================================================
    if "gemini_response" in report_data and report_data["gemini_response"]:
        try:
            st.markdown("## 📝 整合分析報告")
            st.markdown(report_data["gemini_response"])
        except Exception as e:
            st.error("報告內容渲染失敗")
            debug_error(f"文字渲染錯誤: {str(e)}\n{traceback.format_exc()}")
    else:
        st.warning("⚠️ 報告內容缺失，可能生成失敗或無有效分析結果")

    # ===================================================================
    # 3. 圖表渲染核心邏輯
    # ===================================================================
    if "charts_data" in report_data and report_data["charts_data"]:
        st.markdown("---")
        st.markdown("## 📊 相關圖表")
        
        # 初始化映射表檢查
        if "chart_mapping" not in st.session_state:
            st.error("圖表映射表丟失，請重新生成報告")
            return

        for chart in report_data["charts_data"]:
            # 3.1 數據有效性驗證
            if not isinstance(chart, dict) or "id" not in chart:
                debug_error(f"無效圖表數據格式: {chart}")
                continue
                
            chart_id = chart["id"]
            
            # 3.2 從映射表獲取真實數據
            if chart_id not in st.session_state.chart_mapping:
                st.warning(f"圖表 {chart_id} 數據缺失，可能已被清理")
                debug_error(f"映射表缺少 {chart_id}，當前映射表: {list(st.session_state.chart_mapping.keys())}")
                continue
                
            real_url = st.session_state.chart_mapping[chart_id]

            # 3.3 動態渲染
            try:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    # 統一從映射表加載
                    st.image(
                        real_url,
                        caption=f"圖表 {chart_id}",
                        use_container_width=True,
                        output_format="PNG"  # 確保相容性
                    )
                with col2:
                    # 添加互動元素
                    with st.expander("🔍 原始數據"):
                        st.code(f"圖表ID: {chart_id}\n存儲路徑: {real_url[:100]}...", language="text")
                        
            except Exception as e:
                error_msg = f"圖表 {chart_id} 渲染失敗: {str(e)}"
                st.error(error_msg)
                debug_error(f"{error_msg}\n{traceback.format_exc()}")

    # ===================================================================
    # 4. PDF下載按鈕（強化錯誤處理）
    # ===================================================================
    if report_data.get("pdf_buffer"):
        try:
            # 使用臨時文件確保跨平台相容性
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(report_data["pdf_buffer"].getvalue())
                tmp_path = tmp.name
                
            with open(tmp_path, "rb") as f:
                st.download_button(
                    label="⬇️ 下載完整報告 (PDF)",
                    data=f,
                    file_name="整合分析報告.pdf",
                    mime="application/pdf",
                    help="包含文字分析與所有關聯圖表",
                    key=f"dl_{hash(time.time())}"  # 避免按鈕ID衝突
                )
                
        except Exception as e:
            st.error("PDF文件生成失敗，請重試或聯繫管理員")
            debug_error(f"PDF下載錯誤: {str(e)}\n{traceback.format_exc()}")
            
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp_path = tmp.name  # 獲取完整路徑
                # ... 寫入數據
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    debug_error(f"臨時文件清理失敗: {str(e)}")

def generate_integrated_report(model_params_gemini, max_retries=3):
    """直接分析完整記憶流生成整合報告"""
    default_report = {
        "gemini_response": "報告生成失敗，請檢查日誌",
        "charts_data": [],
        "pdf_buffer": None
    }
    
    try:
        # ==== 提取分析材料 ====
        analysis_materials = {
            "gpt_reports": [],
            "gemini_reports": [],
            "charts": [],
            "code_blocks": []
        }
        
        # 確保圖表映射表存在
        if "chart_mapping" not in st.session_state:
            st.session_state.chart_mapping = {}

        # 讀取 CSV 數據以提供給分析提示
        csv_data = None
        if hasattr(st.session_state, 'uploaded_file_path') and st.session_state.uploaded_file_path:
            try:
                csv_data = pd.read_csv(st.session_state.uploaded_file_path)
                debug_log(f"CSV data loaded for analysis: {csv_data.shape}")
            except Exception as e:
                debug_error(f"Error reading CSV for analysis: {e}")
        
        # 遍历消息历史提取关键信息
        for idx, msg in enumerate(st.session_state.messages):
            # 处理 Assistant 消息：提取模型报告
            if msg["role"] == "assistant":
                content = msg["content"]
                # 标准化内容格式（处理多模态消息）
                if isinstance(content, list):
                    content = " ".join([
                        item.get("text", "") 
                        for item in content 
                        if isinstance(item, dict)
                    ])
                # 识别报告来源
                if "GPT" in content or "gpt" in content.lower():
                    analysis_materials["gpt_reports"].append((idx, content))
                elif "Gemini" in content or "gemini" in content.lower():
                    analysis_materials["gemini_reports"].append((idx, content))
            
            # 处理 User 消息：提取图表但不包含 Base64 数据
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        chart_id = f"chart_{len(analysis_materials['charts']) + 1}"
                        image_url = item["image_url"]["url"]
                        
                        # 將圖片路徑保存到映射表中，但不轉換為 Base64 以節省 Token
                        # 僅當生成 PDF 或顯示圖表時才實際讀取圖片
                        st.session_state.chart_mapping[chart_id] = image_url
                        
                        # 在分析材料中僅保存圖表 ID 和參考標籤，而非完整圖像數據
                        analysis_materials["charts"].append({
                            "id": chart_id,
                            "label": f"Chart {len(analysis_materials['charts']) + 1}"
                        })
                        
                        debug_log(f"已索引圖表: {chart_id} -> {image_url[:30]}...")
            
            # 提取代码片段（独立于角色）
            if "```python" in str(msg["content"]):
                code_blocks = re.findall(r'```python(.*?)```', str(msg["content"]), re.DOTALL)
                if code_blocks:
                    analysis_materials["code_blocks"].append({
                        "position": idx,
                        "code": code_blocks[0].strip()
                    })

        # 構建圖表引用信息（僅使用 ID 而非數據）
        chart_references = []
        for idx, chart in enumerate(analysis_materials["charts"]):
            chart_references.append(f"   - {chart['id']}: 參考標籤 '{chart['label']}'")
        
        # 建構優化的分析提示詞 (不包含 Base64 數據)
        analysis_prompt = f"""
[系統角色]
您現在是資料科學與AI分析專家，請基於完整對話記憶流與CSV資料執行全面分析。您的任務是提供詳盡且資料驅動的報告，突顯關鍵發現和實用見解。

[輸入資料結構]
### 原始訊息流概覽
訊息總數：{len(st.session_state.messages)}
最新GPT報告位置：{analysis_materials['gpt_reports'][-1][0] if analysis_materials['gpt_reports'] else '無'}
最新Gemini報告位置：{analysis_materials['gemini_reports'][-1][0] if analysis_materials['gemini_reports'] else '無'}

### CSV資料基本資訊（必須深入分析）
檔案名稱：{st.session_state.uploaded_file_path.split('/')[-1] if hasattr(st.session_state, 'uploaded_file_path') and st.session_state.uploaded_file_path else '無上傳檔案'}
資料大小：{f"{csv_data.shape[0]} 列 × {csv_data.shape[1]} 欄" if csv_data is not None else '無資料'}
欄位名稱：{', '.join(csv_data.columns.tolist()) if csv_data is not None else '無資料'}
{
    "數值型欄位統計：\n" + "\n".join([f"   - {col}: 平均值={csv_data[col].mean():.2f}, 中位數={csv_data[col].median():.2f}, 標準差={csv_data[col].std():.2f}" 
                                for col in csv_data.select_dtypes(include=['number']).columns]) 
    if csv_data is not None and not csv_data.select_dtypes(include=['number']).empty 
    else "無數值型欄位統計資料"
}
{
    "類別型欄位統計：\n" + "\n".join([f"   - {col}: 唯一值數量={csv_data[col].nunique()}, 最常見值='{csv_data[col].mode()[0]}' (出現{csv_data[col].value_counts().iloc[0]}次)" 
                                for col in csv_data.select_dtypes(include=['object', 'category']).columns]) 
    if csv_data is not None and not csv_data.select_dtypes(include=['object', 'category']).empty 
    else "無類別型欄位統計資料"
}
{
    "缺失值分析：\n" + "\n".join([f"   - {col}: {csv_data[col].isna().sum()}個缺失值 ({csv_data[col].isna().mean()*100:.1f}%)" 
                               for col in csv_data.columns if csv_data[col].isna().any()]) 
    if csv_data is not None and csv_data.isna().any().any() 
    else "無缺失值"
}

### 可驗證素材
1. 分析圖表（共{len(analysis_materials['charts'])}張）：
{chr(10).join(chart_references)}

2. 程式碼片段（共{len(analysis_materials['code_blocks'])}段）：
{chr(10).join([f"   - 位置{cb['position']}: {cb['code'][:50]}..." for cb in analysis_materials['code_blocks']])}

3. 先前分析總結
- 請從對話歷史中提取所有模型之前的分析結論
- 確保這些先前的發現不會在新報告中丟失
- 特別注意之前生成的數據指標和趨勢預測

[核心任務]
執行三維度交叉驗證並提供深入CSV資料分析：

🔍 CSV數據深入分析（必須完成）
   - 提供上傳CSV文件的全面分析，識別關鍵模式、相關性和異常
   - 交叉分析數值型和類別型欄位之間的關係
   - 挖掘並顯示不容易被察覺的數據洞見
   - 基於CSV數據提供具體的建議和行動方案

📊 模型分析比較
   - 比對GPT與Gemini在關鍵結論點的差異
   - 識別矛盾級別（輕微/中度/嚴重）
   - 範例：在銷售預測中，GPT預測Q3增長{{x}}%而Gemini預測{{y}}%，差異源於...

⚙️ 證據鏈完整性審查
   - 驗證圖表與結論的對應關係（請通過圖表ID引用，如：圖表 chart_1 顯示...）
   - 檢查程式碼片段是否支持分析結論
   - 範例：在位置{analysis_materials['code_blocks'][0]['position'] if analysis_materials['code_blocks'] else 'N/A'}的程式碼中...

[強制格式]
```markdown
# 整合資料分析報告

## CSV檔案綜合分析
- 檔案總覽：詳述檔案特徵和關鍵結構
- 數據關鍵發現：請以段落式論述而非表格方式呈現至少3項關鍵發現，每項發現需包括：
  - 分析維度：具體分析的角度或指標
  - 數據證據：支持這一發現的數據事實，含具體數值
  - 業務影響：該發現對相關業務可能產生的影響
  - 建議行動：基於發現提出的具體可執行建議

## 核心差異分析
針對GPT與Gemini分析差異，請以段落式論述而非表格方式呈現以下內容：
1. 【差異一】：詳細描述差異維度、各模型觀點及整合結論
2. 【差異二】：詳細描述差異維度、各模型觀點及整合結論
3. 【差異三】：詳細描述差異維度、各模型觀點及整合結論

## 數據洞察
### 1. 趨勢與模式
- 時間序列發現：詳細分析時間相關的趨勢、周期性和變化點
- 相關性分析：詳細分析各變量間的關聯性，包括強相關和弱相關關係
- 異常點識別：明確指出數據中的異常點，分析成因並評估影響

### 2. 預測與建議
- 趨勢預測：基於歷史數據和模式對未來趨勢作出具體預測
- 業務建議：提供3-5條具體可行的業務建議，每條包含理由和預期收益
- 後續分析方向：至少3個值得進一步探索的分析方向

## 方法論評估
- 分析技術比較：詳細比較不同分析方法的優缺點和適用場景
- 模型準確性：評估各分析模型的準確度、置信區間和可靠性
- 交叉驗證結果：描述交叉驗證的方法與結果，以及對結論可信度的影響

## 風險與優化建議
- 資料問題：明確指出數據收集、處理或分析中的潛在問題
- 分析局限性：坦誠分析本次分析的局限性和潛在偏差
- 改進建議：至少3條提升數據質量或分析效果的具體建議

## 優化路線
1. 立即行動：詳細説明可立即實施的改進措施，包括具體步驟和預期效果
2. 中期計劃：分析中期（3-6個月）優化方向和具體實施計劃
3. 長期戰略：提出長期數據分析戰略建議，包括資源需求和預期收益
```

[特別指令]
1. 必須使用訊息位置標記來源（如：@msg_12）
2. 禁用模糊詞彙（"可能"、"大概"等），需明確結論
3. 引用圖表時，只需使用圖表ID（如 chart_1, chart_2 等），無需描述圖表內容
4. 新增驗證哈希：{{"hash": "{hash(str(st.session_state.messages))}"}}
5. 必須詳細分析上傳的CSV資料，並將其作為報告的核心部分
6. 禁止生成空泛的結論，每個結論必須有CSV數據支持
7. 確保報告結構完整，包括所有標題部分
8. 禁止使用表格格式進行呈現，使用段落式論述代替
9. 報告必須有足夠深度，每個部分至少包含3-5個段落的詳細分析
10. 適當使用要點符號和編號呈現多個觀點，但主要結論需用完整段落展開
"""
        # 生成 Gemini 响应
        cross_validation_prompt = {
            "role": "system",
            "content": analysis_prompt
        }
        
        # 插入系統提示並獲取響應
        st.session_state.messages.insert(0, cross_validation_prompt)
        response_gemini = get_gemini_response(model_params_gemini, max_retries)
        st.session_state.messages.pop(0)

        # 构建最终报告 - 此時從映射表查詢圖片數據
        final_report = {
            "gemini_response": response_gemini,
            "charts_data": [],
            "pdf_buffer": None
        }
        
        # 將圖表引用轉換為完整圖表數據（僅在報告渲染階段）
        for chart in analysis_materials["charts"]:
            chart_id = chart["id"]
            if chart_id in st.session_state.chart_mapping:
                final_report["charts_data"].append({
                    "id": chart_id,
                    "label": chart.get("label", f"Chart {chart_id}")
                })
        
        # 生成 PDF
        pdf_buffer = _generate_pdf(final_report)
        final_report["pdf_buffer"] = pdf_buffer
        
        # 將最終報告保存到session_state中，供新頁面使用
        st.session_state.integrated_report = final_report
        
        # 在當前頁面渲染報告
        _render_integrated_report(final_report)
        
        # 添加前往新頁面的按鈕
        st.markdown("---")
        st.markdown("## 🔄 報告頁面")
        st.info("您可以在獨立報告頁面查看完整報告內容")
        
        return final_report
        
    except Exception as e:
        debug_error(f"生成整合報告失敗: {str(e)}\n{traceback.format_exc()}")
        st.error("報告生成異常，請檢查日誌")
        return default_report

def _generate_pdf(report_data):
    """将报告内容与图表生成PDF"""


    # 註冊中文字體 - 使用系統自帶的中文字體
    try:
        # 嘗試註冊Windows下的微軟雅黑字體
        pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))
        cn_font_name = 'SimSun'
    except:
        try:
            # 嘗試註冊Arial Unicode MS (廣泛支持Unicode字符)
            pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
            cn_font_name = 'Arial'
        except:
            # 如果上述字體都無法找到，使用默認字體
            cn_font_name = 'Helvetica'
            debug_error("無法找到支持中文的字體，PDF中的中文可能顯示不正確")
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    
    # 建立文檔樣式
    styles = getSampleStyleSheet()
    normal_style = ParagraphStyle(
        'NormalWithCN',
        parent=styles['Normal'],
        fontName=cn_font_name,
        fontSize=10,
        leading=14,  # 行間距
        wordWrap='CJK',  # 支援中日韓文字換行
    )
    
    title_style = ParagraphStyle(
        'TitleWithCN',
        parent=styles['Heading1'],
        fontName=cn_font_name,
        fontSize=16,
        leading=20,
        alignment=TA_LEFT,
    )
    
    # 添加標題
    story.append(Paragraph("整合分析報告", title_style))
    story.append(Spacer(1, 12))
    
    # 檢查必要字段
    if "gemini_response" not in report_data:
        debug_error("PDF生成失敗：缺少 gemini_response 字段")
        story.append(Paragraph("報告生成失敗，缺少必要數據", normal_style))
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    # 處理報告内容 (使用Paragraph來支持換行和格式化)
    text_content = report_data["gemini_response"] or "（報告内容為空）"
    
    # 處理Markdown格式
    text_content = text_content.replace('\n\n', '<br/><br/>')
    text_content = text_content.replace('\n', '<br/>')
    
    # 添加文本内容
    story.append(Paragraph(text_content, normal_style))
    
    # 插入圖表（僅處理有效數據）
    if report_data.get("charts_data"):
        story.append(Spacer(1, 20))
        story.append(Paragraph("📊 相關圖表", title_style))
        story.append(Spacer(1, 12))
    
    for idx, chart in enumerate(report_data.get("charts_data", [])):
        chart_id = chart["id"]
        if chart_id in st.session_state.chart_mapping:
            chart_fig = st.session_state.chart_mapping[chart_id]
            try:
                # 使用Paragraph添加圖表標題
                chart_title = f"圖表 {idx+1}: {chart.get('label', chart_id)}"
                story.append(Spacer(1, 10))
                story.append(Paragraph(chart_title, normal_style))
                
                # 處理不同類型的圖表數據
                img_bytes = None
                
                # 檢查數據類型並相應處理
                try:
                    if hasattr(chart_fig, 'write_image'):  # Plotly圖表對象
                        debug_log(f"處理Plotly圖表: {chart_id}")
                        img_bytes = BytesIO()
                        chart_fig.write_image(img_bytes, format='png')
                        img_bytes.seek(0)
                    elif isinstance(chart_fig, str):  # 字符串（URL或base64）
                        if chart_fig.startswith('data:image'):
                            debug_log(f"處理Base64圖像: {chart_id}")
                            # 處理base64編碼的圖像
                            header, data = chart_fig.split(",", 1)
                            img_bytes = BytesIO(base64.b64decode(data))
                        else:
                            # 記錄不支持的字符串格式
                            debug_log(f"圖表格式不支持: {chart_id} - 字符串但非Base64")
                            story.append(Paragraph(f"圖表 {chart_id} 格式不支持，無法在PDF中顯示", normal_style))
                            continue
                    else:
                        # 使用默認方式嘗試處理
                        debug_log(f"嘗試默認處理圖表: {chart_id} - 類型 {type(chart_fig)}")
                        img_bytes = BytesIO()
                        chart_fig.write_image(img_bytes, format='png')
                        img_bytes.seek(0)
                except Exception as e:
                    debug_error(f"處理圖表數據失敗: {chart_id} - {str(e)}")
                    story.append(Paragraph(f"圖表 {chart_id} 處理失敗: {str(e)}", normal_style))
                    continue
                
                # 添加圖片到文檔
                if img_bytes:
                    from reportlab.platypus import Image
                    # 調整圖片大小，確保不超過頁面寬度，設定最大寬度為400（小於頁面寬度456）
                    img = Image(img_bytes, width=400, height=300, kind='proportional')
                    story.append(img)
                    story.append(Paragraph(f"圖表ID: {chart_id}", normal_style))
                    story.append(Spacer(1, 10))
                else:
                    story.append(Paragraph(f"圖表 {chart_id} 無法獲取圖像數據", normal_style))
            except Exception as e:
                debug_error(f"PDF插入圖表失敗: {str(e)}")
                story.append(Paragraph(f"圖表 {chart_id} 處理失敗: {str(e)}", normal_style))
    
    # 生成PDF
    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        error_msg = f"PDF生成失敗: {str(e)}"
        traceback_msg = traceback.format_exc()
        debug_error(error_msg)
        debug_error(traceback_msg)
        st.error(error_msg)
        st.code(traceback_msg, language="python")
        return None

def _render_integrated_report(report_data):
    """渲染報告內容與控制項"""
    if not isinstance(report_data, dict):
        st.error("无效的报告数据格式")
        return
    
    # 渲染文字报告
    if "gemini_response" in report_data:
        st.markdown(report_data["gemini_response"])
    else:
        st.warning("报告内容缺失")
    
    # ===================================================================
    # 3. 圖表渲染 (PDF友好版本)
    # ===================================================================
    if "charts_data" in report_data and report_data["charts_data"]:
        st.markdown("---")
        st.markdown("## 📊 相關圖表")
        
        # 初始化映射表檢查
        if "chart_mapping" not in st.session_state:
            st.error("圖表映射表未初始化")
            return

        for idx, chart in enumerate(report_data["charts_data"]):
            # 數據有效性驗證
            if not isinstance(chart, dict) or "id" not in chart:
                debug_error(f"無效圖表數據格式: {chart}")
                continue
                
            chart_id = chart["id"]
            
            # 從映射表獲取真實數據
            if chart_id not in st.session_state.chart_mapping:
                st.warning(f"圖表 {chart_id} 數據缺失")
                continue
                
            real_url = st.session_state.chart_mapping[chart_id]

            # 圖表渲染 - 單列清晰版面
            try:
                # 標題和圖表編號
                st.subheader(f"圖表 {idx+1}: {chart.get('label', chart_id)}")
                
                # 圖片顯示 - 固定寬度適合PDF
                st.image(real_url, caption=f"圖表 {chart_id}", use_container_width=False, width=650, output_format="PNG")
                
                # 圖表元數據 - 簡潔格式
                st.caption(f"圖表ID: {chart_id}")
                
                # 分隔線確保PDF中的圖表間距
                if idx < len(report_data["charts_data"]) - 1:
                    st.markdown("---")
                    
            except Exception as e:
                st.error(f"圖表 {chart_id} 渲染失敗: {str(e)}")
                debug_error(f"圖表渲染錯誤: {str(e)}")
    # PDF下载按钮
    if report_data.get("pdf_buffer"):
        try:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(report_data["pdf_buffer"].getvalue())
                    tmp_path = tmp.name
                    
                with open(tmp_path, "rb") as f:
                    st.download_button(
                        label="⬇️ 下載完整報告 (PDF)",
                        data=f,
                        file_name="整合分析報告.pdf",
                        mime="application/pdf",
                        help="包含文字分析與報告圖表",
                        key=f"dl_{hash(time.time())}"  # 避免按鈕ID衝突
                    )
            finally:
                # 確保臨時文件被清理
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        debug_error(f"臨時文件清理失敗: {str(e)}")
                
        except Exception as e:
            st.error("PDF文件生成失敗，請重試或聯繫管理員")
            debug_error(f"PDF生成異常: {str(e)}")

def generate_questions():
    """基于对话历史生成3个后续问题"""
    if not st.session_state.messages:
        return []
    
    # 格式化最近的消息歷史
    recent_messages = st.session_state.messages[-10:]
    formatted_history = ""
    for msg in recent_messages:
        role = msg["role"]
        content = msg["content"]
        
        # 處理多模態消息內容
        if isinstance(content, list):
            content = " ".join([item.get("text", "") for item in content if isinstance(item, dict) and "text" in item])
        
        formatted_history += f"{role.upper()}: {content}\n\n"
    
    # 构建生成问题的prompt
    prompt = f"""
请基于以下对话历史，生成3个用户可能继续提出的问题。要求：
1. 问题需直接相关于对话内容
2. 使用繁体中文
3. 用数字编号列表格式返回，不要其他内容

当前对话历史：
{formatted_history}
"""
    
    # 使用现有模型生成问题
    try:
        # 優先使用 session 中的 API key，如果沒有再從環境變量獲取
        api_key = st.session_state.get("openai_api_key_input", os.getenv("OPENAI_API_KEY", ""))
        if not api_key:
            debug_error("缺少 OpenAI API 密鑰，無法生成問題建議")
            return []
            
        client = initialize_client(api_key)
        
        # 創建正確的 API 請求格式
        messages = [{"role": "user", "content": prompt}]
        response = get_openai_response(
            client,
            {
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 200,
                "messages": messages
            }
        )
        
        # 提取问题列表
        questions = []
        for line in response.split('\n'):
            if re.match(r'^\d+[\.、]', line.strip()):
                question = re.sub(r'^\d+[\.、]\s*', '', line).strip()
                questions.append(question)
                if len(questions) >= 3:
                    break
        valid_questions = [q for q in questions if len(q.strip()) > 0]
        return valid_questions[:3]  # 确保最多返回3个问题
    
    except Exception as e:
        debug_error(f"生成問題失敗: {str(e)}")
        return []

def show_question_suggestions():
    """在输入框上方显示问题建议"""
    if "generated_questions" in st.session_state and st.session_state.generated_questions:
        st.markdown("""
        <style>
            .question-box {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 12px;
                margin: 8px 0;
                cursor: pointer;
                transition: all 0.2s;
            }
            .question-box:hover {
                background-color: #f5f5f5;
            }
            .source-count {
                color: #666;
                font-size: 0.8em;
                float: right;
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("**📚 推荐问题**")
        for i, q in enumerate(st.session_state.generated_questions):
            # 模拟来源数量（实际可根据需求从数据获取）
            source_count = random.randint(1, 5)  
            
            # 使用columns创建卡片式布局
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                clicked = st.button(
                    q,
                    key=f"sug_q_{i}",
                    use_container_width=True,
                    help="点击提交此问题",
                    type="secondary"
                )
            with col2:
                st.markdown(f'<div class="source-count">{source_count} 个来源</div>', unsafe_allow_html=True)
            
            if clicked:
                if "question_input" in st.session_state:
                    del st.session_state.question_input
                
                # 直接模拟用户消息
                append_message("user", q)
                debug_log(f"已模拟用户提问: {q}")
                
                # 強制刷新并跳转到消息处理
                st.session_state.need_process_question = True  # 新增状态标记
                st.rerun()
# 新增统一的问题处理函数
def process_question():
    """处理自动生成的问题"""
    try:
        # 获取最后一条用户消息
        last_user_msg = next(
            msg for msg in reversed(st.session_state.messages)
            if msg["role"] == "user"
        )
        
        # 調用模型生成回复
        with st.spinner("正在生成回答..."):
            # 優先使用 session 中的 API key，如果沒有再從環境變量獲取
            api_key = st.session_state.get("openai_api_key_input", os.getenv("OPENAI_API_KEY", ""))
            if not api_key:
                st.error("缺少 OpenAI API 密鑰，無法生成回答")
                debug_error("缺少 OpenAI API 密鑰，無法生成回答")
                return
                
            client = initialize_client(api_key)
            
            # 取最近 10 條消息作為上下文，包括最新的用戶問題
            recent_messages = st.session_state.messages[-10:]
            
            # 創建 API 請求所需的消息格式
            formatted_messages = []
            for msg in recent_messages:
                content = msg["content"]
                # 處理多模態消息
                if isinstance(content, list):
                    # 過濾出純文本內容
                    text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and "text" in item]
                    content = " ".join(text_parts)
                
                formatted_messages.append({
                    "role": msg["role"],
                    "content": content
                })
            
            # 使用合適的模型參數
            model_name = st.session_state.get("selected_model", "gpt-4o")
            model_params = {
                "model": model_name,
                "temperature": 0.7,
                "messages": formatted_messages
            }
            
            # 調用 OpenAI API 獲取回覆
            response = get_openai_response(client, model_params)
            
            # 添加助手回复
            append_message("assistant", response)
            
            # 渲染消息
            with st.chat_message("assistant"):
                st.write(response)
                
            # 生成新的問題建議
            st.session_state.generated_questions = generate_questions()
                
    except StopIteration:
        debug_error("未找到用户问题消息")
    except Exception as e:
        debug_error(f"问题处理失败: {str(e)}")
        st.error(f"回答生成失敗: {str(e)}")







# ------------------------------
# 主應用入口
# ------------------------------

def main():
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="wide")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory + 🖋️ Canvas (With Debug & Deep Analysis)")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ace_code" not in st.session_state:
        st.session_state.ace_code = ""
    if "editor_location" not in st.session_state:
        st.session_state.editor_location = "Sidebar"
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "uploaded_image_path" not in st.session_state:
        st.session_state.uploaded_image_path = None
    if "image_base64" not in st.session_state:
        st.session_state.image_base64 = None
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "deep_analysis_mode" not in st.session_state:
        st.session_state.deep_analysis_mode = False
    if "second_response" not in st.session_state:
        st.session_state.second_response = ""
    if "third_response" not in st.session_state:
        st.session_state.third_response = ""
    if "deep_analysis_image" not in st.session_state:
        st.session_state.deep_analysis_image = None
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    if "debug_errors" not in st.session_state:
        st.session_state.debug_errors = []
    if "thinking_protocol" not in st.session_state:
        st.session_state.thinking_protocol = None
    if "gemini_ai_chat" not in st.session_state:
        st.session_state.gemini_ai_chat = None
    if "gemini_ai_history" not in st.session_state: 
        st.session_state.gemini_ai_history = []
    if "generated_questions" not in st.session_state:
        st.session_state.generated_questions = []
    with st.sidebar:
        st.subheader("🔑 API Key Settings")
        default_openai_key = os.getenv("OPENAI_API_KEY", "")
        openai_api_key = st.text_input("OpenAI API Key", value=default_openai_key, type="password")
        default_gemini_key = os.getenv("GEMINI_API_KEY", "")
        gemini_api_key = st.text_input("Gemini API Key", 
                                       value=default_gemini_key, 
                                       type="password",
                                       key="gemini_api_key")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if gemini_api_key:
            st.session_state["gemini_api_key_input"] = gemini_api_key 

        selected_model = st.selectbox(
            "選擇模型", 
            LLM_MODELS, 
            index=0, 
            key="selected_model"
        )
        
        if "selected_model" in st.session_state:
            current_model = st.session_state.selected_model.lower()
            if "gemini" in current_model:
                gemini_key = os.getenv("GEMINI_API_KEY") or st.session_state.get("gemini_api_key")
                if not gemini_key:
                    st.error("使用Gemini模型需在下方輸入API金鑰 🔑")
                    st.stop()
            elif "gpt" in current_model:
                openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
                if not openai_key:
                    st.error("使用OpenAI模型需在下方輸入API金鑰 🔑")
                    st.stop()

        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
        st.session_state.deep_analysis_mode = st.checkbox("Deep Analysis Mode", value=True)

        if "memory" not in st.session_state:
            st.session_state.memory = []

        if "conversation_initialized" not in st.session_state:
            openai_api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key_input")
            if openai_api_key or gemini_api_key:
                client = initialize_client(openai_api_key)
                st.session_state.conversation_initialized = True
                st.session_state.messages = []
                debug_log("Conversation initialized with OpenAI client.")
            else:
                st.warning("⬅️ 請在側邊欄輸入OpenAI API金鑰以初始化聊天機器人")

        if st.session_state.debug_mode:
            debug_log(f"Currently using model => {selected_model}")

        if st.button("🗑️ Clear Memory"):
            st.session_state.memory = []
            st.session_state.messages = []
            st.session_state.ace_code = ""
            st.session_state.uploaded_file_path = None
            st.session_state.uploaded_image_path = None
            st.session_state.image_base64 = None
            st.session_state.deep_analysis_mode = False
            st.session_state.second_response = ""
            st.session_state.third_response = ""
            st.session_state.deep_analysis_image = None
            st.session_state.debug_logs = []
            st.session_state.debug_errors = []
            st.session_state.thinking_protocol = None
            st.session_state.gemini_ai_chat = None
            st.success("Memory cleared!")
            debug_log("Memory has been cleared.")

        st.subheader("🧠 Memory State")
        if st.session_state.messages:
            memory_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            st.text_area("Current Memory", value=memory_content, height=200)
            debug_log(f"Current memory content: {memory_content}")
        else:
            st.text_area("Current Memory", value="No messages yet.", height=200)
            debug_log("No messages in memory.")

        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        csv_data = None
        if uploaded_file:
            st.session_state.uploaded_file_path = save_uploaded_file(uploaded_file)
            debug_log(f"Uploaded file path: {st.session_state.uploaded_file_path}")
            try:
                csv_data = pd.read_csv(st.session_state.uploaded_file_path)
                st.write("### Data Preview")
                st.dataframe(csv_data)
                debug_log(f"CSV Data Columns: {list(csv_data.columns)}")
            except Exception as e:
                csv_columns = "Unable to read columns"
                if st.session_state.debug_mode:
                    st.error(f"Error reading CSV: {e}")
                debug_log(f"Error reading CSV: {e}")
        else:
            csv_columns = "No file uploaded"
            debug_log("No CSV file uploaded.")

        st.subheader("🖼️ Upload an Image")
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"], key="image_uploader")
        if uploaded_image:
            st.session_state.uploaded_image = add_user_image(uploaded_image)
            debug_log(f"Uploaded image path: {st.session_state.uploaded_image}")

            
        st.subheader("客制化提示詞")  # 新增區塊
        if st.button("✨ 生成相關問題", key="generate_questions_btn"):
            with st.spinner("正在生成推薦問題..."):
                st.session_state.generated_questions = generate_questions()

        st.subheader("🧠 Upload Thinking Protocol")
        uploaded_thinking_protocol = st.file_uploader("Choose a thinking_protocol.md file:", type=["md"], key="thinking_protocol_uploader")
        if uploaded_thinking_protocol:
            try:
                thinking_protocol_content = uploaded_thinking_protocol.read().decode("utf-8")
                st.session_state.thinking_protocol = thinking_protocol_content
                append_message("user", thinking_protocol_content)
                st.success("Thinking Protocol uploaded successfully!")
                debug_log("Thinking Protocol uploaded and added to messages.")
            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"Error reading Thinking Protocol: {e}")
                debug_log(f"Error reading Thinking Protocol: {e}")

        st.subheader("Editor Location")
        location = st.radio(
            "Choose where to display the editor:",
            ["Main", "Sidebar"],
            index=0 if st.session_state.editor_location == "Main" else 1
        )
        st.session_state.editor_location = location
        debug_log(f"Editor location set to: {st.session_state.editor_location}")

        with st.expander("🛠️ 调试与会话信息", expanded=False):
            if st.session_state.debug_mode:
                st.subheader("调试日志")
                if st.session_state.debug_logs:
                    debug_logs_combined = "\n".join(st.session_state.debug_logs)
                    st.text_area("Debug Logs", value=debug_logs_combined, height=200)
                else:
                    st.write("没有调试日志。")
                st.subheader("调试错误")
                if st.session_state.debug_errors:
                    debug_errors_combined = "\n".join(st.session_state.debug_errors)
                    st.text_area("Debug Errors", value=debug_errors_combined, height=200)
                else:
                    st.write("没有调试错误。")
            st.subheader("会话信息 (messages.json)")
            if "messages" in st.session_state:
                messages_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=4)
                st.text_area("messages.json", value=messages_json, height=300)
                st.download_button(
                    label="📥 下载 messages.json",
                    data=messages_json,
                    file_name="messages.json",
                    mime="application/json"
                )
                st.markdown("---")
                if st.button("📄 显示原始消息"):
                    st.subheader("🔍 原始消息内容")
                    st.json(st.session_state.messages)
            else:
                st.write("没有找到 messages。")

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_url = item["image_url"]["url"]
                        st.image(image_url, caption="📷 上傳的圖片", use_container_width=True)
                        debug_log(f"Displaying image from {message['role']}: {image_url}")
                    else:
                        st.write(item)
                        debug_log(f"Displaying non-image content from {message['role']}: {item}")
            elif isinstance(message["content"], str) and "```python" in message["content"]:
                code_match = re.search(r'```python\s*(.*?)\s*```', message["content"], re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                    st.code(code, language="python")
                    debug_log(f"Displaying code from {message['role']}: {code}")
                else:
                    st.write(message["content"])
                    debug_log(f"Displaying message {idx} from {message['role']}: {message['content']}")
    if st.session_state.get("need_process_question", False):
        st.session_state.need_process_question = False
        process_question()
    user_input = st.chat_input("Hi! Ask me anything...", key="main_input")
    show_question_suggestions()  
    if user_input:
        append_message("user", user_input)
        with st.chat_message("user"):
            st.write(user_input)
            debug_log(f"User input added to messages: {user_input}")

        with st.spinner("Thinking..."):
            try:
                if openai_api_key or gemini_api_key:
                    client = initialize_client(openai_api_key)
                else:
                    raise ValueError("OpenAI API Key is not provided.")
                debug_log(f"Uploaded file path: {st.session_state.uploaded_file_path}")
                debug_log(f"Uploaded image path: {st.session_state.uploaded_image_path}")

                if not any(msg["role"] == "system" for msg in st.session_state.messages):
                    system_prompt = "You are an assistant that helps with data analysis."
                    append_message("system", system_prompt)
                    debug_log("System prompt added to messages.")

                if "question_input" in st.session_state:
                    user_input = st.session_state.pop("question_input")
                    if user_input:
                        append_message("user", user_input)

                if st.session_state.uploaded_image_path is not None and st.session_state.image_base64:
                    prompt = user_input
                    debug_log("User input with image data already appended.")
                else:
                    if st.session_state.uploaded_file_path is not None:
                        try:
                            df_temp = pd.read_csv(st.session_state.uploaded_file_path)
                            csv_columns = ", ".join(df_temp.columns)
                            debug_log(f"CSV columns: {csv_columns}")
                        except Exception as e:
                            csv_columns = "Unable to read columns"
                            if st.session_state.debug_mode:
                                st.error(f"Error reading CSV: {e}")
                            debug_log(f"Error reading CSV: {e}")
                    else:
                        csv_columns = "No file uploaded"
                        debug_log("No CSV file uploaded.")

                    if st.session_state.uploaded_file_path is not None and csv_columns != "No file uploaded":
                        prompt = f"""Please respond with a JSON object in the format:
{{
    "content": "這是我的觀察跟分析: {{analysis}}",
    "code": "import pandas as pd\\nimport streamlit as st\\nimport matplotlib.pyplot as plt\\n# Read CSV file (use st.session_state.uploaded_file_path variable)\\ndata = pd.read_csv(st.session_state.uploaded_file_path)\\n\\n# Add your plotting or analysis logic here\\n\\n# For example, to display a plot using st.pyplot():\\n# fig, ax = plt.subplots()\\n# ax.scatter(data['colA'], data['colB'])\\n# st.pyplot(fig)"
}}
Important:
1) 必須使用 st.session_state.uploaded_file_path 作為 CSV 路徑 (instead of a hardcoded path)
2) Must use st.pyplot() to display any matplotlib figure
3) Return only valid JSON (escape any special characters if needed)
4) 請確保圖表中的字體已經套用以下字型：字型位置：{font_path}。請注意，這是必要的步驟，確保所有標題、標籤等文字都以指定字型顯示。
Based on the request: {user_input}.
Available columns: {csv_columns}.

!重要!需求共有3
1.圖表的顏色考慮使用其他的，不要使用預設
2.在生成代碼時需要考慮plot的美觀性
3.然後請使用繁體中文回應
"""
                        debug_log("Prompt constructed for CSV input with JSON response.")
                        append_message("system", prompt)
                        debug_log("System prompt appended to messages.")
                    else:
                        prompt = f"Please answer this question entirely in Traditional Chinese: {user_input}"
                        debug_log("Prompt constructed for plain text input.")
                        append_message("system", prompt)
                        debug_log("Plain text system prompt appended to messages.")

                model_params = {
                    "model": selected_model,
                    "temperature": 0.5,
                    "max_tokens": 16384
                }
                response_content = get_llm_response(client, model_params)
                debug_log(f"Full assistant response: {response_content}")

                if response_content:
                    append_message("assistant", response_content)
                    with st.chat_message("assistant"):
                        st.write(response_content)
                        debug_log(f"Assistant response added to messages: {response_content}")

                    json_str = extract_json_block(response_content)
                    try:
                        response_json = json.loads(json_str)
                        debug_log("JSON parsing successful.")
                    except Exception as e:
                        debug_log(f"json.loads parsing error: {e}")
                        debug_error(f"json.loads parsing error: {e}")
                        response_json = {"content": json_str, "code": ""}
                        debug_log("Fallback to raw response for content.")

                    content = response_json.get("content", "Here is my analysis:")
                    append_message("assistant", content)
                    code = response_json.get("code", "")
                    if code:
                        code_block = f"```python\n{code}\n```"
                        append_message("assistant", code_block)
                        with st.chat_message("assistant"):
                            st.code(code, language="python")
                        st.session_state.ace_code = code
                        debug_log("ace_code updated with new code.")

                    if st.session_state.deep_analysis_mode and code:
                        st.write("### [Deep Analysis] Automatically executing the generated code and sending the chart to GPT-4o for analysis...")
                        debug_log("Deep analysis mode activated.")
                        global_vars = {
                            "uploaded_file_path": st.session_state.uploaded_file_path,
                            "uploaded_image_path": st.session_state.uploaded_image_path,
                        }
                        exec_result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                        st.write("#### Execution Result")
                        st.text(exec_result)
                        debug_log(f"Execution result: {exec_result}")
                        fig = plt.gcf()
                        buf = BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
                        st.session_state.deep_analysis_image = chart_base64
                        debug_log("Chart has been converted to base64.")
                        prompt_2 = f"""基於圖片給我更多資訊"""
                        debug_log(f"Deep Analysis Prompt: {prompt_2}")
                        append_message("user", prompt_2)
                        debug_log("Deep analysis prompt appended to messages.")
                        image_content = [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{chart_base64}"}
                        }]
                        append_message("user", image_content)
                        second_raw_response = get_llm_response(client, model_params)
                        debug_log(f"Deep analysis response: {second_raw_response}")
                        if second_raw_response:
                            append_message("assistant", second_raw_response)
                            st.session_state.second_response = second_raw_response
                            with st.chat_message("assistant"):
                                st.write(second_raw_response)
                                debug_log(f"Deep analysis response added to messages: {second_raw_response}")
                            prompt_3 = f"""
First response content: {content}
Second response chart analysis content: {second_raw_response}

請把前兩次的分析內容做分析總結，有數據的話就顯示得漂亮一點，主要是需要讓使用者感到很厲害。並且以繁體中文作為回答用的語言。
另外需要解釋傳給妳的圖表，以一個沒有資料科學背景的小白解釋我所傳的圖表。還有根據第二次的圖表分析得出的結論，直接預測之後的走向，例如:"之後這個數值的走向會呈現向上的趨勢"等...
不要跟使用者說甚麼妳可以使用RFM分析，交叉分析之類的方法。我需要妳直接預測之後的走向，比如往上還是往下。
"""
                            debug_log(f"Final Summary Prompt: {prompt_3}")
                            append_message("user", prompt_3)
                            debug_log("Final summary prompt appended to messages.")
                            third_raw_response = get_llm_response(client, model_params)
                            debug_log(f"Final summary response: {third_raw_response}")
                            if third_raw_response:
                                append_message("assistant", third_raw_response)
                                st.session_state.third_response = third_raw_response
                                with st.chat_message("assistant"):
                                    st.write(third_raw_response)
                                    debug_log(f"Final summary response added to messages: {third_raw_response}")
                                st.write("#### [Deep Analysis] Chart:")
                                try:
                                    img_data = base64.b64decode(st.session_state.deep_analysis_image)
                                    st.image(img_data, caption="Chart generated from deep analysis", use_container_width=True)
                                    debug_log("Deep analysis chart displayed.")
                                except Exception as e:
                                    if st.session_state.debug_mode:
                                        st.error(f"Error displaying chart: {e}")
                                    debug_log(f"Error displaying chart: {e}")
            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"An error occurred: {e}")
                debug_log(f"An error occurred: {e}")

    # 新增：多模型交叉驗證按鈕
    if st.button("多模型交叉驗證"):
        if openai_api_key:
            client = initialize_client(openai_api_key)
        else:
            st.error("OpenAI API Key is required for cross validation.")
            st.stop()
    
        # 設定兩個模型的參數（可根據需要調整）
        model_params_openai = {
            "model": "gpt-4o",
            "temperature": 0.5,
            "max_tokens": 4096
        }
        model_params_gemini = {
            "model": "models/gemini-2.0-flash",
            "temperature": 0.5,
            "max_tokens": 4096
        }
        with st.spinner("正在執行多模型交叉驗證..."):
            cross_validated_response = get_cross_validated_response(model_params_gemini)
            # 將交叉驗證結果添加到記憶流中，這樣整合報告可以使用這些結果
            simulate_system_message_addition(cross_validated_response, "交叉驗證報告")
            
            # 顯示交叉驗證結果
            with st.expander("🔍 交叉驗證結果", expanded=True):
                st.markdown("### Gemini 交叉驗證")
                st.markdown(cross_validated_response["gemini_response"])
        
        # 顯示整合報告（圖表和PDF下載按鈕由_render_integrated_report函數處理）
        with st.spinner("正在生成整合報告..."):
            st.markdown("## 📊 整合分析報告")
            generate_integrated_report(model_params_gemini)
                    
    if "ace_code" not in st.session_state:
        st.session_state.ace_code = ""

    if st.session_state.editor_location == "Main":
        with st.expander("🖋️ Persistent Code Editor (Main)", expanded=False):
            edited_code = st_ace(
                value=st.session_state.ace_code,
                language="python",
                theme="monokai",
                height=300,
                key="persistent_editor_main"
            )
            if edited_code != st.session_state.ace_code:
                st.session_state.ace_code = edited_code
                debug_log("ace_code updated from main editor.")
            if st.button("▶️ Execute Code", key="execute_code_main"):
                global_vars = {
                    "uploaded_file_path": st.session_state.uploaded_file_path,
                    "uploaded_image_path": st.session_state.uploaded_image_path,
                }
                debug_log(f"Executing code with uploaded_file_path = {st.session_state.uploaded_file_path}")
                debug_log(f"Executing code with uploaded_image_path = {st.session_state.uploaded_image_path}")
                result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                st.write("### Execution Result")
                st.text(result)
                debug_log(f"Code execution result: {result}")
            
    else:
        with st.sidebar.expander("🖋️ Persistent Code Editor (Sidebar)", expanded=False):
            edited_code = st_ace(
                value=st.session_state.ace_code,
                language="python",
                theme="monokai",
                height=300,
                key="persistent_editor_sidebar"
            )
            if edited_code != st.session_state.ace_code:
                st.session_state.ace_code = edited_code
                debug_log("ace_code updated from sidebar editor.")
            if st.button("▶️ Execute Code", key="execute_code_sidebar"):
                global_vars = {
                    "uploaded_file_path": st.session_state.uploaded_file_path,
                    "uploaded_image_path": st.session_state.uploaded_image_path,
                }
                debug_log(f"Executing code with uploaded_file_path = {st.session_state.uploaded_file_path}")
                debug_log(f"Executing code with uploaded_image_path = {st.session_state.uploaded_image_path}")
                result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                st.write("### Execution Result")
                st.text(result)
                debug_log(f"Code execution result: {result}")

if __name__ == "__main__":
    main()
