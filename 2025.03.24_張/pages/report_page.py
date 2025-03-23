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
from PIL import Image
import time
import tempfile
from reportlab.lib.pagesizes import letter
from io import BytesIO
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT

# 頁面設定
st.set_page_config(page_title="整合分析報告", page_icon="📊", layout="wide")
st.title("📊 整合分析報告")

# 檢查報告數據是否存在
if "integrated_report" not in st.session_state:
    st.warning("⚠️ 尚未生成報告，請回到主頁面先生成報告。")
    st.stop()

# 獲取報告數據
report_data = st.session_state.integrated_report

# ===================================================================
# 報告內容渲染（直接從_render_integrated_report複製）
# ===================================================================

# 1. 基本驗證與錯誤處理
if not isinstance(report_data, dict):
    st.error("❌ 無效的報告數據格式")
    st.stop()

# 2. 文字報告渲染
if "gemini_response" in report_data and report_data["gemini_response"]:
    try:
        st.markdown("## 📝 整合分析報告")
        st.markdown(report_data["gemini_response"])
    except Exception as e:
        st.error("報告內容渲染失敗")
        st.error(f"錯誤原因: {str(e)}")
else:
    st.warning("⚠️ 報告內容缺失，可能生成失敗或無有效分析結果")

# 3. 圖表渲染核心邏輯
if "charts_data" in report_data and report_data["charts_data"]:
    st.markdown("---")
    st.markdown("## 📊 相關圖表")
    
    # 初始化映射表檢查
    if "chart_mapping" not in st.session_state:
        st.error("圖表映射表丟失，請重新生成報告")
        st.stop()

    for chart in report_data["charts_data"]:
        # 3.1 數據有效性驗證
        if not isinstance(chart, dict) or "id" not in chart:
            st.warning(f"無效圖表數據格式")
            continue
            
        chart_id = chart["id"]
        
        # 3.2 從映射表獲取真實數據
        if chart_id not in st.session_state.chart_mapping:
            st.warning(f"圖表 {chart_id} 數據缺失，可能已被清理")
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

# 4. PDF下載按鈕
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
                key=f"dl_{int(time.time())}"  # 避免按鈕ID衝突
            )
            
    except Exception as e:
        st.error("PDF文件生成失敗，請重試或聯繫管理員")
        st.error(f"錯誤原因: {str(e)}")
        
    # 清理臨時文件
    try:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    except:
        pass
