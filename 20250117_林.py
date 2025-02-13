# 20250213_下拉選單_已完成

import streamlit as st

# 初始化 session state
if 'active_menu' not in st.session_state:
    st.session_state.active_menu = None

def toggle_menu(menu_name):
    if st.session_state.active_menu == menu_name:
        st.session_state.active_menu = None
    else:
        st.session_state.active_menu = menu_name

# CSS 樣式
st.markdown("""
<style>
    /* 全局樣式 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    .stApp {
        background: #000000;
        font-family: 'Inter', sans-serif;
        color: #fff;
        padding: 0 !important;
    }
    
    /* 頂部導航欄 */
    div[data-testid="stHorizontalBlock"] {
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        margin: 0;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
    }
    
    div[data-testid="stHorizontalBlock"] > div {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 0 !important;
    }
    
    /* 按鈕基本樣式 */
    .stButton > button {
        background: transparent !important;
        border: none !important;
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        padding: 8px 16px !important;
        margin: 0 !important;
        height: auto !important;
        width: auto !important;
        transition: all 0.2s ease !important;
        text-transform: none !important;
        line-height: 20px !important;
    }
    
    /* 按鈕懸停效果 */
    .stButton > button:hover {
        color: #fff !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* 子菜單樣式 */
    div[data-testid="stHorizontalBlock"] + div[data-testid="stHorizontalBlock"] {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        margin-top: 48px !important;
        padding: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* 內容區域樣式 */
    .content-wrapper {
        margin-top: 80px;
        padding: 20px;
    }
    
    /* 移除默認樣式 */
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    div[data-testid="stToolbar"] {
        display: none;
    }
    
    header[data-testid="stHeader"] {
        display: none;
    }
    
    footer {
        display: none;
    }
    
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* 修復間距問題 */
    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    
    /* 修復 columns 間距 */
    div[data-testid="column"] {
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# 主導航按鈕
cols = st.columns([1,1,1,1,1,1])

with cols[0]:
    st.button("數據分析", key="nav_data", on_click=toggle_menu, args=('data',))
with cols[1]:
    st.button("市場預測", key="nav_market", on_click=toggle_menu, args=('market',))
with cols[2]:
    st.button("風險評估", key="nav_risk", on_click=toggle_menu, args=('risk',))
with cols[3]:
    st.button("客戶分析", key="nav_customer", on_click=toggle_menu, args=('customer',))
with cols[4]:
    st.button("營運優化", key="nav_operation", on_click=toggle_menu, args=('operation',))
with cols[5]:
    st.button("諮詢服務", key="nav_consulting", on_click=toggle_menu, args=('consulting',))

# 子選單
if st.session_state.active_menu == 'data':
    cols = st.columns([1,1,1,1])
    with cols[0]:
        st.button("銷售分析", key="data_sales")
    with cols[1]:
        st.button("財務分析", key="data_finance")
    with cols[2]:
        st.button("庫存分析", key="data_inventory")
    with cols[3]:
        st.button("效能分析", key="data_performance")

elif st.session_state.active_menu == 'market':
    cols = st.columns([1,1,1])
    with cols[0]:
        st.button("市場趨勢", key="market_trend")
    with cols[1]:
        st.button("競爭分析", key="market_competition")
    with cols[2]:
        st.button("商機探勘", key="market_opportunity")

elif st.session_state.active_menu == 'risk':
    cols = st.columns([1,1,1])
    with cols[0]:
        st.button("風險評估", key="risk_assessment")
    with cols[1]:
        st.button("風險預警", key="risk_warning")
    with cols[2]:
        st.button("風險管理", key="risk_management")

# 主要內容區域
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

# Hero 區域
st.markdown("""
<div style="height: 100vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1352&q=80'); background-size: cover; background-position: center;">
    <h1 style="font-size: 48px; font-weight: 500; margin-bottom: 24px;">AI 企業智能助理</h1>
    <p style="font-size: 18px; color: rgba(255,255,255,0.8); max-width: 600px; line-height: 1.6;">為您的企業帶來智能化的數據分析與決策支持，提升營運效率，創造更大價值。</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

