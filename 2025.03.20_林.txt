import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm
import xlsxwriter
from io import BytesIO
import openai
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# 設置 OpenAI API key
api_key = ""
os.environ["OPENAI_API_KEY"] = api_key

# 郵件設置
EMAIL_SENDER = "skeswinnie@gmail.com"
EMAIL_PASSWORD = "dkyu hpmy tpai rjwf"

# 設置頁面配置
st.set_page_config(
    page_title="商店銷售分析系統",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 設置全局樣式
st.markdown("""
    <style>
    /* 主要內容區域背景 */
    .stApp {
        background: linear-gradient(-45deg, #E6F3FF, #E8FBFF, #FFF8E1);
        background-size: 400% 400%;
        animation: gradient 20s ease infinite;
        padding: 2rem;
    }
   
    @keyframes gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
   
    /* 側邊欄樣式 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(182, 251, 255, 0.2));
        border-right: 1px solid rgba(230, 230, 230, 0.5);
        box-shadow: 2px 0 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    section[data-testid="stSidebar"]:hover {
        box-shadow: 2px 0 20px rgba(135, 206, 235, 0.2);
    }

    /* 側邊欄內部元素 */
    section[data-testid="stSidebar"] .stMarkdown {
        background-color: transparent;
        transition: all 0.3s ease;
    }
   
    /* 標題樣式 */
    h1, h2, h3, h4, h5, h6 {
        color: #2E4053;
        font-weight: 700;
        padding: 0.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.9), rgba(182, 251, 255, 0.3));
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }

    h1:hover, h2:hover, h3:hover, h4:hover, h5:hover, h6:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(135, 206, 235, 0.3);
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.95), rgba(131, 164, 212, 0.2));
    }

    /* 按鈕樣式 */
    .stButton > button {
        background: linear-gradient(120deg, #87CEEB, #ADD8E6);
        color: #2E4053;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 4px 15px rgba(135, 206, 235, 0.4);
        background: linear-gradient(120deg, #ADD8E6, #87CEEB);
    }

    .stButton > button:active {
        transform: translateY(1px);
    }

    /* 添加彩虹邊框效果 */
    @keyframes borderGlow {
        0% {
            border-color: #87CEEB;
        }
        25% {
            border-color: #ADD8E6;
        }
        50% {
            border-color: #B6FBFF;
        }
        75% {
            border-color: #FFD700;
        }
        100% {
            border-color: #87CEEB;
        }
    }

    /* 其他樣式保持不變 */
    /* 數據框樣式 */
    .dataframe {
        background: rgba(255, 255, 255, 0.9);
        border: none !important;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .dataframe:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(255, 154, 139, 0.3);
    }

    .dataframe th {
        background-color: #faf9f6;
        color: #333333;
        font-weight: 600;
        padding: 12px !important;
        transition: all 0.3s ease;
    }

    .dataframe td {
        color: #333333;
        border: none !important;
        border-bottom: 1px solid rgba(182, 251, 255, 0.3) !important;
        padding: 12px !important;
        transition: all 0.2s ease;
    }

    .dataframe tr:hover {
        background-color: rgba(131, 164, 212, 0.1);
        transform: scale(1.01);
    }

    /* 圖表容器 */
    .stPlot {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(131, 164, 212, 0.1));
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 1rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .stPlot:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 5px 15px rgba(255, 154, 139, 0.3);
    }

    /* 選擇器樣式 */
    .stSelectbox {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 154, 139, 0.1));
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .stSelectbox:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 106, 136, 0.2);
    }

    /* 指標卡片樣式 */
    div[data-testid="stMetricValue"] {
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.9), rgba(182, 251, 255, 0.2));
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }

    div[data-testid="stMetricValue"]:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 4px 15px rgba(131, 164, 212, 0.3);
    }

    /* Tab 樣式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(131, 164, 212, 0.1));
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab-list"]:hover {
        box-shadow: 0 4px 15px rgba(255, 106, 136, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.9), rgba(182, 251, 255, 0.2));
        border-radius: 6px;
        padding: 0.5rem 1rem;
        color: #2E4053;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-1px);
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.95), rgba(131, 164, 212, 0.2));
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background: linear-gradient(120deg, #87CEEB, #ADD8E6);
        border-radius: 6px;
        transition: all 0.3s ease;
    }

    /* 添加全局動畫效果 */
    * {
        transition: background-color 0.3s ease;
    }

    /* 添加載入動畫 */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .element-container {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

def create_figure_layout():
    return {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'xaxis_title': "",
        'yaxis_title': "",
        'showlegend': False
    }

# 新增表格和文字樣式
st.markdown("""
    <style>
    /* 標題文字樣式 */
    h1, h2, h3, h4, h5, h6 {
        color: #333333;
        font-weight: 600;
    }

    /* 一般文字樣式 */
    p, span, div {
        color: #333333;
    }

    /* 表格樣式 */
    .dataframe {
        color: #333333;
        background-color: #ffffff;
    }

    .dataframe th {
        background-color: #faf9f6;
        color: #333333;
        font-weight: 600;
        padding: 8px;
    }

    .dataframe td {
        color: #333333;
        padding: 8px;
    }

    /* 表格hover效果 */
    .dataframe tr:hover {
        background-color: #faf9f6;
    }

    /* 數據標籤樣式 */
    .metric-label {
        color: #333333;
        font-weight: 500;
    }

    /* 圖表標題和軸標籤 */
    .plot-container text {
        color: #333333 !important;
        fill: #333333 !important;
    }

    /* Streamlit特定元素樣式 */
    .stMarkdown, .stText {
        color: #333333;
    }

    .st-bb {
        color: #333333;
    }

    .st-bw {
        color: #333333;
    }

    /* 連結顏色 */
    a {
        color: #666666;
    }

    a:hover {
        color: #333333;
    }

    /* 強調文字 */
    .highlight {
        color: #666666;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

def generate_profit_loss_statement(df):
    """生成並分析損益表"""
    st.header("💰 損益分析")

    # 1. 基本收入和成本計算
    total_sales = df['Item_Outlet_Sales'].sum()
    avg_mrp = df['Item_MRP'].mean()
    total_items = len(df)
   
    # 成本估算
    cogs_ratio = 0.7  # 假設銷貨成本為銷售額的70%
    cogs = total_sales * cogs_ratio
   
    # 費用估算
    operating_expenses = {
        '人事費用': total_sales * 0.15,
        '租金': total_sales * 0.08,
        '水電費': total_sales * 0.03,
        '行銷費用': total_sales * 0.05,
        '其他費用': total_sales * 0.04
    }
    total_operating_expenses = sum(operating_expenses.values())
   
    # 2. 損益表主體
    st.subheader("📊 損益表")
   
    # 計算毛利和淨利
    gross_profit = total_sales - cogs
    operating_profit = gross_profit - total_operating_expenses
    net_profit = operating_profit * 0.8  # 假設稅率20%
   
    # 顯示損益表
    pl_data = {
        '項目': [
            '營業收入',
            '銷貨成本',
            '毛利',
            '營業費用',
            '營業利益',
            '稅前淨利',
            '所得稅費用',
            '稅後淨利'
        ],
        '金額': [
            total_sales,
            cogs,
            gross_profit,
            total_operating_expenses,
            operating_profit,
            operating_profit,
            operating_profit * 0.2,
            net_profit
        ],
        '佔收入比': [
            100.0,
            (cogs / total_sales) * 100,
            (gross_profit / total_sales) * 100,
            (total_operating_expenses / total_sales) * 100,
            (operating_profit / total_sales) * 100,
            (operating_profit / total_sales) * 100,
            (operating_profit * 0.2 / total_sales) * 100,
            (net_profit / total_sales) * 100
        ]
    }
   
    pl_df = pd.DataFrame(pl_data)
    st.dataframe(pl_df.style.format({
        '金額': '${:,.2f}',
        '佔收入比': '{:.1f}%'
    }))

    # 3. 營業費用明細
    st.subheader("💸 營業費用分析")
   
    # 顯示營業費用明細
    expense_data = {
        '費用項目': list(operating_expenses.keys()),
        '金額': list(operating_expenses.values()),
        '佔營收比': [(v / total_sales) * 100 for v in operating_expenses.values()]
    }
   
    expense_df = pd.DataFrame(expense_data)
   
    col1, col2 = st.columns([2, 1])
   
    with col1:
        # 營業費用圓餅圖
        fig = px.pie(
            expense_df,
            values='金額',
            names='費用項目',
            title='營業費用結構'
        )
        st.plotly_chart(fig, use_container_width=True)
   
    with col2:
        st.dataframe(expense_df.style.format({
            '金額': '${:,.2f}',
            '佔營收比': '{:.1f}%'
        }))

    # 4. 利潤分析
    st.subheader("📈 利潤分析")
   
    col1, col2 = st.columns(2)
   
    with col1:
        # 按商品類型的毛利分析
        product_profit = df.groupby('Item_Type').agg({
            'Item_Outlet_Sales': 'sum',
            'Item_MRP': lambda x: (x * 0.7).sum()  # 估算成本
        }).round(2)
       
        product_profit['毛利'] = product_profit['Item_Outlet_Sales'] - product_profit['Item_MRP']
        product_profit['毛利率'] = (product_profit['毛利'] / product_profit['Item_Outlet_Sales']) * 100
       
        fig = px.bar(
            product_profit.sort_values('毛利率', ascending=True),
            y=product_profit.index,
            x='毛利率',
            title='各商品類別毛利率',
            labels={'y': '商品類別', 'x': '毛利率 (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
   
    with col2:
        # 按商店類型的毛利分析
        store_profit = df.groupby('Outlet_Type').agg({
            'Item_Outlet_Sales': 'sum',
            'Item_MRP': lambda x: (x * 0.7).sum()  # 估算成本
        }).round(2)
       
        store_profit['毛利'] = store_profit['Item_Outlet_Sales'] - store_profit['Item_MRP']
        store_profit['毛利率'] = (store_profit['毛利'] / store_profit['Item_Outlet_Sales']) * 100
       
        fig = px.bar(
            store_profit.sort_values('毛利率', ascending=True),
            y=store_profit.index,
            x='毛利率',
            title='各商店類型毛利率',
            labels={'y': '商店類型', 'x': '毛利率 (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # 5. 損益關鍵指標
    st.subheader("🎯 損益關鍵指標")
   
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        gross_margin = (gross_profit / total_sales) * 100
        st.metric(
            "毛利率",
            f"{gross_margin:.1f}%",
            delta="良好" if gross_margin > 30 else ("尚可" if gross_margin > 20 else "注意")
        )
   
    with col2:
        operating_margin = (operating_profit / total_sales) * 100
        st.metric(
            "營業利益率",
            f"{operating_margin:.1f}%",
            delta="良好" if operating_margin > 15 else ("尚可" if operating_margin > 10 else "注意")
        )
   
    with col3:
        net_margin = (net_profit / total_sales) * 100
        st.metric(
            "淨利率",
            f"{net_margin:.1f}%",
            delta="良好" if net_margin > 10 else ("尚可" if net_margin > 5 else "注意")
        )
   
    with col4:
        expense_ratio = (total_operating_expenses / total_sales) * 100
        st.metric(
            "費用率",
            f"{expense_ratio:.1f}%",
            delta="良好" if expense_ratio < 30 else ("尚可" if expense_ratio < 40 else "注意")
        )

    # 6. 損益改善建議
    st.subheader("💡 損益改善建議")
   
    recommendations = []
   
    # 根據毛利率提供建議
    if gross_margin < 30:
        recommendations.append("• 毛利率偏低，建議：\n  - 檢討定價策略\n  - 優化採購成本\n  - 調整商品組合")
   
    # 根據費用率提供建議
    if expense_ratio > 35:
        recommendations.append("• 費用率偏高，建議：\n  - 檢討人事配置效率\n  - 優化租金支出\n  - 加強費用控管")
   
    # 根據商品毛利分析提供建議
    low_margin_products = product_profit[product_profit['毛利率'] < 20].index.tolist()
    if low_margin_products:
        recommendations.append(f"• 以下商品類別毛利率偏低，建議調整策略：\n  - {', '.join(low_margin_products)}")
   
    # 根據商店毛利分析提供建議
    low_margin_stores = store_profit[store_profit['毛利率'] < 25].index.tolist()
    if low_margin_stores:
        recommendations.append(f"• 以下商店類型毛利率偏低，建議強化營運：\n  - {', '.join(low_margin_stores)}")
   
    # 顯示建議
    if recommendations:
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.markdown("""
        整體損益表現良好，建議：
        • 持續監控成本結構，維持良好獲利能力
        • 適度投資於成長機會，擴大營收規模
        • 定期檢討費用效益，確保資源最佳配置
        """)

    # 7. 同業比較分析
    st.subheader("🏢 同業比較分析")
   
    # 模擬同業數據
    industry_data = {
        '指標': ['毛利率', '營業利益率', '淨利率', '費用率'],
        '本公司': [gross_margin, operating_margin, net_margin, expense_ratio],
        '同業平均': [32, 15, 10, 35],
        '同業最佳': [40, 20, 15, 30]
    }
   
    industry_df = pd.DataFrame(industry_data)
   
    # 繪製雷達圖
    fig = go.Figure()
   
    fig.add_trace(go.Scatterpolar(
        r=industry_df['本公司'],
        theta=industry_df['指標'],
        fill='toself',
        name='本公司'
    ))
   
    fig.add_trace(go.Scatterpolar(
        r=industry_df['同業平均'],
        theta=industry_df['指標'],
        fill='toself',
        name='同業平均'
    ))
   
    fig.add_trace(go.Scatterpolar(
        r=industry_df['同業最佳'],
        theta=industry_df['指標'],
        fill='toself',
        name='同業最佳'
    ))
   
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 50]
            )),
        showlegend=True,
        title='關鍵指標同業比較'
    )
   
    st.plotly_chart(fig, use_container_width=True)

def generate_customer_revenue_report(df):
    """生成並分析客戶收入報表"""
    st.header("👥 客戶收入分析報表")
   
    # 1. 基本收入統計
    st.subheader("📊 基本收入統計")
   
    # 計算每個商店的客戶收入
    store_revenue = df.groupby('Outlet_Identifier').agg({
        'Item_Outlet_Sales': ['sum', 'mean', 'count', 'std'],
        'Item_MRP': 'mean'
    }).round(2)
   
    store_revenue.columns = ['總收入', '平均交易金額', '交易次數', '收入標準差', '平均商品價格']
    store_revenue = store_revenue.sort_values('總收入', ascending=False)
   
    # 添加其他計算指標
    store_revenue['客單價'] = store_revenue['總收入'] / store_revenue['交易次數']
    store_revenue['價格效率'] = store_revenue['平均交易金額'] / store_revenue['平均商品價格']
   
    # 顯示收入統計表
    st.write("各店鋪收入統計：")
    st.dataframe(store_revenue.style.format({
        '總收入': '${:,.2f}',
        '平均交易金額': '${:,.2f}',
        '交易次數': '{:,.0f}',
        '收入標準差': '${:,.2f}',
        '平均商品價格': '${:,.2f}',
        '客單價': '${:,.2f}',
        '價格效率': '{:,.2f}'
    }))
   
    # 2. 收入分布分析
    st.subheader("📈 收入分布分析")
   
    col1, col2 = st.columns(2)
   
    with col1:
        # 繪製收入分布圖
        fig = px.box(df,
                    y='Item_Outlet_Sales',
                    x='Outlet_Type',
                    title='各類型商店收入分布')
        st.plotly_chart(fig, use_container_width=True)
   
    with col2:
        # 繪製收入趨勢圖（按商店類型）
        store_type_revenue = df.groupby('Outlet_Type')['Item_Outlet_Sales'].agg(['sum', 'mean', 'count'])
        fig = px.bar(store_type_revenue,
                    title='各類型商店收入比較',
                    labels={'value': '金額', 'variable': '指標'},
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
   
    # 3. 商品類型收入分析
    st.subheader("🏷️ 商品類型收入分析")
   
    # 計算每種商品類型的收入
    item_type_revenue = df.groupby('Item_Type').agg({
        'Item_Outlet_Sales': ['sum', 'mean', 'count'],
        'Item_MRP': 'mean'
    }).round(2)
   
    item_type_revenue.columns = ['總收入', '平均收入', '銷售數量', '平均價格']
    item_type_revenue['毛利率'] = ((item_type_revenue['平均收入'] - item_type_revenue['平均價格'])
                                / item_type_revenue['平均價格'] * 100)
   
    # 顯示商品類型收入分析
    st.write("商品類型收入分析：")
    st.dataframe(item_type_revenue.style.format({
        '總收入': '${:,.2f}',
        '平均收入': '${:,.2f}',
        '銷售數量': '{:,.0f}',
        '平均價格': '${:,.2f}',
        '毛利率': '{:,.1f}%'
    }))
   
    # 4. 收入區間分析
    st.subheader("💰 收入區間分析")
   
    # 創建收入區間
    df['收入區間'] = pd.qcut(df['Item_Outlet_Sales'],
                        q=5,
                        labels=['低收入', '中低收入', '中等收入', '中高收入', '高收入'])
   
    # 計算每個收入區間的統計數據
    revenue_range_stats = df.groupby('收入區間').agg({
        'Item_Outlet_Sales': ['count', 'sum', 'mean'],
        'Item_MRP': 'mean'
    }).round(2)
   
    revenue_range_stats.columns = ['交易次數', '總收入', '平均收入', '平均商品價格']
   
    # 顯示收入區間統計
    st.write("收入區間分析：")
    st.dataframe(revenue_range_stats.style.format({
        '交易次數': '{:,.0f}',
        '總收入': '${:,.2f}',
        '平均收入': '${:,.2f}',
        '平均商品價格': '${:,.2f}'
    }))
   
    # 5. 位置與規模分析
    st.subheader("📍 位置與規模分析")
   
    col1, col2 = st.columns(2)
   
    with col1:
        # 計算不同位置類型的收入
        location_revenue = df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].agg(['sum', 'mean', 'count'])
        fig = px.bar(location_revenue,
                    title='各地區收入比較',
                    labels={'value': '金額', 'variable': '指標'})
        st.plotly_chart(fig, use_container_width=True)
   
    with col2:
        # 計算不同規模的收入
        size_revenue = df.groupby('Outlet_Size')['Item_Outlet_Sales'].agg(['sum', 'mean', 'count'])
        fig = px.bar(size_revenue,
                    title='各規模收入比較',
                    labels={'value': '金額', 'variable': '指標'})
        st.plotly_chart(fig, use_container_width=True)
   
    # 6. 關鍵績效指標 (KPI)
    st.subheader("📈 關鍵績效指標")
   
    col1, col2, col3, col4 = st.columns(4)
   
    # 計算KPI
    total_revenue = df['Item_Outlet_Sales'].sum()
    avg_transaction = df['Item_Outlet_Sales'].mean()
    total_transactions = len(df)
    revenue_growth = (df.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().pct_change().mean() * 100)
   
    with col1:
        st.metric("總收入", f"${total_revenue:,.2f}")
    with col2:
        st.metric("平均交易金額", f"${avg_transaction:,.2f}")
    with col3:
        st.metric("總交易次數", f"{total_transactions:,}")
    with col4:
        st.metric("收入成長率", f"{revenue_growth:,.1f}%")
   
    # 7. 客戶分群分析
    st.subheader("👥 客戶分群分析")
   
    # 使用K-means進行客戶分群
    X = df[['Item_Outlet_Sales', 'Item_MRP']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Customer_Segment'] = kmeans.fit_predict(X_scaled)
   
    # 計算每個分群的統計數據
    segment_stats = df.groupby('Customer_Segment').agg({
        'Item_Outlet_Sales': ['mean', 'count', 'sum'],
        'Item_MRP': 'mean'
    }).round(2)
   
    segment_stats.columns = ['平均消費', '客戶數量', '總收入', '平均商品價格']
   
    # 根據平均消費重新命名分群
    segment_mapping = {
        segment_stats['平均消費'].idxmin(): '經濟型客戶',
        segment_stats['平均消費'].idxmax(): '高價值客戶',
        segment_stats['平均消費'].iloc[1]: '中間型客戶'
    }
   
    segment_stats.index = segment_stats.index.map(segment_mapping)
   
    # 顯示分群結果
    st.write("客戶分群分析：")
    st.dataframe(segment_stats.style.format({
        '平均消費': '${:,.2f}',
        '客戶數量': '{:,.0f}',
        '總收入': '${:,.2f}',
        '平均商品價格': '${:,.2f}'
    }))
   
    # 8. 分析建議
    st.subheader("💡 分析建議")
   
    recommendations = []
   
    # 根據收入分布提供建議
    top_store = store_revenue.index[0]
    bottom_store = store_revenue.index[-1]
    revenue_gap = store_revenue.loc[top_store, '總收入'] / store_revenue.loc[bottom_store, '總收入']
   
    if revenue_gap > 2:
        recommendations.append(f"• 店鋪間收入差距較大（最高/最低 = {revenue_gap:.1f}倍），建議分析成功店鋪經驗並推廣")
   
    # 根據客單價提供建議
    low_transaction_stores = store_revenue[store_revenue['客單價'] < store_revenue['客單價'].mean()]
    if not low_transaction_stores.empty:
        recommendations.append(f"• 有{len(low_transaction_stores)}家店鋪的客單價低於平均值，建議加強商品組合和銷售策略")
   
    # 根據商品類型收入提供建議
    top_product_type = item_type_revenue['總收入'].idxmax()
    top_margin_type = item_type_revenue['毛利率'].idxmax()
   
    recommendations.append(f"• {top_product_type}類商品銷售額最高，建議確保庫存充足")
    recommendations.append(f"• {top_margin_type}類商品毛利率最高，建議適當增加促銷力度")
   
    # 根據位置分析提供建議
    best_location = df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].mean().idxmax()
    recommendations.append(f"• {best_location}地區的平均收入表現最好，建議在類似地區尋找展店機會")
   
    # 根據客戶分群提供建議
    high_value_ratio = (segment_stats.loc['高價值客戶', '客戶數量'] /
                       segment_stats['客戶數量'].sum() * 100)
   
    if high_value_ratio < 20:
        recommendations.append("• 高價值客戶佔比較低，建議制定會員優惠計劃提升客戶忠誠度")
   
    for rec in recommendations:
        st.markdown(rec)

def generate_balance_sheet(df):
    """生成並分析資產負債表"""
    st.header("💰 資產負債表分析")

    # 1. 計算資產項目
    ## 流動資產
    inventory_value = (df['Item_Weight'] * df['Item_MRP']).sum()  # 庫存價值
    accounts_receivable = df['Item_Outlet_Sales'].sum() * 0.1  # 應收帳款（假設銷售額的10%）
    cash_equivalent = df['Item_Outlet_Sales'].sum() * 0.15  # 現金及約當現金（假設銷售額的15%）
   
    current_assets = inventory_value + accounts_receivable + cash_equivalent
   
    ## 非流動資產
    fixed_assets = df.groupby('Outlet_Identifier').size().shape[0] * 1000000  # 每家店估值100萬
    equipment_value = df.groupby('Outlet_Type')['Outlet_Identifier'].nunique().sum() * 100000   # 每家店設備10萬
   
    total_assets = current_assets + fixed_assets + equipment_value

    # 2. 計算負債項目
    ## 流動負債
    accounts_payable = inventory_value * 0.4  # 應付帳款（假設庫存40%未付款）
    short_term_debt = current_assets * 0.2  # 短期借款（假設流動資產20%）
   
    current_liabilities = accounts_payable + short_term_debt
   
    ## 非流動負債
    long_term_debt = fixed_assets * 0.5  # 長期借款（假設固定資產50%）
   
    total_liabilities = current_liabilities + long_term_debt
   
    # 3. 計算權益
    total_equity = total_assets - total_liabilities

    # 4. 顯示資產負債表
    st.subheader("📊 資產負債表")
   
    col1, col2 = st.columns(2)
   
    with col1:
        st.markdown("### 資產")
        st.markdown("#### 流動資產")
        st.write(f"現金及約當現金：${cash_equivalent:,.2f}")
        st.write(f"應收帳款：${accounts_receivable:,.2f}")
        st.write(f"存貨：${inventory_value:,.2f}")
        st.write(f"**流動資產合計：${current_assets:,.2f}**")
       
        st.markdown("#### 非流動資產")
        st.write(f"固定資產：${fixed_assets:,.2f}")
        st.write(f"設備：${equipment_value:,.2f}")
        st.write(f"**非流動資產合計：${fixed_assets + equipment_value:,.2f}**")
       
        st.markdown(f"### **資產總計：${total_assets:,.2f}**")
   
    with col2:
        st.markdown("### 負債")
        st.markdown("#### 流動負債")
        st.write(f"應付帳款：${accounts_payable:,.2f}")
        st.write(f"短期借款：${short_term_debt:,.2f}")
        st.write(f"**流動負債合計：${current_liabilities:,.2f}**")
       
        st.markdown("#### 非流動負債")
        st.write(f"長期借款：${long_term_debt:,.2f}")
        st.write(f"**非流動負債合計：${long_term_debt:,.2f}**")
       
        st.markdown(f"**負債總計：${total_liabilities:,.2f}**")
       
        st.markdown("### 權益")
        st.write(f"**權益總計：${total_equity:,.2f}**")
   
    # 5. 財務分析
    st.subheader("📈 財務分析")
   
    # 計算關鍵財務比率
    current_ratio = current_assets / current_liabilities
    debt_ratio = total_liabilities / total_assets
    equity_ratio = total_equity / total_assets
    asset_turnover = df['Item_Outlet_Sales'].sum() / total_assets
   
    # 顯示財務比率
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        st.metric("流動比率", f"{current_ratio:.2f}")
        st.caption("流動資產/流動負債")
        if current_ratio >= 2:
            st.success("流動性良好")
        elif current_ratio >= 1:
            st.warning("流動性尚可")
        else:
            st.error("流動性不足")
   
    with col2:
        st.metric("負債比率", f"{debt_ratio:.2%}")
        st.caption("總負債/總資產")
        if debt_ratio <= 0.4:
            st.success("負債水平健康")
        elif debt_ratio <= 0.6:
            st.warning("負債水平適中")
        else:
            st.error("負債水平過高")
   
    with col3:
        st.metric("權益比率", f"{equity_ratio:.2%}")
        st.caption("總權益/總資產")
        if equity_ratio >= 0.6:
            st.success("自有資金充足")
        elif equity_ratio >= 0.4:
            st.warning("自有資金適中")
        else:
            st.error("自有資金不足")
   
    with col4:
        st.metric("資產週轉率", f"{asset_turnover:.2f}")
        st.caption("銷售額/總資產")
        if asset_turnover >= 2:
            st.success("資產運用效率高")
        elif asset_turnover >= 1:
            st.warning("資產運用效率中等")
        else:
            st.error("資產運用效率低")

    # 6. 店鋪資產分析
    st.subheader("🏪 店鋪資產分析")
   
    # 計算每家店的資產情況
    store_stats = df.groupby('Outlet_Identifier').agg({
        'Item_Outlet_Sales': 'sum',
        'Item_Weight': lambda x: (x * df.loc[x.index, 'Item_MRP']).sum()  # 庫存價值
    }).round(2)
   
    store_stats.columns = ['銷售額', '庫存價值']
    store_stats['固定資產'] = 1000000  # 每家店固定資產
    store_stats['設備價值'] = 100000   # 每家店設備
    store_stats['總資產'] = store_stats['庫存價值'] + store_stats['固定資產'] + store_stats['設備價值']
    store_stats['資產報酬率'] = store_stats['銷售額'] / store_stats['總資產']
   
    # 顯示店鋪資產分析
    st.dataframe(store_stats.style.format({
        '銷售額': '${:,.2f}',
        '庫存價值': '${:,.2f}',
        '固定資產': '${:,.2f}',
        '設備價值': '${:,.2f}',
        '總資產': '${:,.2f}',
        '資產報酬率': '{:.2%}'
    }))

    # 7. 資產結構分析
    st.subheader("📊 資產結構分析")
   
    # 準備資產結構數據
    asset_structure = pd.DataFrame({
        '資產項目': ['現金及約當現金', '應收帳款', '存貨', '固定資產', '設備'],
        '金額': [cash_equivalent, accounts_receivable, inventory_value, fixed_assets, equipment_value]
    })
   
    # 計算佔比
    asset_structure['佔比'] = asset_structure['金額'] / total_assets
   
    # 繪製資產結構圖
    fig = px.pie(asset_structure,
                 values='金額',
                 names='資產項目',
                 title='資產結構分布')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 8. 分析建議
    st.subheader("💡 分析建議")
   
    recommendations = []
   
    # 根據財務比率提供建議
    if current_ratio < 1.5:
        recommendations.append("• 建議增加流動資產或減少流動負債以提升流動性")
    if debt_ratio > 0.5:
        recommendations.append("• 負債比率較高，建議控制舉債規模")
    if asset_turnover < 1:
        recommendations.append("• 資產運用效率偏低，建議優化庫存管理")
   
    # 根據店鋪表現提供建議
    low_performing_stores = store_stats[store_stats['資產報酬率'] < store_stats['資產報酬率'].mean()]
    if not low_performing_stores.empty:
        recommendations.append(f"• 有{len(low_performing_stores)}家店鋪的資產報酬率低於平均，建議進行營運改善")
   
    # 根據資產結構提供建議
    if (inventory_value / current_assets) > 0.6:
        recommendations.append("• 存貨佔比過高，建議加強庫存管理")
    if (cash_equivalent / current_assets) < 0.1:
        recommendations.append("• 現金比率偏低，建議增加營運資金")
   
    for rec in recommendations:
        st.markdown(rec)

def generate_financial_ratios(df):
    """財務比率分析"""
    st.header("📈 財務比率分析")

    # 1. 基本財務數據計算
    total_sales = df['Item_Outlet_Sales'].sum()
    total_cost = (df['Item_MRP'] * 0.7).sum()  # 假設成本為MRP的70%
    inventory_value = df['Item_MRP'].sum() * 0.3  # 假設庫存為MRP的30%
    accounts_receivable = total_sales * 0.15  # 假設15%的銷售額為應收帳款
    accounts_payable = total_cost * 0.2  # 假設20%的成本為應付帳款
    fixed_assets = total_sales * 0.4  # 假設固定資產為銷售額的40%
    current_assets = inventory_value + accounts_receivable + (total_sales * 0.1)  # 加上現金
    current_liabilities = accounts_payable + (total_cost * 0.1)  # 加上短期負債
    total_assets = current_assets + fixed_assets
    total_liabilities = current_liabilities + (fixed_assets * 0.5)  # 加上長期負債
    total_equity = total_assets - total_liabilities
    operating_income = total_sales - total_cost
    net_income = operating_income * 0.8  # 假設稅後淨利為營業利潤的80%

    # 2. 流動性比率分析
    st.subheader("💧 流動性比率")
    col1, col2, col3 = st.columns(3)
   
    with col1:
        current_ratio = current_assets / current_liabilities
        st.metric("流動比率", f"{current_ratio:.2f}",
                 delta="良好" if current_ratio > 2 else ("尚可" if current_ratio > 1 else "注意"))
   
    with col2:
        quick_ratio = (current_assets - inventory_value) / current_liabilities
        st.metric("速動比率", f"{quick_ratio:.2f}",
                 delta="良好" if quick_ratio > 1 else ("尚可" if quick_ratio > 0.5 else "注意"))
   
    with col3:
        cash_ratio = (total_sales * 0.1) / current_liabilities
        st.metric("現金比率", f"{cash_ratio:.2f}",
                 delta="良好" if cash_ratio > 0.5 else ("尚可" if cash_ratio > 0.2 else "注意"))

    # 3. 營運效率比率
    st.subheader("⚡ 營運效率比率")
    col1, col2, col3 = st.columns(3)
   
    with col1:
        inventory_turnover = total_cost / inventory_value
        st.metric("存貨週轉率", f"{inventory_turnover:.2f}次/年",
                 delta="良好" if inventory_turnover > 6 else ("尚可" if inventory_turnover > 4 else "注意"))
   
    with col2:
        receivable_turnover = total_sales / accounts_receivable
        st.metric("應收帳款週轉率", f"{receivable_turnover:.2f}次/年",
                 delta="良好" if receivable_turnover > 12 else ("尚可" if receivable_turnover > 8 else "注意"))
   
    with col3:
        asset_turnover = total_sales / total_assets
        st.metric("總資產週轉率", f"{asset_turnover:.2f}次/年",
                 delta="良好" if asset_turnover > 2 else ("尚可" if asset_turnover > 1 else "注意"))

    # 4. 獲利能力比率
    st.subheader("💰 獲利能力比率")
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        gross_margin = ((total_sales - total_cost) / total_sales) * 100
        st.metric("毛利率", f"{gross_margin:.1f}%",
                 delta="良好" if gross_margin > 30 else ("尚可" if gross_margin > 20 else "注意"))
   
    with col2:
        operating_margin = (operating_income / total_sales) * 100
        st.metric("營業利益率", f"{operating_margin:.1f}%",
                 delta="良好" if operating_margin > 15 else ("尚可" if operating_margin > 10 else "注意"))
   
    with col3:
        net_margin = (net_income / total_sales) * 100
        st.metric("淨利率", f"{net_margin:.1f}%",
                 delta="良好" if net_margin > 10 else ("尚可" if net_margin > 5 else "注意"))
   
    with col4:
        roe = (net_income / total_equity) * 100
        st.metric("股東權益報酬率", f"{roe:.1f}%",
                 delta="良好" if roe > 15 else ("尚可" if roe > 10 else "注意"))

    # 5. 財務結構比率
    st.subheader("🏗️ 財務結構比率")
    col1, col2, col3 = st.columns(3)
   
    with col1:
        debt_ratio = (total_liabilities / total_assets) * 100
        st.metric("負債比率", f"{debt_ratio:.1f}%",
                 delta="良好" if debt_ratio < 40 else ("尚可" if debt_ratio < 60 else "注意"))
   
    with col2:
        equity_ratio = (total_equity / total_assets) * 100
        st.metric("權益比率", f"{equity_ratio:.1f}%",
                 delta="良好" if equity_ratio > 60 else ("尚可" if equity_ratio > 40 else "注意"))
   
    with col3:
        debt_equity_ratio = (total_liabilities / total_equity)
        st.metric("負債對權益比率", f"{debt_equity_ratio:.2f}",
                 delta="良好" if debt_equity_ratio < 1 else ("尚可" if debt_equity_ratio < 2 else "注意"))

    # 6. 成長與趨勢分析
    st.subheader("📈 成長與趨勢分析")
   
    # 計算各商店的銷售成長率
    store_sales = df.groupby('Outlet_Identifier')['Item_Outlet_Sales'].sum()
    store_growth = store_sales.pct_change() * 100
   
    # 繪製成長趨勢圖
    fig = px.line(x=store_sales.index,
                  y=store_sales.values,
                  title='銷售額趨勢',
                  labels={'x': '商店', 'y': '銷售額'})
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
   
    # 7. 財務健康評估
    st.subheader("🏥 財務健康評估")
   
    # 計算綜合評分
    score = 0
    score += 20 if current_ratio > 2 else (10 if current_ratio > 1 else 0)
    score += 20 if debt_ratio < 0.4 else (10 if debt_ratio < 0.6 else 0)
    score += 20 if gross_margin > 30 else (10 if gross_margin > 20 else 0)
    score += 20 if asset_turnover > 2 else (10 if asset_turnover > 1 else 0)
    score += 20 if roe > 15 else (10 if roe > 10 else 0)
   
    # 顯示評分和建議
    col1, col2 = st.columns([1, 2])
   
    with col1:
        st.metric("財務健康評分", f"{score}/100",
                 delta="優異" if score >= 80 else ("良好" if score >= 60 else "需要改善"))
   
    with col2:
        st.markdown("### 改善建議")
        recommendations = []
       
        if current_ratio < 2:
            recommendations.append("• 提高流動性：考慮增加營運資金或減少短期負債")
        if debt_ratio > 0.5:
            recommendations.append("• 降低負債：考慮償還部分負債或增加自有資金")
        if gross_margin < 30:
            recommendations.append("• 提升毛利：檢討定價策略和成本控制")
        if asset_turnover < 2:
            recommendations.append("• 提高資產使用效率：檢討庫存管理和固定資產使用情況")
        if roe < 15:
            recommendations.append("• 改善獲利能力：增加營收或控制成本")
       
        for rec in recommendations:
            st.markdown(rec)

def generate_operational_metrics(df):
    """營運指標分析"""
    st.header("⚙️ 營運指標分析")

    # 1. 商品效率分析
    st.subheader("📦 商品效率分析")
   
    # 計算商品相關指標
    product_metrics = df.groupby('Item_Type').agg({
        'Item_Outlet_Sales': ['sum', 'mean', 'count'],
        'Item_MRP': 'mean',
        'Item_Weight': 'mean'
    }).round(2)
   
    product_metrics.columns = ['總銷售額', '平均銷售額', '銷售次數', '平均單價', '平均重量']
    product_metrics['銷售額佔比'] = (product_metrics['總銷售額'] / product_metrics['總銷售額'].sum()) * 100
    product_metrics['單位利潤'] = product_metrics['平均銷售額'] - product_metrics['平均單價']
   
    # 顯示商品效率分析
    st.dataframe(product_metrics.style.format({
        '總銷售額': '${:,.2f}',
        '平均銷售額': '${:,.2f}',
        '銷售次數': '{:,.0f}',
        '平均單價': '${:,.2f}',
        '平均重量': '{:,.2f}kg',
        '銷售額佔比': '{:,.1f}%',
        '單位利潤': '${:,.2f}'
    }))

    # 2. 商店效率分析
    st.subheader("🏪 商店效率分析")
   
    # 計算商店相關指標
    store_metrics = df.groupby(['Outlet_Identifier', 'Outlet_Type', 'Outlet_Size']).agg({
        'Item_Outlet_Sales': ['sum', 'mean', 'count'],
        'Item_MRP': 'sum'
    }).round(2)
   
    store_metrics.columns = ['總銷售額', '平均交易額', '交易次數', '商品總價值']
    store_metrics = store_metrics.reset_index()
   
    # 計算額外的效率指標
    store_metrics['坪效'] = store_metrics['總銷售額'] / store_metrics.apply(
        lambda x: 100 if x['Outlet_Size'] == 'Small' else (200 if x['Outlet_Size'] == 'Medium' else 300), axis=1
    )
    store_metrics['存貨周轉率'] = store_metrics['總銷售額'] / store_metrics['商品總價值']
    store_metrics['日均銷售額'] = store_metrics['總銷售額'] / 365
   
    # 顯示商店效率分析
    st.dataframe(store_metrics.style.format({
        '總銷售額': '${:,.2f}',
        '平均交易額': '${:,.2f}',
        '交易次數': '{:,.0f}',
        '商品總價值': '${:,.2f}',
        '坪效': '${:,.2f}/m²',
        '存貨周轉率': '{:,.2f}次/年',
        '日均銷售額': '${:,.2f}'
    }))

    # 3. 營運效率視覺化
    st.subheader("📊 營運效率視覺化")
   
    col1, col2 = st.columns(2)
   
    with col1:
        # 商品類別銷售效率
        fig = px.scatter(
            product_metrics.reset_index(),
            x='平均單價',
            y='單位利潤',
            size='銷售次數',
            color='銷售額佔比',
            hover_name='Item_Type',
            title='商品類別銷售效率矩陣'
        )
        st.plotly_chart(fig, use_container_width=True)
   
    with col2:
        # 商店類型效率比較
        fig = px.bar(store_metrics,
                    x='Outlet_Identifier',
                    y='坪效',
                    color='Outlet_Type',
                    title='商店坪效比較',
                    labels={'Outlet_Identifier': '商店', 'value': '坪效 ($/m²)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # 4. 營運效率KPI
    st.subheader("🎯 營運效率KPI")
   
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        avg_transaction = df['Item_Outlet_Sales'].mean()
        st.metric(
            label="平均交易金額",
            value=f"${avg_transaction:.2f}",
            help="平均每筆交易的金額"
        )
   
    with col2:
        inventory_turnover = store_metrics['存貨周轉率'].mean()
        st.metric(
            label="平均存貨周轉率",
            value=f"{inventory_turnover:.2f}次/年",
            help="平均每年存貨周轉次數"
        )
   
    with col3:
        space_efficiency = store_metrics['坪效'].mean()
        st.metric(
            label="平均坪效",
            value=f"${space_efficiency:.2f}/m²",
            help="平均每平方米的銷售額"
        )
   
    with col4:
        daily_sales = store_metrics['日均銷售額'].mean()
        st.metric(
            label="平均日銷售額",
            value=f"${daily_sales:.2f}",
            help="平均每日的銷售額"
        )

    # 5. 營運效率建議
    st.subheader("💡 營運效率改善建議")
   
    # 根據分析結果生成建議
    recommendations = []
   
    # 商品相關建議
    low_profit_products = product_metrics[product_metrics['單位利潤'] < 0].index.tolist()
    if low_profit_products:
        recommendations.append(f"• 以下商品類別利潤偏低，建議檢討定價策略：{', '.join(low_profit_products)}")
   
    low_turnover_stores = store_metrics[store_metrics['存貨周轉率'] < 8]['Outlet_Identifier'].tolist()
    if low_turnover_stores:
        recommendations.append(f"• 以下商店存貨周轉率偏低，建議優化庫存管理：{', '.join(low_turnover_stores)}")
   
    low_efficiency_stores = store_metrics[store_metrics['坪效'] < 500]['Outlet_Identifier'].tolist()
    if low_efficiency_stores:
        recommendations.append(f"• 以下商店坪效偏低，建議改善空間利用：{', '.join(low_efficiency_stores)}")
   
    # 顯示建議
    for rec in recommendations:
        st.markdown(rec)
   
    # 如果沒有特別的問題，顯示一般性建議
    if not recommendations:
        st.markdown("""
        整體營運效率良好，建議：
        • 持續監控商品組合，確保最佳獲利
        • 定期評估商店坪效，優化空間利用
        • 維持良好的存貨周轉率，避免資金積壓
        """)

def load_and_process_data(file):
    """載入並處理數據"""
    try:
        # 讀取上傳的文件
        df = pd.read_csv(file)
       
        # 檢查必要的列是否存在
        expected_columns = [
            'Item_Identifier', 'Item_Weight', 'Item_Fat_Content',
            'Item_Visibility', 'Item_Type', 'Item_MRP',
            'Outlet_Identifier', 'Outlet_Establishment_Year',
            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
        ]
       
        missing_columns = [col for col in expected_columns if col not in df.columns]
       
        if missing_columns:
            st.error(f"數據缺少必要的列: {', '.join(missing_columns)}")
            st.info("""
            請確保您的CSV文件包含所有必要列。
            目前缺少的列已在上方列出。
            """)
            return None
       
        # 基礎數據清理
        # 1. 處理缺失值
        df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
        df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
       
        # 2. 標準化分類變量
        df['Item_Fat_Content'] = df['Item_Fat_Content'].str.lower()
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
            'reg': 'regular',
            'low fat': 'low_fat',
            'lowfat': 'low_fat',
            'lf': 'low_fat',
            'regular': 'regular'
        })
       
        # 3. 計算基本的統計特徵
        df['Store_Age'] = 2025 - df['Outlet_Establishment_Year']  # 計算商店年齡
       
        # 顯示數據基本信息
        st.success(f"""
        數據加載成功！
        - 總記錄數：{len(df):,} 筆
        - 商品種類數：{df['Item_Type'].nunique()} 種
        - 商店數量：{df['Outlet_Identifier'].nunique()} 家
        """)
       
        return df
       
    except pd.errors.EmptyDataError:
        st.error("上傳的文件是空的，請檢查文件內容。")
        return None
    except Exception as e:
        st.error(f"數據處理時發生錯誤：{str(e)}")
        st.info("請確保您上傳的是正確格式的CSV文件。")
        return None

def show_header_metrics(df):
    """顯示頂部關鍵指標"""
    # 計算關鍵指標
    total_items = len(df['Item_Identifier'].unique())
    total_stores = len(df['Outlet_Identifier'].unique())
    avg_item_mrp = df['Item_MRP'].mean()
    avg_visibility = df['Item_Visibility'].mean() * 100  # 轉換為百分比
   
    # 創建四個列來顯示指標
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        st.metric(
            label="商品種類",
            value=f"{total_items:,}",
            help="獨特商品的總數量"
        )
   
    with col2:
        st.metric(
            label="商店數量",
            value=f"{total_stores:,}",
            help="商店的總數量"
        )
   
    with col3:
        st.metric(
            label="平均商品價格",
            value=f"¥{avg_item_mrp:.2f}",
            help="商品的平均標價"
        )
   
    with col4:
        st.metric(
            label="平均商品能見度",
            value=f"{avg_visibility:.2f}%",
            help="商品的平均展示佔比"
        )

def analyze_products(df):
    """商品分析"""
    st.header("📦 商品分析")
   
    # 創建兩列佈局
    col1, col2 = st.columns(2)
   
    with col1:
        st.subheader("商品類型分布")
        # 計算每種商品類型的統計數據
        type_stats = df.groupby('Item_Type').agg({
            'Item_Identifier': 'count',
            'Item_MRP': ['mean', 'min', 'max'],
            'Item_Weight': 'mean',
            'Item_Visibility': 'mean'
        }).round(2)
       
        # 重命名列
        type_stats.columns = [
            '商品數量',
            '平均價格',
            '最低價格',
            '最高價格',
            '平均重量',
            '平均能見度'
        ]
       
        # 顯示統計數據
        st.dataframe(type_stats.style.format({
            '平均價格': '¥{:.2f}',
            '最低價格': '¥{:.2f}',
            '最高價格': '¥{:.2f}',
            '平均重量': '{:.2f}kg',
            '平均能見度': '{:.2%}'
        }))
   
    with col2:
        st.subheader("商品價格分布")
        fig = px.box(df,
                    x='Item_Type',
                    y='Item_MRP',
                    title='各類型商品價格分布')
        st.plotly_chart(fig, use_container_width=True)
   
    # 商品脂肪含量分析
    st.subheader("商品脂肪含量分析")
    col3, col4 = st.columns(2)
   
    with col3:
        fat_content_dist = df['Item_Fat_Content'].value_counts()
        fig = px.pie(values=fat_content_dist.values,
                    names=fat_content_dist.index,
                    title='商品脂肪含量分布')
        st.plotly_chart(fig, use_container_width=True)
   
    with col4:
        avg_price_by_fat = df.groupby('Item_Fat_Content')['Item_MRP'].mean().round(2)
        fig = px.bar(x=avg_price_by_fat.index,
                    y=avg_price_by_fat.values,
                    title='不同脂肪含量商品的平均價格')
        st.plotly_chart(fig, use_container_width=True)
   
    # 商品能見度分析
    st.subheader("商品能見度分析")
    col5, col6 = st.columns(2)
   
    with col5:
        fig = px.histogram(df,
                          x='Item_Visibility',
                          title='商品能見度分布',
                          nbins=50)
        st.plotly_chart(fig, use_container_width=True)
   
    with col6:
        avg_visibility_by_type = df.groupby('Item_Type')['Item_Visibility'].mean().sort_values(ascending=False)
        fig = px.bar(x=avg_visibility_by_type.index,
                    y=avg_visibility_by_type.values,
                    title='各類型商品的平均能見度')
        st.plotly_chart(fig, use_container_width=True)

def analyze_stores(df):
    """商店分析"""
    st.header("🏪 商店分析")
   
    # 創建兩列佈局
    col1, col2 = st.columns(2)
   
    with col1:
        st.subheader("商店類型分布")
        # 計算每種商店類型的統計數據
        store_stats = df.groupby('Outlet_Type').agg({
            'Item_Identifier': 'count',
            'Item_MRP': ['mean', 'min', 'max'],
            'Outlet_Identifier': 'nunique'
        }).round(2)
       
        # 重命名列
        store_stats.columns = [
            '商品數量',
            '平均商品價格',
            '最低商品價格',
            '最高商品價格',
            '商店數量'
        ]
       
        # 顯示統計數據
        st.dataframe(store_stats.style.format({
            '平均商品價格': '¥{:.2f}',
            '最低商品價格': '¥{:.2f}',
            '最高商品價格': '¥{:.2f}',
            '商品數量': '{:,.0f}',
            '商店數量': '{:,.0f}'
        }))
   
    with col2:
        st.subheader("商店規模分布")
        size_dist = df.groupby(['Outlet_Type', 'Outlet_Size']).size().unstack(fill_value=0)
        fig = px.bar(size_dist,
                    title='不同類型商店的規模分布',
                    barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
   
    # 商店位置分析
    st.subheader("商店位置分析")
    col3, col4 = st.columns(2)
   
    with col3:
        location_dist = df.groupby('Outlet_Location_Type')['Outlet_Identifier'].nunique()
        fig = px.pie(values=location_dist.values,
                    names=location_dist.index,
                    title='商店位置分布')
        st.plotly_chart(fig, use_container_width=True)
   
    with col4:
        avg_price_by_location = df.groupby('Outlet_Location_Type')['Item_MRP'].mean().round(2)
        fig = px.bar(x=avg_price_by_location.index,
                    y=avg_price_by_location.values,
                    title='不同位置商店的平均商品價格')
        st.plotly_chart(fig, use_container_width=True)
   
    # 商店年齡分析
    st.subheader("商店年齡分析")
    col5, col6 = st.columns(2)
   
    with col5:
        df['Store_Age'] = 2025 - df['Outlet_Establishment_Year']
        fig = px.histogram(df,
                          x='Store_Age',
                          title='商店年齡分布',
                          nbins=20)
        st.plotly_chart(fig, use_container_width=True)
   
    with col6:
        avg_price_by_age = df.groupby('Store_Age')['Item_MRP'].mean().round(2)
        fig = px.line(x=avg_price_by_age.index,
                     y=avg_price_by_age.values,
                     title='商店年齡與平均商品價格的關係')
        st.plotly_chart(fig, use_container_width=True)

def perform_advanced_analysis(df):
    """進階分析"""
    st.header("🔍 進階分析")
   
    # 選擇要分析的特徵
    st.subheader("🎯 銷售預測模型")
   
    # 準備特徵
    features = ['Item_MRP', 'Item_Weight', 'Item_Visibility']
    target = 'Item_Outlet_Sales'  # 使用銷售額作為目標變量
   
    # 移除缺失值
    analysis_df = df[features + [target]].dropna()
   
    # 分割特徵和目標
    X = analysis_df[features]
    y = analysis_df[target]
   
    # 標準化特徵
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    # 訓練隨機森林模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)
   
    # 計算特徵重要性
    feature_importance = pd.DataFrame({
        '特徵': features,
        '重要性': rf_model.feature_importances_
    }).sort_values('重要性', ascending=True)
   
    # 顯示特徵重要性
    st.write("特徵重要性分析：")
   
    # 使用水平條形圖顯示特徵重要性
    fig = px.bar(feature_importance,
                x='重要性',
                y='特徵',
                orientation='h',
                title='特徵重要性分析')
   
    base_layout = create_figure_layout()
    base_layout['height'] = 400
    fig.update_layout(**base_layout)
   
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
   
    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_pred = rf_model.predict(X_test)
   
    # 顯示模型評估指標
    col1, col2 = st.columns(2)
   
    with col1:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        st.metric("均方根誤差 (RMSE)", f"{rmse:.2f}")
   
    with col2:
        r2 = r2_score(y_test, y_pred)
        st.metric("決定係數 (R²)", f"{r2:.2%}")
   
    # 預測結果散點圖
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={'x': '實際銷售額', 'y': '預測銷售額'},
        title='預測銷售額 vs 實際銷售額'
    )
   
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='理想線',
        line=dict(dash='dash', color='red')
    ))
   
    fig.update_layout(**create_figure_layout())
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
   
    # 模型解釋
    st.markdown("""
    #### 模型分析結果
    1. 特徵重要性排序顯示了各個因素對銷售額的影響程度
    2. 預測模型的準確度通過 R² 值來衡量，範圍從 0 到 1，越接近 1 表示模型越準確
    3. RMSE 值表示預測誤差的大小，數值越小表示預測越準確
   
    #### 應用建議
    1. 重點關注高重要性的特徵，優化相關策略
    2. 根據模型預測結果，調整庫存和定價策略
    3. 持續監控模型表現，定期更新預測模型
    """)

def perform_correlation_analysis(df):
    """執行相關性分析"""
    st.subheader("🔄 相關性分析")
   
    # 選擇數值型特徵
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
   
    # 使用SimpleImputer處理缺失值
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = df[numeric_features].copy()
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
   
    # 計算相關係數矩陣
    corr_matrix = X_imputed.corr()
   
    # 繪製熱力圖
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="相關係數"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu",
        title="特徵相關性熱力圖"
    )
    fig.update_layout(width=800, height=800)
    fig.update_layout(**create_figure_layout())
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="correlation_heatmap")
   
    # 執行PCA分析
    st.subheader("主成分分析 (PCA)")
   
    # 標準化數據
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
   
    # 執行PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
   
    # 計算解釋方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
   
    # 繪製解釋方差比圖
    fig = go.Figure()
   
    fig.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(len(explained_variance_ratio))],
        y=explained_variance_ratio,
        name='解釋方差比'
    ))
   
    fig.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(cumulative_variance_ratio))],
        y=cumulative_variance_ratio,
        name='累積解釋方差比',
        line=dict(color='red')
    ))
   
    fig.update_layout(title='主成分解釋方差比',
                     xaxis_title="",
                     yaxis_title="")
    fig.update_layout(**create_figure_layout())
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="pca_variance_ratio")
   
    # 顯示主成分載荷量
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=numeric_features
    )
   
    st.write("### 主成分載荷量")
    st.dataframe(loadings.style.format("{:.3f}"))
   
    # 分析和解釋主要相關性
    st.write("### 主要相關性分析")
   
    # 找出強相關的特徵對
    strong_correlations = []
    for i in range(len(numeric_features)):
        for j in range(i+1, len(numeric_features)):
            corr = corr_matrix.iloc[i,j]
            if abs(corr) > 0.5:  # 設定相關係數閾值
                strong_correlations.append({
                    'feature1': numeric_features[i],
                    'feature2': numeric_features[j],
                    'correlation': corr
                })
   
    # 按相關係數絕對值排序
    strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
   
    # 顯示強相關特徵對
    if strong_correlations:
        for idx, corr in enumerate(strong_correlations):
            correlation_type = "正相關" if corr['correlation'] > 0 else "負相關"
            st.write(f"**{corr['feature1']} 和 {corr['feature2']}**")
            st.write(f"- 相關係數: {corr['correlation']:.3f} ({correlation_type})")
           
            # 繪製散點圖
            fig = px.scatter(df,
                            x=corr['feature1'],
                            y=corr['feature2'],
                            title=f"{corr['feature1']} vs {corr['feature2']}",
                            trendline="ols")
            fig.update_layout(**create_figure_layout())
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"scatter_plot_{idx}")
           
            # 生成業務建議
            if abs(corr['correlation']) > 0.7:
                st.write("💡 **強相關性建議：**")
                if corr['correlation'] > 0:
                    st.write(f"- 考慮將{corr['feature1']}和{corr['feature2']}作為組合指標")
                    st.write(f"- 可以通過提升{corr['feature1']}來帶動{corr['feature2']}的增長")
                else:
                    st.write(f"- 注意{corr['feature1']}和{corr['feature2']}之間的權衡關係")
                    st.write(f"- 需要在兩者之間找到最佳平衡點")
    else:
        st.write("未發現顯著的特徵相關性（相關係數絕對值 > 0.5）")

def perform_price_analysis(df):
    """價格分析"""
    st.header("💰 價格分析")
   
    if 'Item_MRP' not in df.columns:
        st.warning("未找到價格列（Item_MRP）。請確保數據包含價格資訊。")
        return
   
    # 基本價格統計
    st.subheader("基本價格統計")
    price_stats = df['Item_MRP'].describe()
   
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均價格", f"¥{price_stats['mean']:.2f}")
    with col2:
        st.metric("最高價格", f"¥{price_stats['max']:.2f}")
    with col3:
        st.metric("最低價格", f"¥{price_stats['min']:.2f}")
    with col4:
        st.metric("價格標準差", f"¥{price_stats['std']:.2f}")
   
    # 價格分布分析
    st.subheader("價格分布分析")
   
    # 創建價格區間
    price_bins = pd.qcut(df['Item_MRP'], q=10, labels=[f'第{i+1}分位' for i in range(10)])
    price_distribution = df.groupby(price_bins).agg({
        'Item_MRP': ['count', 'mean'],
        'Item_Weight': 'sum'
    })
   
    price_distribution.columns = ['商品數量', '平均價格', '總重量']
   
    # 顯示價格分布表格
    st.write("價格分布統計：")
    st.dataframe(price_distribution.style.format({
        '商品數量': '{:,d}',
        '平均價格': '¥{:,.2f}',
        '總重量': '{:,.2f}kg'
    }))
   
    # 價格分布直方圖
    fig = px.histogram(df,
                      x='Item_MRP',
                      nbins=30,
                      title='價格分布直方圖',
                      labels={'Item_MRP': '價格', 'count': '商品數量'})
    fig.update_layout(**create_figure_layout())
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
   
    # 價格與銷售額的關係
    st.subheader("價格與銷售額關係分析")
   
    # 散點圖
    fig = px.scatter(df,
                    x='Item_MRP',
                    y='Item_Weight',
                    title='價格與銷售額關係',
                    labels={'Item_MRP': '價格',
                           'Item_Weight': '銷售額'})
    fig.update_layout(**create_figure_layout())
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
   
    # 計算價格彈性
    if 'Item_Weight' in df.columns:
        # 計算平均價格和平均銷量
        avg_price = df['Item_MRP'].mean()
        avg_sales = df['Item_Weight'].mean()
       
        # 計算價格變化率和銷量變化率
        price_pct_change = (df['Item_MRP'] - avg_price) / avg_price
        sales_pct_change = (df['Item_Weight'] - avg_sales) / avg_sales
       
        # 計算價格彈性
        price_elasticity = sales_pct_change.mean() / price_pct_change.mean()
       
        st.metric("價格彈性", f"{abs(price_elasticity):.2f}",
                 help="價格彈性表示價格變動對銷售量的影響程度。\n"
                      "數值越大表示價格變動對銷售量的影響越大。")
   
    # 按商品類型的價格分析
    if 'Item_Type' in df.columns:
        st.subheader("商品類型價格分析")
       
        # 計算每種商品類型的價格統計
        type_price_stats = df.groupby('Item_Type').agg({
            'Item_MRP': ['mean', 'min', 'max', 'std'],
            'Item_Weight': 'sum'
        }).round(2)
       
        type_price_stats.columns = ['平均價格', '最低價格', '最高價格', '價格標準差', '總重量']
       
        # 排序並顯示
        type_price_stats = type_price_stats.sort_values('平均價格', ascending=False)
       
        st.write("商品類型價格統計：")
        st.dataframe(type_price_stats.style.format({
            '平均價格': '¥{:,.2f}',
            '最低價格': '¥{:,.2f}',
            '最高價格': '¥{:,.2f}',
            '價格標準差': '¥{:,.2f}',
            '總重量': '{:,.2f}kg'
        }))
   
    # 價格優化建議
    st.subheader("價格優化建議")
   
    # 分析高利潤商品
    if all(col in df.columns for col in ['Item_MRP', 'Item_Weight']):
        df['Profit_Margin'] = (df['Item_Weight'] - df['Item_MRP']) / df['Item_MRP']
       
        high_profit_items = df[df['Profit_Margin'] > df['Profit_Margin'].quantile(0.75)]
       
        st.write("高利潤商品分析：")
        high_profit_stats = high_profit_items.groupby('Item_Type').agg({
            'Item_MRP': 'mean',
            'Profit_Margin': 'mean',
            'Item_Identifier': 'count'
        }).round(3)
       
        high_profit_stats.columns = ['平均價格', '平均利潤率', '商品數量']
        st.dataframe(high_profit_stats.style.format({
            '平均價格': '¥{:.2f}',
            '平均利潤率': '{:.1%}',
            '商品數量': '{:,.0f}'
        }))
       
        # 生成價格優化建議
        st.write("價格優化建議：")
       
        # 基於價格彈性的建議
        if abs(price_elasticity) > 1:
            st.info("📊 市場對價格較為敏感，建議：\n"
                   "1. 考慮實施差異化定價策略\n"
                   "2. 進行小幅度價格調整測試\n"
                   "3. 關注競爭對手的價格變動")
        else:
            st.info("📊 市場對價格相對不敏感，建議：\n"
                   "1. 可以考慮提高高利潤商品的價格\n"
                   "2. 重點關注產品質量和品牌建設\n"
                   "3. 開發高端市場細分")
       
        # 基於利潤率的建議
        high_profit_categories = high_profit_stats[high_profit_stats['平均利潤率'] > 0.2].index.tolist()
        if high_profit_categories:
            st.info(f"💡 以下類別具有較高利潤率，建議增加庫存和促銷力度：\n" +
                   "\n".join([f"- {cat}" for cat in high_profit_categories]))

def analyze_trends(df):
    """趨勢預測分析"""
    st.header("📈 趨勢預測分析")
   
    # 基本趨勢分析
    st.subheader("📊 商品類型趨勢分析")
   
    # 按商品類型分析
    type_trends = df.groupby('Item_Type').agg({
        'Item_Weight': ['mean', 'count'],
        'Item_MRP': 'mean',
        'Item_Visibility': 'mean'
    }).round(3)
   
    type_trends.columns = ['平均重量', '商品數量', '平均價格', '平均可見度']
   
    # 顯示商品類型趨勢
    st.write("各商品類型統計：")
    st.dataframe(type_trends.style.format({
        '平均重量': '{:.2f}kg',
        '商品數量': '{:,.0f}',
        '平均價格': '¥{:,.2f}',
        '平均可見度': '{:.3%}'
    }))
   
    # 商品類型分布趨勢圖
    col1, col2 = st.columns(2)
   
    with col1:
        # 商品類型數量分布
        fig = px.bar(type_trends.reset_index(),
                    x='Item_Type',
                    y='商品數量',
                    title='商品類型數量分布',
                    labels={'Item_Type': '商品類型', '商品數量': '數量'})
        fig.update_layout(**create_figure_layout())
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
   
    with col2:
        # 商品類型平均價格
        fig = px.bar(type_trends.reset_index(),
                    x='Item_Type',
                    y='平均價格',
                    title='商品類型平均價格',
                    labels={'Item_Type': '商品類型', '平均價格': '價格'})
        fig.update_layout(**create_figure_layout())
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
   
    # 商品可見度分析
    st.subheader("👁️ 商品可見度分析")
   
    col1, col2 = st.columns(2)
   
    with col1:
        # 可見度與價格的關係
        fig = px.scatter(df,
                        x='Item_Visibility',
                        y='Item_MRP',
                        color='Item_Type',
                        title='商品可見度與價格關係',
                        labels={'Item_Visibility': '可見度',
                               'Item_MRP': '價格',
                               'Item_Type': '商品類型'})
        fig.update_layout(**create_figure_layout())
        st.plotly_chart(fig, use_container_width=True)
   
    with col2:
        # 可見度與重量的關係
        fig = px.scatter(df,
                        x='Item_Visibility',
                        y='Item_Weight',
                        color='Item_Type',
                        title='商品可見度與重量關係',
                        labels={'Item_Visibility': '可見度',
                               'Item_Weight': '重量',
                               'Item_Type': '商品類型'})
        fig.update_layout(**create_figure_layout())
        st.plotly_chart(fig, use_container_width=True)
   
    # 商店規模趨勢分析
    st.subheader("🏪 商店規模趨勢分析")
   
    # 按商店規模和位置分析
    store_trends = df.groupby(['Outlet_Size', 'Outlet_Location_Type']).agg({
        'Item_MRP': ['mean', 'count'],
        'Item_Visibility': 'mean'
    }).round(3)
   
    store_trends.columns = ['平均價格', '商品數量', '平均可見度']
   
    # 重置索引以便於繪圖
    store_trends_reset = store_trends.reset_index()
   
    col1, col2 = st.columns(2)
   
    with col1:
        # 不同規模商店的商品數量
        fig = px.bar(store_trends_reset,
                    x='Outlet_Size',
                    y='商品數量',
                    color='Outlet_Location_Type',
                    title='不同規模商店的商品數量',
                    barmode='group',
                    labels={'Outlet_Size': '商店規模',
                           'Outlet_Location_Type': '位置類型',
                           '商品數量': '數量'})
        st.plotly_chart(fig, use_container_width=True)
   
    with col2:
        # 不同規模商店的平均價格
        fig = px.bar(store_trends_reset,
                    x='Outlet_Size',
                    y='平均價格',
                    color='Outlet_Location_Type',
                    title='不同規模商店的平均價格',
                    barmode='group',
                    labels={'Outlet_Size': '商店規模',
                           'Outlet_Location_Type': '位置類型',
                           '平均價格': '價格'})
        st.plotly_chart(fig, use_container_width=True)
   
    # 商店類型分析
    st.subheader("🏬 商店類型分析")
   
    # 按商店類型分析
    outlet_type_trends = df.groupby('Outlet_Type').agg({
        'Item_MRP': ['mean', 'count'],
        'Item_Visibility': 'mean',
        'Item_Weight': 'mean'
    }).round(3)
   
    outlet_type_trends.columns = ['平均價格', '商品數量', '平均可見度', '平均重量']
   
    # 顯示商店類型統計
    st.write("各商店類型統計：")
    st.dataframe(outlet_type_trends.style.format({
        '平均價格': '¥{:,.2f}',
        '商品數量': '{:,.0f}',
        '平均可見度': '{:.3%}',
        '平均重量': '{:.2f}kg'
    }))
   
    # 趨勢預測建議
    st.subheader("💡 趨勢預測建議")
   
    # 計算一些關鍵指標
    high_visibility_types = df.groupby('Item_Type')['Item_Visibility'].mean().nlargest(3)
    high_price_types = df.groupby('Item_Type')['Item_MRP'].mean().nlargest(3)
    best_performing_locations = df.groupby('Outlet_Location_Type')['Item_MRP'].mean().nlargest(2)
   
    st.markdown(f"""
    #### 商品策略建議
    1. 高曝光度商品類型：
       {', '.join([f'**{t}** ({v:.1%})' for t, v in high_visibility_types.items()])}
       - 建議增加這些類型的商品陳列空間
       - 可考慮在這些類別推出新品
   
    2. 高價值商品類型：
       {', '.join([f'**{t}** (¥{p:.2f})' for t, p in high_price_types.items()])}
       - 建議優化這些類型的商品組合
       - 考慮開發相關的高端產品線
   
    #### 商店發展建議
    1. 表現最佳的位置類型：
       {', '.join([f'**{l}** (¥{p:.2f})' for l, p in best_performing_locations.items()])}
       - 建議在這些位置類型優先擴展新店
       - 可以將這些位置的成功經驗推廣到其他地區
   
    2. 商店規模策略：
       - 根據數據顯示，{store_trends_reset.groupby('Outlet_Size')['平均價格'].mean().idxmax()}規模商店的平均價格最高
       - 建議在新開店時優先考慮此規模
    """)

def generate_operation_diagnosis(data_summary, risk_summary):
    """生成運營診斷報告"""
    prompt = f"""
    請基於以下數據和風險指標，生成一份全面且深入的運營診斷報告，需包含以下六個部分：

    1. 整體經營狀況評估
       - 數據規模和覆蓋範圍分析
       - 商品結構評估
       - 門店運營效率分析
       - 價格策略評估
       - 整體營運健康度評估

    2. 關鍵指標分析
       - 銷售預測準確度解讀
       - 庫存週轉效率分析
       - 商品多樣性評估
       - 價格區間覆蓋率分析
       - 各指標之間的關聯性

    3. 主要風險點識別
       - 運營風險評估
       - 市場風險評估
       - 庫存風險評估
       - 價格風險評估
       - 競爭風險評估

    4. 具體優化建議
       - 商品結構優化方案
       - 庫存管理改進建議
       - 價格策略調整方案
       - 門店運營優化建議
       - 銷售預測系統改進建議

    5. 數據驅動的行動計劃
       - 短期（1-3個月）改進計劃
       - 中期（3-6個月）優化方案
       - 長期（6-12個月）發展規劃
       - 具體執行步驟和時間表
       - 預期效果和關鍵績效指標

    6. 未來發展方向
       - 市場趨勢分析
       - 業務擴展機會
       - 技術升級建議
       - 人才發展規劃
       - 創新機會點識別

    風險指標：
    {risk_summary}
   
    請用中文回答，確保每個部分都有詳細、具體且可執行的分析和建議。分析時需要：
    1. 結合具體數據支持你的觀點
    2. 提供可量化的改進目標
    3. 考慮實際執行的可行性
    4. 平衡短期效益和長期發展
    5. 突出優先級和重要性
    """
   
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",  # 使用更大上下文的模型
            messages=[
                {"role": "system", "content": "你是一位資深的零售業運營分析專家，擅長數據分析、戰略規劃和提供具體可行的改進建議。你的分析需要全面、深入且具有戰略性，同時保持務實和可執行性。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000  # 增加輸出長度
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"""
        無法生成AI診斷報告：{str(e)}

        整體經營狀況評估：
        - 根據數據顯示，目前營運狀況穩定
       
        主要風險點識別：
        1. 請檢查數據完整性
        2. 關注庫存管理情況
       
        具體優化建議：
        1. 定期進行數據分析
        2. 優化庫存管理流程
       
        未來發展方向：
        1. 加強數據分析能力
        2. 持續優化營運流程
        """

def generate_pdf_report(data_summary, risk_summary, diagnosis):
    """生成PDF格式的診斷報告"""
    buffer = BytesIO()
   
    # 創建PDF文件
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.units import inch
   
    # 註冊中文字體
    try:
        pdfmetrics.registerFont(TTFont('MSJhengHei', 'C:/Windows/Fonts/msjh.ttc'))
    except:
        # 如果找不到中文字體，使用預設字體
        pass
   
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
   
    # 定義樣式
    styles = getSampleStyleSheet()
    chinese_style = ParagraphStyle(
        'ChineseStyle',
        parent=styles['Normal'],
        fontName='MSJhengHei' if 'MSJhengHei' in pdfmetrics.getRegisteredFontNames() else 'Helvetica',
        fontSize=10,
        leading=14,
        spaceBefore=6,
        spaceAfter=6
    )
   
    title_style = ParagraphStyle(
        'ChineseTitle',
        parent=styles['Title'],
        fontName='MSJhengHei' if 'MSJhengHei' in pdfmetrics.getRegisteredFontNames() else 'Helvetica',
        fontSize=24,
        leading=30,
        alignment=1,
        spaceAfter=30,
    )
   
    heading_style = ParagraphStyle(
        'ChineseHeading',
        parent=styles['Heading1'],
        fontName='MSJhengHei' if 'MSJhengHei' in pdfmetrics.getRegisteredFontNames() else 'Helvetica',
        fontSize=16,
        leading=20,
        spaceBefore=12,
        spaceAfter=6,
    )
   
    story = []
   
    # 添加標題
    story.append(Paragraph("運營診斷報告", title_style))
    story.append(Spacer(1, 20))
   
    # 添加風險指標
    story.append(Paragraph("⚠️ 風險指標", heading_style))
    story.append(Paragraph(risk_summary.replace('\n', '<br/>'), chinese_style))
    story.append(Spacer(1, 10))
   
    # 添加診斷報告
    story.append(Paragraph("📋 運營診斷報告", heading_style))
    story.append(Paragraph(diagnosis.replace('\n', '<br/>'), chinese_style))
   
    # 添加頁腳
    footer_style = ParagraphStyle(
        'Footer',
        parent=chinese_style,
        fontSize=8,
        textColor=colors.gray,
        alignment=1
    )
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"報告生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
   
    # 生成PDF
    doc.build(story)
   
    pdf_data = buffer.getvalue()
    buffer.close()
   
    return pdf_data

def get_ai_response(consultation_content, data_context):
    """生成AI諮詢回應"""
    try:
        client = openai.OpenAI(api_key=api_key)
        prompt = f"""
        作為一位專業的商業顧問，請針對以下諮詢內容提供專業的建議和分析：

        {consultation_content}

        請提供：
        1. 問題分析
        2. 具體建議
        3. 可行的解決方案
        4. 後續跟進建議

        回答需要專業、具體且實用。
        """
       
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位專業的商業顧問，擅長提供具體可行的商業建議。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成回應時發生錯誤：{str(e)}"

def main():
    st.title("🏪 商店銷售分析系統")
   
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
   
    # Sidebar navigation
    with st.sidebar:
        st.title("系統介紹")
        st.markdown("""
        ### 🏪 商店銷售分析系統
       
        這是一個全方位的商業分析工具，幫助您更好地理解您的業務數據：
       
        #### 📊 主要功能
        1. **基礎數據分析**
           - 商品銷售分析
           - 商店營運分析
           - 趨勢預測分析
       
        2. **進階分析工具**
           - 相關性分析
           - 價格分析策略
           - 進階數據分析
       
        3. **財務報表分析**
           - 損益表分析
           - 客戶收入報表
           - 資產負債表
       
        4. **營運管理工具**
           - 財務比率分析
           - 營運指標分析
           - 運營診斷報告
       
        #### 💡 使用說明
        1. 上傳您的CSV格式銷售數據
        2. 系統會自動生成各項分析報告
        3. 可下載診斷報告作為決策參考
       
        #### 📈 數據要求
        - CSV檔案格式
        - 需包含銷售、成本等基本數據
        - 建議包含時間戳記以進行趨勢分析
        """)
       
        st.markdown("---")
        st.markdown("### 📫 聯絡資訊")
        st.markdown("如有任何問題，請聯繫系統管理員")
       
    # 上傳資料
    uploaded_file = st.file_uploader("請上傳銷售數據 (CSV格式)", type=['csv'])
   
    if uploaded_file is not None:
        # 讀取並處理數據
        df = load_and_process_data(uploaded_file)
       
        if df is not None:
            # 顯示數據概覽
            st.header("📊 數據概覽")
            st.write(df.head())
           
            # 顯示頂部指標
            show_header_metrics(df)
           
            # 使用 tabs 進行分頁展示
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 , tab12= st.tabs([
                "📦 商品分析",
                "🏪 商店分析",
                "📈 趨勢預測",
                "🔬 進階分析",
                "🔄 相關性分析",
                "💰 價格分析",
                "📊 損益表",
                "👥 客戶收入報表",
                "💰 資產負債表",
                "📈 財務比率分析",
                "📊 營運指標分析",
                " 運營診斷報告",

            ])
           
            with tab1:
                analyze_products(df)
           
            with tab2:
                analyze_stores(df)
           
            with tab3:
                analyze_trends(df)
           
            with tab4:
                perform_advanced_analysis(df)
           
            with tab5:
                perform_correlation_analysis(df)
           
            with tab6:
                perform_price_analysis(df)
           
            with tab7:
                generate_profit_loss_statement(df)
           
            with tab8:
                generate_customer_revenue_report(df)
           
            with tab9:
                generate_balance_sheet(df)
           
            with tab10:
                generate_financial_ratios(df)
           
            with tab11:
                generate_operational_metrics(df)
           
            with tab12:
                data_summary = df.describe().to_string()
                risk_summary = "風險指標：\n- 根據數據分析，風險指標包括商品銷售額、商店營運狀況等。"
                diagnosis = generate_operation_diagnosis(data_summary, risk_summary)
                st.write(diagnosis)
                pdf_data = generate_pdf_report(data_summary, risk_summary, diagnosis)
                st.download_button("下載診斷報告", pdf_data, "診斷報告.pdf", "application/pdf")

if __name__ == "__main__":
    main()

