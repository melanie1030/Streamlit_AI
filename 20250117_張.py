
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
import matplotlib.colors as mcolors
from openai import OpenAI
import os
import streamlit as st
from PIL import Image
from io import BytesIO
import base64
from datetime import datetime
import json
import seaborn as sns 
import tempfile
import io
from pandas.api.types import is_numeric_dtype
from matplotlib.font_manager import FontProperties
# Set the font to a font that supports Chinese characters
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK']
plt.rcParams['axes.unicode_minus'] = False  # 確保負號顯示正常

file_ids = {}
def execute_code_and_generate_image(code: str) -> tuple[bool, str, BytesIO]:
    """
    執行Python程式碼並生成圖表圖片
    返回: (是否成功, 錯誤訊息, 圖片緩衝區)
    """
    try:
        # 創建一個新的圖表
        plt.figure(figsize=(10, 6))
        
        # 執行代碼
        exec(code, globals())
        
        # 獲取當前圖表
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        
        # 重置緩衝區位置
        buffer.seek(0)
        
        return True, None, buffer
    except Exception as e:
        plt.close()  # 確保關閉任何打開的圖表
        return False, str(e), None
def get_output_path(filename, folder="visualizations"):
    """
    返回文件的完整路径，同时确保文件夹存在。
    
    :param filename: str, 文件名
    :param folder: str, 文件所在的文件夹
    :return: str, 文件的完整路径
    """
    os.makedirs(folder, exist_ok=True)  # 如果文件夹不存在，则创建
    return os.path.join(folder, filename)  # 返回完整路径

def clean_csv(file_path, output_path=None):
    """
    自動清理 CSV 文件數據並處理常見問題。
    :param file_path: str, 原始 CSV 文件的路徑。
    :param output_path: str, 清理後數據保存的路徑（可選）。
    :return: pandas.DataFrame, 清理後的數據。
    """
    
    try:
        # 讀取 CSV 文件
        df = pd.read_csv(file_path)  # 加載原始 CSV 文件到 DataFrame
        print(f"原始數據:\n{df.head()}\n")
        
        # 1. 刪除全空的列與行
        df.dropna(how="all", inplace=True)  # 刪除全为空值的行
        df.dropna(axis=1, how="all", inplace=True)  # 刪除全为空值的列
        
        # 2. 填補空值（對數值型和字串型分别處理）
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna("Unknown")       # 3. 自動檢查並處理異常值（以數字列為主）
        for col in df.select_dtypes(include=[np.number]).columns:  # 選擇所有數值列
            mean, std = df[col].mean(), df[col].std()  # 計算均值和標準差
            lower_bound, upper_bound = mean - 3 * std, mean + 3 * std  # 定義合理範圍
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]  # 過濾掉超出範圍的異常值
        
        # 4. 去除重複數據
        df.drop_duplicates(inplace=True)  # 刪除重複行
        
        # 5. 清理列名與字串空格
        df.columns = df.columns.str.strip()  # 去掉列名的多餘空格
        for col in df.select_dtypes(include=["object"]).columns:  # 對字符串列去除空格
            df[col] = df[col].str.strip()
        
        print(f"清理後數據:\n{df.head()}\n")
        
        # 如果指定了輸出路徑，保存清理後的數據
        if output_path:
            df.to_csv(output_path, index=False)  # 將清理後的數據保存到文件
            print(f"清理後的數據已保存到 {output_path}")
        
        return df  # 返回清理後的 DataFrame
    
    except Exception as e:
        print(f"清理過程中出現錯誤: {e}")  # 打印錯誤信息
        return None

def convert_csv_to_jsonl(csv_file_path, jsonl_file_path):
    """
    將 CSV 文件轉換為 JSONL 文件
    :param csv_file_path: str, CSV 文件的路徑
    :param jsonl_file_path: str, JSONL 文件的保存路徑
    """
    try:
        # 讀取 CSV 文件
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        # 將每行轉換為 JSON 格式，並寫入 JSONL 文件
        with open(jsonl_file_path, 'w', encoding='utf-8') as file:
            for record in df.to_dict(orient='records'):
                file.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"JSONL 文件已成功保存到: {jsonl_file_path}")
    except Exception as e:
        print(f"轉換過程中出現錯誤: {e}")
# 將圖像轉換為Base64編碼的函數
def get_image_base64(image_raw):
    """將PIL Image物件轉換為base64編碼
    Args:
        image_raw: PIL Image物件
    Returns:
        str: base64編碼的圖片字串
    """
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format or 'PNG')
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def save_and_encode_chart(fig, chart_number):
    """儲存圖表並轉換為base64編碼
    Args:
        fig: matplotlib figure物件
        chart_number: 圖表編號
    Returns:
        tuple: (base64編碼的圖片字串, 圖片路徑)
    """
    # 儲存圖表
    chart_path = f"chart_{chart_number}.png"
    fig.savefig(chart_path)
    
    # 讀取圖片並轉換為base64
    with Image.open(chart_path) as img:
        base64_str = get_image_base64(img)
    
    return base64_str, chart_path

def add_chart_to_messages(chart_path):
    """
    將圖表加入到對話訊息中
    Args:
        chart_path: 圖片路徑
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 加入圖表到訊息
    chart_message = f"""生成的圖表：
                        圖表 #{len(st.session_state.generated_charts) + 1}
                        生成時間：{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
                        """
    st.session_state.messages.append({
        "role": "assistant",
        "content": chart_message
    })
    
    # 儲存圖表資訊
    if "generated_charts" not in st.session_state:
        st.session_state.generated_charts = []
    
    st.session_state.generated_charts.append({
        "number": len(st.session_state.generated_charts) + 1,
        "path": chart_path,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# 生成 EDA 报告
def generate_eda_report(data):
    columns = data.columns.tolist()
    shape = data.shape
    missing_values = data.isnull().sum()
    numeric_cols = data.select_dtypes(include=['float64']).columns.tolist()
    integer_cols = data.select_dtypes(include=['int64']).columns.tolist()

    report = (
        f"Your dataset contains the following information:\n\n"
        f"Column names: A total of {len(columns)} columns, including {', '.join(columns)}.\n"
        f"Data dimensions: {shape[0]} rows and {shape[1]} columns.\n"
        f"Missing values:\n" +
        "\n".join([f"Column '{col}' has {missing_values[col]} missing values." for col in columns if missing_values[col] > 0]) +
        ("\nNo missing values.\n" if missing_values.sum() == 0 else "") +
        "\nData types:\n"
        f"Numeric columns: {', '.join(numeric_cols)}\n"
        f"Integer columns: {', '.join(integer_cols)} (possibly categorical variables)\n\n"
        "Next steps:\n\n"
        "1. Handle missing values using appropriate methods.\n"
        "2. Plot a correlation matrix to understand feature relationships.\n"
        "3. Perform basic feature engineering.\n"
    )
    return report

# 保存 EDA 報告為 JSONL 文件
def save_eda_report_as_jsonl(report, output_path):
    with open(output_path, 'w', encoding='utf-8') as jsonl_file:
        jsonl_file.write(json.dumps({"text": report}, ensure_ascii=False) + '\n')

# 缺失值处理
def handle_missing_values(data):
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            data[col] = data[col].fillna(data[col].median())
        elif data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
    return data
# 绘制并保存相关性矩阵
def plot_correlation_matrix(data, output_path):
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    print(f"相關性矩陣:\n{correlation_matrix}\n")  # Print 相關性矩陣
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix of Features")
    st.pyplot(plt)
    plt.savefig(output_path)
    print(f"相關性矩陣圖表已保存到 {output_path}")  # Print 圖表保存路徑
    plt.close()


# 圖像轉換為 Base64
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded

# 上傳 JSONL 文件到 OpenAI
def upload_jsonl_to_openai(client, jsonl_path):
    with open(jsonl_path, "rb") as file:
        response = client.files.create(file=file, purpose="fine-tune")
    return response.id

# 上傳 Base64 圖像到 OpenAI
def upload_base64_image_to_openai(client, base64_image, file_name):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file_name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(base64.b64decode(base64_image))

    with open(temp_file_path, "rb") as file:
        response = client.files.create(file=file, purpose="user_data")
        file_id = response.id

    os.remove(temp_file_path)
    return file_id

# 保存特徵工程處理後的數據
def save_processed_data(data, output_path):
    data.to_csv(output_path, index=False)

# 上傳處理後的數據
def upload_processed_data(client, processed_data_path):
    with open(processed_data_path, "rb") as file:
        response = client.files.create(file=file, purpose="user_data")
    return response.id

# 请求 OpenAI 生成特征工程代码
def Feature_Engineering_report(client, eda_report, file_ids):

    prompt = f"""
    You are tasked with creating a detailed data analysis report based on the following uploaded files:

    1. EDA Report (JSONL file): File ID {file_ids['eda_jsonl']}
    2. Correlation Matrix: File ID {file_ids['correlation_matrix']}
    3. Processed Data (CSV): File ID {file_ids['processed_data']}

    The report should include:
    - Key insights from the EDA report.
    - Interpretation of the correlation matrix.
    - Detailed description of feature engineering steps taken.
    - Recommendations for further analysis or model building.

    EDA Report:
    {eda_report}
    """

    try:
        # 使用 chat.completions.create 生成報告
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        # 提取生成的報告
        report = response.choices[0].message.content
        print(f"生成的報告:\n{report}\n")
        return report

    except Exception as e:
        print(f"發生錯誤：{e}")
        raise RuntimeError(f"Error during report generation: {e}")

# 提取代码块
def extract_code_blocks(content):
    clean_code_lines = []
    in_code_block = False
    for line in content.splitlines():
        if "```" in line:
            in_code_block = not in_code_block
            continue
        if in_code_block:
            clean_code_lines.append(line)
    return "\n".join(clean_code_lines).strip()

def validate_font_code(code):
    if "plt.rcParams['font.sans-serif']" not in code:
        return False, "生成的代碼缺少字型設置。"
    return True, "代碼驗證成功。"
def generate_final_report(client, eda_report, file_ids):
    """
    Generates a comprehensive final report by utilizing the OpenAI Chat API with pre-uploaded file IDs.

    Args:
        client (OpenAI): The initialized OpenAI client.
        eda_report (str): The exploratory data analysis report.
        file_ids (dict): A dictionary containing file IDs for different parts of the report.

    Returns:
        str: The generated final report summarizing the EDA, correlations, and feature engineering.
    """
    required_keys = ["eda_jsonl", "correlation_matrix", "feature_engineering_report", "processed_data"]
    missing_keys = [key for key in required_keys if key not in file_ids]
    if missing_keys:
        raise ValueError(f"Missing required file IDs: {', '.join(missing_keys)}")

    # 构建最终报告的提示
    prompt = f"""
    You are tasked with creating a detailed data analysis report based on the following uploaded files:

    1. EDA Report (JSONL file): File ID {file_ids['eda_jsonl']}
    2. Correlation Matrix: File ID {file_ids['correlation_matrix']}
    3. Feature Engineering Report: File ID {file_ids['feature_engineering_report']}
    4. Processed Data (CSV): File ID {file_ids['processed_data']}

    The report should include:
    - Key insights from the EDA report.
    - Interpretation of the correlation matrices (both original and processed).
    - Detailed description of feature engineering steps taken.
    - Recommendations for further analysis or model building.

    EDA Report:
    {eda_report}
    """

    # 调用 Chat API 生成报告
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a highly skilled data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        # 提取生成的报告内容
        full_report = response.choices[0].message.content
        print(f"生成的最終報告:\n{full_report}")
        return full_report

    except Exception as e:
        raise RuntimeError(f"Error during final report generation: {e}")

# 生成最终报告
def final_report(df):
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 生成 EDA 報告
    eda_report = generate_eda_report(df)
    st.write("### EDA Report")
    st.write(eda_report)

    # 保存並上傳 EDA 報告
    eda_jsonl_path = os.path.join(output_dir, "eda_report.jsonl")
    with open(eda_jsonl_path, "w", encoding="utf-8") as file:
        file.write(json.dumps({"text": eda_report}, ensure_ascii=False) + "\n")
    file_ids['eda_jsonl'] = upload_jsonl_to_openai(client, eda_jsonl_path)

    # 2. 缺失值處理
    df = handle_missing_values(df)

    # 3. 繪製相關性矩陣
    correlation_path = os.path.join(output_dir, "correlation_matrix.png")
    plot_correlation_matrix(df, correlation_path)
    file_ids['correlation_matrix'] = upload_processed_data(client, correlation_path)

    # 4. 保存處理後的數據並上傳
    processed_jsonl_path = os.path.join(output_dir, "processed_data.jsonl")
    with open(processed_jsonl_path, "w", encoding="utf-8") as file:
        for record in df.to_dict(orient="records"):
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    file_ids['processed_data'] = upload_jsonl_to_openai(client, processed_jsonl_path)

    # 5. 生成特徵工程報告
    feature_engineering_report = Feature_Engineering_report(client, eda_report, file_ids)

    # 保存並上傳特徵工程報告
    feature_report_path = os.path.join(output_dir, "feature_engineering_report.jsonl")
    with open(feature_report_path, "w", encoding="utf-8") as file:
        file.write(json.dumps({"text": feature_engineering_report}, ensure_ascii=False) + "\n")
    file_ids['feature_engineering_report'] = upload_jsonl_to_openai(client, feature_report_path)

    # 6. 生成最終報告
    return generate_final_report(client, eda_report, file_ids)


# 設置頁面配置
st.set_page_config(page_title="CSV 數據分析與視覺化聊天機器人", page_icon="📊")

# 側邊欄 API Key 輸入
st.sidebar.title("OpenAI API Key")
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key",
    placeholder="Paste your OpenAI API key, sk-",
    type="password"
)


# 主頁面標題
st.title("CSV數據分析聊天機器人")

# 檢查 API Key
if not user_api_key:
    st.warning("請在側邊欄輸入您的 OpenAI API Key")
    st.stop()
else:
    client = OpenAI(api_key=user_api_key)
    llm = ChatOpenAI(
        api_key=user_api_key,
        model="gpt-4",
        temperature=0.3,
        streaming=True,
        MAX_TOKENS=600
    )

# 初始化會話狀態
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "chart_counter" not in st.session_state:
    st.session_state.chart_counter = 0
if "generated_charts" not in st.session_state:
    st.session_state.generated_charts = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# 上傳 CSV 文件
uploaded_file = st.sidebar.file_uploader("選擇 CSV 文件", type="csv")
# 初始化 ChatOpenAI 模型

if uploaded_file is not None and not st.session_state.analysis_done:

    # 創建 Pandas DataFrame Agent
    
    try:
        # 清理 CSV 文件
        df = clean_csv(uploaded_file)  # 清理上傳的 CSV 文件
        agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True
                )
        if df is not None and not df.empty:
            st.session_state.df = df  # 將清理後的數據保存到 session state
            st.sidebar.write("資料集內容：")
            st.sidebar.dataframe(df)
            
            st.write("數據預覽:")
            st.dataframe(df.head())
            st.write("正在生成分析報告，請稍候...")
            csv_report = final_report(df)  # 生成分析報告
            st.session_state.messages.append({
                "role": "assistant",
                "content": csv_report
            })
            st.session_state.analysis_done = True
            st.success("分析報告已完成，現在可以進行對話！")
    
    except Exception as e:
        st.error(f"發生錯誤: {str(e)}")

if st.session_state.analysis_done:
    df = st.session_state.df
    # 建立對話輸入框
    user_input = st.chat_input("請輸入您的問題...")

    # 顯示歷史訊息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 處理用戶輸入
    if user_input:
        # 儲存用戶訊息
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # 顯示用戶訊息
        with st.chat_message("user"):
            st.markdown(user_input)


        # 定義系統提示詞
        system_prompt = """你是一個專業的數據分析助手，專門協助用戶分析CSV數據、生成視覺化圖表，並進行數據預測。請使用繁體中文回應。

主要功能：
- 數據分析與視覺化（折線圖、柱狀圖、散點圖、熱力圖等）
- 時間序列預測與趨勢分析
- 數據相關性分析
- 異常值檢測
- 數據分佈分析
- 圖表交叉比對分析

重要規則：
1. 所有回應必須提供完整的Python程式碼，包含所有必要的import語句
2. 不要將程式碼分段解釋，直接給出可以執行的完整代碼
3. 預測分析必須同時生成預測圖表，展示預測結果
4. 圖表必須包含完整的標題、軸標籤和圖例
5. 確保所有文字使用英文
6. 程式碼中必須包含適當的錯誤處理
7. 使用 'st.session_state.df' 存取數據框架
8. 使用 st.session_state.generated_charts 訪問已生成的圖表信息
- 每個圖表包含: {'number': 編號, 'path': 圖表路徑, 'timestamp': 生成時間}
9. 支持圖表的交叉比對和分析

圖表要求：
- 時間序列數據：同時顯示歷史數據和預測數據的折線圖
- 分類數據：使用柱狀圖或圓餅圖
- 相關性：使用散點圖或熱力圖
- 分佈分析：使用直方圖或箱型圖
- 圖表比對：支持多圖表並排顯示和比較分析

預測分析要求：
- 使用scikit-learn或statsmodels等庫進行預測
- 在圖表中用不同顏色區分實際數據和預測數據
- 顯示預測的置信區間
- 加入預測準確度指標（如RMSE、MAE等）

圖表交叉比對功能：
- 支持讀取已保存的圖表進行比較
- 可以進行多圖表並排分析
- 支持不同時間段的數據比較
- 可以比較不同預測模型的結果

程式碼格式：
```python
# 必要的import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.font_manager import FontProperties
import matplotlib.image as mpimg
# ... 其他需要的庫
# 使用matplotlib設置字型
font_path = "C:/Users/pear2/Desktop/python練習檔/projet/NotoSansCJKtc-VF.ttf"
font_prop = FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False  # 確保負號顯示正常

# 使用session state中的DataFrame和圖表信息
df = st.session_state.df

請注意：
- 確保代碼可以直接執行
- 包含所有必要的錯誤處理
- 使用繁體中文註解
- 圖表要美觀且信息完整
- 使用 st.session_state.df 訪問數據框架
- 使用 st.session_state.generated_charts 訪問已生成的圖表
- 支持圖表的交叉比對和分析
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Based on this DataFrame: {df.head().to_string()}\n\nQuestion: {user_input}"}
        ]

        # 處理完整回應
        full_response = ""
        
        # 創建助手的聊天消息容器
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # 使用串流模式獲取回應
            for chunk in client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3,
                stream=True,
            ):
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    # 只在完整句子結束時更新顯示
                    if full_response.endswith(("。", "？", "！", "\n")):
                        message_placeholder.markdown(full_response + "▌")
            
            # 最後一次更新，確保顯示完整內容
            message_placeholder.markdown(full_response)
        
        # 保存助手回應到歷史
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # 檢查是否包含 Python 代碼
        if "```python" in full_response:
            code_start = full_response.find("```python") + len("```python")
            code_end = full_response.find("```", code_start)
            code = full_response[code_start:code_end].strip()
            
            # 執行代碼並生成圖表
            success, error_msg, image_buffer = execute_code_and_generate_image(code)
            
            if success and image_buffer:
                # 增加計數器
                if 'chart_counter' not in st.session_state:
                    st.session_state.chart_counter = 0
                st.session_state.chart_counter += 1
                
                # 保存圖表到文件
                filepath = save_chart_to_file(image_buffer, st.session_state.chart_counter)
                
                # 將圖表加入到對話中
                add_chart_to_messages(filepath)
                
                # 顯示圖表
                st.image(filepath, caption=f"圖表 #{st.session_state.chart_counter}", use_column_width=True)
            else:
                st.error(f"生成圖表時發生錯誤：{error_msg}")
else:
    st.info("請上傳一個 CSV 文件以開始分析")

# 在側邊欄顯示歷史圖表
with st.sidebar:
    if st.session_state.generated_charts:
        st.subheader("歷史生成的圖表")
        for chart in st.session_state.generated_charts:
            if os.path.exists(chart['path']):
                with st.expander(f"圖表 #{chart['number']} - {chart['timestamp']}"):
                    st.image(chart['path'], use_column_width=True)
