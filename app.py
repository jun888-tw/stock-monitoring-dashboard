import google.generativeai as genai
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="AI 供應鏈與半導體核心持股儀表板",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ 持股估值與雷達")

# 1. 側邊欄設定與 API Key 安全讀取（相容本機與雲端環境）
api_key = ""

# 嘗試從 Streamlit Secrets 讀取（僅在雲端部署或設定 Secrets 時有效）
try:
  if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
  pass

# 若未能在 Secrets 取得 Key（如本機端），則顯示左側邊欄輸入框
if not api_key:
  st.sidebar.header("🔑 Gemini API 設定")
  api_key = st.sidebar.text_input(
      "輸入你的 Gemini API Key:", type="password"
  )
st.sidebar.header("🎯 關注標的管理")
DEFAULT_STOCKS = {
    # 晶圓代工與 IC 設計
    "台積電美股 (TSM)": "TSM",
    "台積電台股 (2330)": "2330.TW",
    "聯發科 (2454)": "2454.TW",
    # 散熱與機殼
    "健策 (3653)": "3653.TW",
    "奇鋐 (3017)": "3017.TW",
    "雙鴻 (3324)": "3324.TW",
    # 伺服器代工與組裝
    "廣達 (2382)": "2382.TW",
    "緯穎 (6669)": "6669.TW",
    "緯創 (3231)": "3231.TW",
    # 電源、UPS 與被動元件
    "台達電 (2308)": "2308.TW",
    "光寶科 (2301)": "2301.TW",
    "碩天 (3617)": "3617.TW",
    "國巨 (2327)": "2327.TW",
    # ABF 載板與 CCL 銅箔基板
    "台光電 (2383)": "2383.TW",
    "景碩 (3189)": "3189.TW",
    "欣興 (3037)": "3037.TW",
    "南電 (8046)": "8046.TW",
    # 廠務工程與水氣整合
    "漢唐 (2404)": "2404.TW",
    "亞翔 (6139)": "6139.TW",
    "兆聯實業 (6691)": "6691.TW",
    # 美股科技核心
    "輝達 (NVDA)": "NVDA",
    "Google (GOOGL)": "GOOGL",
    "Palantir (PLTR)": "PLTR",
    "AMD": "AMD",
    "亞馬遜 (AMZN)": "AMZN",
    "特斯拉 (TSLA)": "TSLA",
}

selected_labels = st.sidebar.multiselect(
    "選擇要監控的股票：",
    options=list(DEFAULT_STOCKS.keys()),
    default=list(DEFAULT_STOCKS.keys()),
)


# 2. 數據抓取與估值燈號判斷
@st.cache_data(ttl=1800)
def fetch_stock_data(ticker_code):
  try:
    data = yf.Ticker(ticker_code).info
    price = data.get("currentPrice") or data.get("regularMarketPrice") or "N/A"
    f_pe = data.get("forwardPE", "N/A")
    t_pe = data.get("trailingPE", "N/A")
    margin = data.get("grossMargins", "N/A")

    if margin != "N/A":
      margin = f"{float(margin) * 100:.1f}%"

    # 簡單估值燈號判斷
    status = "⚪ 數據不足"
    if isinstance(f_pe, (int, float)):
      f_pe = round(f_pe, 2)
      if f_pe < 18:
        status = "🟢 估值偏低 (加碼區)"
      elif 18 <= f_pe <= 30:
        status = "🟡 估值合理"
      else:
        status = "🔴 估值偏高"

    return {
        "現價": price,
        "Forward PE": f_pe,
        "Trailing PE": (
            round(t_pe, 2) if isinstance(t_pe, (int, float)) else t_pe
        ),
        "毛利率": margin,
        "估值狀態": status,
    }
  except Exception:
    return {
        "現價": "N/A",
        "Forward PE": "N/A",
        "Trailing PE": "N/A",
        "毛利率": "N/A",
        "估值狀態": "⚪ 異常",
    }


table_data = []
for label in selected_labels:
  ticker = DEFAULT_STOCKS[label]
  info = fetch_stock_data(ticker)
  info["標的名稱"] = label
  info["代號"] = ticker
  table_data.append(info)

if table_data:
  df = pd.DataFrame(table_data)
  df = df[[
      "標的名稱",
      "代號",
      "現價",
      "Forward PE",
      "Trailing PE",
      "毛利率",
      "估值狀態",
  ]]
  st.subheader("📊 核心持股估值與狀態總覽")
  st.dataframe(df, use_container_width=True)

st.divider()

# 3. AI 深度個股診斷模組
st.subheader("🤖 AI 個股雷達與財報健檢")

target_stock = st.selectbox("選擇要進行 AI 深度診斷的標的：", selected_labels)

col1, col2 = st.columns([1, 1])

with col1:
  news_text = st.text_area(
      "貼上近期新聞 / 法說會摘要 / 營收訊息：",
      height=150,
      placeholder="例如：台積電近期公佈月營收維持強勁成長，AI 晶片與 3 奈米需求旺盛...",
  )

with col2:
  uploaded_file = st.file_uploader(
      "或上傳財報 PDF (可選)：", type=["pdf"]
  )

if st.button("🚀 執行 Gemini 深度分析"):
  if not api_key:
    st.error("請先在左側邊欄輸入你的 Gemini API Key！")
  else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3.5-flash")

    with st.spinner(f"Gemini 正在分析 {target_stock} 中..."):
      prompt = f"""
            你是一位專業的半導體與 AI 產業資深投資分析師。
            請針對【{target_stock}】結合以下資訊進行深度健檢：
            1. 評估其 Forward PE 估值與當前產業地位。
            2. 解讀以下市場訊息或新聞：
            {news_text if news_text else '無額外新聞輸入'}
            
            請給出：
            - **基本面結論（綠燈/黃燈/紅燈）**
            - **核心優勢與加碼時機點**
            - **潛在風險或地雷警告（如存貨、CapEx 或客戶集中度）**
            """

      try:
        # 若有上傳 PDF 則包含進去
        inputs = [prompt]
        if uploaded_file is not None:
          pdf_data = uploaded_file.read()
          inputs.append({
              "mime_type": "application/pdf",
              "data": pdf_data,
          })

        response = model.generate_content(inputs)
        st.success("分析完成！")
        st.markdown(response.text)
      except Exception as e:
        st.error(f"分析失敗，錯誤訊息：{e}")
