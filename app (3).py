import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* ── Background ── */
  .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 50%, #0d1321 100%); }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131625 0%, #0e1420 100%);
    border-right: 1px solid #2a2d3e;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e2235 0%, #161929 100%);
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 16px;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
    color: #000;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,212,255,0.35);
  }

  /* ── Login card ── */
  .login-card {
    background: linear-gradient(135deg, #1e2235 0%, #161929 100%);
    border: 1px solid #2a2d3e;
    border-radius: 20px;
    padding: 40px;
    max-width: 420px;
    margin: 60px auto;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  }
  .login-title {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    color: #00d4ff;
    margin-bottom: 8px;
  }
  .login-sub { color: #8892a4; font-size: 14px; margin-bottom: 32px; }

  /* ── Section header ── */
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    color: #00d4ff;
    border-left: 4px solid #00d4ff;
    padding-left: 14px;
    margin: 28px 0 18px;
  }

  /* ── Tags / badges ── */
  .badge-green { background:#0d3b2e; color:#00e676; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:700; }
  .badge-red   { background:#3b0d0d; color:#ff5252; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:700; }

  /* ── Prediction box ── */
  .pred-box {
    background: linear-gradient(135deg, #0d2235 0%, #0a1a28 100%);
    border: 1px solid #00d4ff44;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
  }
  .pred-price { font-family:'Space Mono',monospace; font-size:48px; color:#00d4ff; font-weight:700; }
  .pred-label { color:#8892a4; font-size:13px; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE  (login)
# ─────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ─────────────────────────────────────────────
#  STOCK TICKERS LIST
# ─────────────────────────────────────────────
STOCK_LIST = {
    "🇺🇸 US Stocks": [
        "AAPL – Apple", "MSFT – Microsoft", "GOOGL – Alphabet",
        "AMZN – Amazon", "META – Meta", "TSLA – Tesla",
        "NVDA – NVIDIA", "NFLX – Netflix", "AMD – AMD",
        "INTC – Intel", "JPM – JPMorgan", "BAC – Bank of America",
        "WMT – Walmart", "DIS – Disney", "KO – Coca-Cola",
    ],
    "🇮🇳 Indian Stocks (NSE)": [
        "RELIANCE.NS – Reliance", "TCS.NS – TCS", "INFY.NS – Infosys",
        "HDFCBANK.NS – HDFC Bank", "ICICIBANK.NS – ICICI Bank",
        "WIPRO.NS – Wipro", "LT.NS – L&T", "SBIN.NS – SBI",
        "BAJFINANCE.NS – Bajaj Finance", "ADANIENT.NS – Adani Ent",
    ],
    "₿ Crypto": [
        "BTC-USD – Bitcoin", "ETH-USD – Ethereum",
        "BNB-USD – BNB", "SOL-USD – Solana", "ADA-USD – Cardano",
    ],
}

def ticker_from_label(label: str) -> str:
    return label.split(" – ")[0].strip()

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    df = df.copy()
    df["MA20"]  = close.rolling(20).mean()
    df["MA50"]  = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    rolling_mean = close.rolling(20).mean()
    rolling_std  = close.rolling(20).std()
    df["BB_Upper"] = rolling_mean + 2 * rolling_std
    df["BB_Lower"] = rolling_mean - 2 * rolling_std
    return df


def build_features(df: pd.DataFrame) -> tuple:
    df = add_indicators(df).dropna()
    close = df["Close"].squeeze()

    feature_cols = ["MA20", "MA50", "RSI", "MACD"]
    X = df[feature_cols].values
    y = close.values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled  = scaler_X.fit_transform(X)
    y_scaled  = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    return X_scaled, y_scaled, scaler_X, scaler_y, df, feature_cols


def train_and_predict(df: pd.DataFrame, model_choice: str, days_ahead: int):
    X_scaled, y_scaled, scaler_X, scaler_y, df_feat, feature_cols = build_features(df)

    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_test)

    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae  = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2   = r2_score(y_test_actual, y_pred_actual)

    # Future prediction
    last_row_scaled = X_scaled[-1].reshape(1, -1)
    future_pred_scaled = model.predict(last_row_scaled)
    future_price = scaler_y.inverse_transform(future_pred_scaled.reshape(-1, 1))[0][0]

    test_dates = df_feat.index[split:]
    return {
        "test_dates":  test_dates,
        "y_test":      y_test_actual,
        "y_pred":      y_pred_actual,
        "future_price": future_price,
        "mae": mae, "rmse": rmse, "r2": r2,
    }

# ─────────────────────────────────────────────
#  LOGIN PAGE
# ─────────────────────────────────────────────
def show_login():
    # Fake valid credentials (demo)
    USERS = {"admin": "admin123", "demo": "demo123", "analyst": "stock@2024"}

    st.markdown("""
    <div class="login-card">
      <div class="login-title">📈 Smart Stock Analyzer</div>
      <div class="login-sub">AI-Powered Market Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.container():
            st.markdown("#### 🔐 Sign In")
            username = st.text_input("Username", placeholder="admin / demo / analyst")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            if st.button("Login →", use_container_width=True):
                if username in USERS and USERS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username  = username
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

            st.markdown("""
            ---
            **Demo Credentials:**
            | Username | Password |
            |----------|----------|
            | admin    | admin123 |
            | demo     | demo123  |
            | analyst  | stock@2024 |
            """)

# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
def show_app():
    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown(f"### 👤 Welcome, **{st.session_state.username}**!")
        st.markdown("---")

        st.markdown("### 📌 Select Stock")
        selected_label = None
        for category, tickers in STOCK_LIST.items():
            with st.expander(category, expanded=(category == "🇺🇸 US Stocks")):
                for t in tickers:
                    if st.button(t, key=t, use_container_width=True):
                        st.session_state["selected_ticker"] = t
        
        # Manual input
        st.markdown("---")
        st.markdown("### ✍️ Custom Ticker")
        custom = st.text_input("Enter any ticker (e.g. GOOG)", key="custom_ticker")
        if custom:
            st.session_state["selected_ticker"] = custom.upper() + " – Custom"

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        period    = st.selectbox("Data Period", ["3mo","6mo","1y","2y","5y"], index=2)
        model_choice = st.selectbox("ML Model", ["Linear Regression", "Random Forest"])
        days_ahead   = st.slider("Predict Days Ahead", 1, 30, 7)

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    # ── HEADER ──
    st.markdown("""
    <h1 style='font-family:Space Mono,monospace;color:#00d4ff;margin-bottom:4px;'>
        📈 Smart Stock Analyzer
    </h1>
    <p style='color:#8892a4;margin-top:0;'>AI-powered stock analysis &amp; ML price prediction</p>
    """, unsafe_allow_html=True)

    ticker_label = st.session_state.get("selected_ticker", "AAPL – Apple")
    ticker = ticker_from_label(ticker_label)
    stock_name = ticker_label.split(" – ")[-1] if " – " in ticker_label else ticker

    st.markdown(f"### Analyzing: `{ticker}` — {stock_name}")

    # ── LOAD DATA ──
    with st.spinner(f"Fetching data for {ticker}..."):
        df = load_data(ticker, period)

    if df.empty:
        st.error("⚠️ Could not load data. Check your ticker symbol or internet connection.")
        return

    df_ind = add_indicators(df)
    close  = df["Close"].squeeze()

    # ── KPI METRICS ──
    latest   = float(close.iloc[-1])
    prev     = float(close.iloc[-2])
    change   = latest - prev
    pct_chg  = change / prev * 100
    hi52     = float(close.rolling(252).max().iloc[-1])
    lo52     = float(close.rolling(252).min().iloc[-1])
    avg_vol  = int(df["Volume"].squeeze().mean())

    badge = "badge-green" if change >= 0 else "badge-red"
    sign  = "+" if change >= 0 else ""

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Current Price",  f"${latest:.2f}",  f"{sign}{change:.2f} ({sign}{pct_chg:.2f}%)")
    c2.metric("📈 52-Week High",   f"${hi52:.2f}")
    c3.metric("📉 52-Week Low",    f"${lo52:.2f}")
    c4.metric("📊 Avg Volume",     f"{avg_vol:,}")

    # ── TABS ──
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Charts", "📉 Indicators", "🤖 ML Prediction", "📋 Raw Data"])

    # ──────────────────────────────────────────
    #  TAB 1 – PRICE CHARTS
    # ──────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Candlestick & Moving Averages</div>', unsafe_allow_html=True)

        fig = go.Figure()

        # Candlestick
        open_  = df["Open"].squeeze()
        high_  = df["High"].squeeze()
        low_   = df["Low"].squeeze()
        close_ = df["Close"].squeeze()

        fig.add_trace(go.Candlestick(
            x=df.index, open=open_, high=high_, low=low_, close=close_,
            name="OHLC",
            increasing_line_color="#00e676", decreasing_line_color="#ff5252",
        ))

        for col, color, label in [
            ("MA20",  "#f9a825", "MA 20"),
            ("MA50",  "#ab47bc", "MA 50"),
            ("MA200", "#ef5350", "MA 200"),
        ]:
            if col in df_ind.columns:
                fig.add_trace(go.Scatter(
                    x=df_ind.index, y=df_ind[col].squeeze(),
                    mode="lines", name=label,
                    line=dict(color=color, width=1.5),
                ))

        # Bollinger Bands
        if "BB_Upper" in df_ind.columns:
            fig.add_trace(go.Scatter(
                x=df_ind.index, y=df_ind["BB_Upper"].squeeze(),
                mode="lines", name="BB Upper",
                line=dict(color="#00bcd4", dash="dot", width=1),
            ))
            fig.add_trace(go.Scatter(
                x=df_ind.index, y=df_ind["BB_Lower"].squeeze(),
                mode="lines", name="BB Lower",
                line=dict(color="#00bcd4", dash="dot", width=1),
                fill="tonexty", fillcolor="rgba(0,188,212,0.05)",
            ))

        fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9",
            xaxis=dict(showgrid=False, rangeslider_visible=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2235"),
            legend=dict(bgcolor="#0d1117"),
            height=520,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume chart
        st.markdown('<div class="section-header">Volume</div>', unsafe_allow_html=True)
        vol = df["Volume"].squeeze()
        colors = ["#00e676" if c >= p else "#ff5252"
                  for c, p in zip(close_.values[1:], close_.values[:-1])]
        colors = ["#00e676"] + colors

        vol_fig = go.Figure(go.Bar(
            x=df.index, y=vol,
            marker_color=colors,
            name="Volume",
        ))
        vol_fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2235"),
            height=220,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(vol_fig, use_container_width=True)

    # ──────────────────────────────────────────
    #  TAB 2 – INDICATORS
    # ──────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">RSI (Relative Strength Index)</div>', unsafe_allow_html=True)
        rsi_series = df_ind["RSI"].squeeze().dropna()

        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=rsi_series.index, y=rsi_series,
            mode="lines", name="RSI",
            line=dict(color="#f9a825", width=2),
            fill="tozeroy", fillcolor="rgba(249,168,37,0.07)",
        ))
        rsi_fig.add_hline(y=70, line_color="#ff5252", line_dash="dash", annotation_text="Overbought (70)")
        rsi_fig.add_hline(y=30, line_color="#00e676", line_dash="dash", annotation_text="Oversold (30)")
        rsi_fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9",
            yaxis=dict(range=[0,100], showgrid=True, gridcolor="#1e2235"),
            xaxis=dict(showgrid=False),
            height=280, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(rsi_fig, use_container_width=True)

        current_rsi = float(rsi_series.iloc[-1])
        if current_rsi > 70:
            st.warning(f"⚠️ RSI = {current_rsi:.1f} — **Overbought**. Price may pullback soon.")
        elif current_rsi < 30:
            st.success(f"✅ RSI = {current_rsi:.1f} — **Oversold**. Potential buying opportunity.")
        else:
            st.info(f"ℹ️ RSI = {current_rsi:.1f} — **Neutral zone**.")

        st.markdown('<div class="section-header">MACD</div>', unsafe_allow_html=True)
        macd_s  = df_ind["MACD"].squeeze().dropna()
        signal_s = df_ind["MACD_Signal"].squeeze().dropna()

        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=macd_s.index, y=macd_s, mode="lines",
                                      name="MACD", line=dict(color="#00d4ff", width=2)))
        macd_fig.add_trace(go.Scatter(x=signal_s.index, y=signal_s, mode="lines",
                                      name="Signal", line=dict(color="#f9a825", width=1.5)))
        hist = macd_s - signal_s
        macd_fig.add_trace(go.Bar(x=hist.index, y=hist,
                                  name="Histogram",
                                  marker_color=np.where(hist >= 0, "#00e676", "#ff5252")))
        macd_fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2235"),
            height=280, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(macd_fig, use_container_width=True)

        # Price distribution
        st.markdown('<div class="section-header">Price Distribution</div>', unsafe_allow_html=True)
        hist_fig = px.histogram(
            x=close_.values, nbins=50,
            color_discrete_sequence=["#00d4ff"],
            labels={"x": "Price"},
        )
        hist_fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2235"),
            height=260, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    # ──────────────────────────────────────────
    #  TAB 3 – ML PREDICTION
    # ──────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Machine Learning Price Prediction</div>', unsafe_allow_html=True)

        st.info(f"**Model:** {model_choice} | **Training data:** 80% | **Test data:** 20% | **Predicting:** {days_ahead} day(s) ahead")

        with st.spinner("Training model..."):
            results = train_and_predict(df, model_choice, days_ahead)

        # Prediction box
        future_price = results["future_price"]
        direction    = "🚀 Bullish" if future_price > latest else "🔻 Bearish"
        diff_pct     = (future_price - latest) / latest * 100

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("🎯 Predicted Price",  f"${future_price:.2f}",  f"{diff_pct:+.2f}%")
        col_b.metric("📐 MAE (Error)",      f"${results['mae']:.2f}")
        col_c.metric("📏 R² Score",         f"{results['r2']:.4f}")

        # Actual vs Predicted chart
        st.markdown('<div class="section-header">Actual vs Predicted (Test Set)</div>', unsafe_allow_html=True)
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(
            x=results["test_dates"], y=results["y_test"],
            mode="lines", name="Actual",
            line=dict(color="#00e676", width=2),
        ))
        pred_fig.add_trace(go.Scatter(
            x=results["test_dates"], y=results["y_pred"],
            mode="lines", name="Predicted",
            line=dict(color="#f9a825", width=2, dash="dash"),
        ))
        pred_fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2235"),
            height=400, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(pred_fig, use_container_width=True)

        # Future projection
        st.markdown('<div class="section-header">Future Price Projection</div>', unsafe_allow_html=True)
        future_dates  = [df.index[-1] + timedelta(days=i+1) for i in range(days_ahead)]
        future_prices = np.linspace(latest, future_price, days_ahead)

        fut_fig = go.Figure()
        fut_fig.add_trace(go.Scatter(
            x=close_.index[-60:], y=close_.values[-60:],
            mode="lines", name="Historical",
            line=dict(color="#00d4ff", width=2),
        ))
        fut_fig.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            mode="lines+markers", name="Forecast",
            line=dict(color="#f9a825", width=2.5, dash="dot"),
            marker=dict(color="#f9a825", size=6),
        ))
        fut_fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2235"),
            height=360, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fut_fig, use_container_width=True)

        # Model explanation
        with st.expander("ℹ️ How does this ML model work?"):
            st.markdown(f"""
**Features used for prediction:**
- `MA20` — 20-day moving average (short-term trend)
- `MA50` — 50-day moving average (medium-term trend)
- `RSI` — Momentum oscillator (overbought/oversold)
- `MACD` — Trend-following momentum indicator

**Model: {model_choice}**
{"- Fits a straight line through the feature space to predict price" if model_choice == "Linear Regression" else "- Builds 100 decision trees and averages their predictions for robustness"}

**Important Disclaimer:** This is for educational purposes only. Stock prices are influenced by many factors not captured here. Never make financial decisions solely based on ML predictions.
            """)

    # ──────────────────────────────────────────
    #  TAB 4 – RAW DATA
    # ──────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Historical Data</div>', unsafe_allow_html=True)
        display_df = df.copy()
        # Flatten MultiIndex columns if present
        if isinstance(display_df.columns, pd.MultiIndex):
            display_df.columns = [col[0] for col in display_df.columns]
        display_df = display_df.round(2)
        display_df.index = display_df.index.strftime("%Y-%m-%d")
        st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)

        # Download button
        csv = display_df.to_csv().encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"{ticker}_data.csv",
            mime="text/csv",
        )

        # Summary statistics
        st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
        st.dataframe(display_df.describe().round(2), use_container_width=True)

# ─────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────
if st.session_state.logged_in:
    show_app()
else:
    show_login()
