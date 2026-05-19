"""
streamlit_app.py  –  Complete Stock Market Platform Dashboard
FIXED: charts now display correctly with proper scaling, area fill, candlestick aggregation, and live-data cache.
"""
import os, json, random
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pymongo import MongoClient

# ── Try importing predict_price at module level (PYTHONPATH=/app in Docker) ──
try:
    from app.ml.predict import predict_price as _ml_predict
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0e1a; }
  [data-testid="stSidebar"] { background: #0f1526; }
  .metric-card {
    background: #141d35; border: 1px solid #1e2d50;
    border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
  }
  .price-up   { color: #00e676; font-weight: 700; }
  .price-down { color: #ff5252; font-weight: 700; }
  .section-header {
    font-size: 1.1rem; font-weight: 600; color: #90caf9;
    border-bottom: 1px solid #1e2d50; padding-bottom: 6px; margin-bottom: 14px;
  }
  div[data-testid="metric-container"] > div { background: #141d35 !important; border-radius: 10px; padding: 10px; }
  .stTabs [data-baseweb="tab"] { background: #141d35; color: #90caf9; }
  .stTabs [aria-selected="true"] { background: #1565c0 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

STOCKS      = ["AAPL", "TSLA", "MSFT", "NVDA", "META", "AMZN"]
BASE_PRICES = {"AAPL":175.5,"TSLA":245.8,"MSFT":380.2,"NVDA":485.6,"META":312.4,"AMZN":145.9}
MONGO_URI   = os.getenv("MONGO_URI",        "mongodb://localhost:27017")
MONGO_DB    = os.getenv("MONGO_DB",         "stocks_db")
MONGO_COL   = os.getenv("MONGO_COLLECTION", "live_prices")

# ── Plotly dark theme defaults ────────────────────────────────
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0a0e1a",
    font=dict(color="#c9d1d9", size=12),
    margin=dict(l=60, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)

# ── Data helpers ─────────────────────────────────────────────
# FIX: use @st.cache_resource but bust it on every rerun so live data flows through.
#      The sidebar "Refresh now" button calls st.cache_resource.clear() explicitly.
@st.cache_resource
def _mongo_client():
    try:
        c = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        c.admin.command("ping")
        return c
    except Exception:
        return None

def get_mongo():
    return _mongo_client()

def load_live_data(limit=500) -> pd.DataFrame:
    client = get_mongo()
    if client is not None:
        try:
            col  = client[MONGO_DB][MONGO_COL]
            docs = list(col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
            if docs:
                df = pd.DataFrame(docs)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
                if not df.empty:
                    return df
        except Exception:
            pass
    return _mock_data()

def _mock_data() -> pd.DataFrame:
    rows = []
    now  = datetime.utcnow()
    for symbol in STOCKS:
        price = BASE_PRICES[symbol]
        for i in range(120):
            ts    = now - timedelta(minutes=120 - i)
            price *= 1 + random.gauss(0, 0.003)
            op = round(price * random.uniform(0.998, 1.002), 2)
            hi = round(price * random.uniform(1.000, 1.015), 2)
            lo = round(price * random.uniform(0.985, 1.000), 2)
            cl = round(price, 2)
            rows.append({"symbol": symbol, "timestamp": ts,
                         "price": cl, "open": op, "high": hi,
                         "low": lo, "close": cl,
                         "volume": random.randint(500_000, 5_000_000)})
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")

def load_ml_metrics() -> dict:
    for path in ["data/model_metrics.json", "/app/data/model_metrics.json"]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}

def run_prediction(symbol, open_p, high, low, volume, model_type="random_forest") -> float:
    if ML_AVAILABLE:
        try:
            return _ml_predict(symbol, open_p, high, low, volume, model_type)
        except Exception:
            pass
    return round(BASE_PRICES.get(symbol, 100) * random.uniform(0.97, 1.03), 2)


# ── FIX helper: aggregate raw tick data into OHLCV bars ───────
def aggregate_ohlcv(sdf: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """
    Resample a per-tick dataframe into OHLC candles.
    Expects columns: timestamp, price, open, high, low, close, volume.
    freq examples: '1min', '5min', '15min'
    """
    sdf = sdf.set_index("timestamp").sort_index()
    agg = sdf["price"].resample(freq).ohlc()          # open/high/low/close from price
    agg["volume"] = sdf["volume"].resample(freq).sum()
    agg = agg.dropna(subset=["open"])
    agg = agg.reset_index()
    agg.rename(columns={"timestamp": "ts"}, inplace=True)
    return agg


# ── FIX helper: normalise a series to % change from first point ──
def pct_change_series(series: pd.Series) -> pd.Series:
    first = series.iloc[0]
    if first == 0:
        return series * 0
    return (series - first) / first * 100


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Stock Platform")
    st.markdown("---")
    selected_stocks  = st.multiselect("Stocks to display", STOCKS, default=STOCKS)
    chart_type       = st.radio("Chart type", ["Line", "Candlestick", "Area"], index=0)
    candle_freq      = st.select_slider(
        "Candle interval",
        options=["1min", "5min", "15min", "30min"],
        value="5min",
    )
    auto_refresh     = st.toggle("Auto-refresh", value=True)
    refresh_interval = st.selectbox(
        "Refresh interval",
        options=[30, 60, 120, 300],
        format_func=lambda x: {30:"30 seconds",60:"1 minute",120:"2 minutes",300:"5 minutes"}[x],
        index=1, disabled=not auto_refresh,
    )
    st.markdown("---")
    st.markdown("**Data source**")
    st.caption("MongoDB live feed with mock fallback")
    if not ML_AVAILABLE:
        st.warning("ML models not loaded — using mock predictions.")
    st.markdown("---")
    if st.button("Refresh now"):
        st.cache_resource.clear()
        st.rerun()

# ── Load data ─────────────────────────────────────────────────
df          = load_live_data(limit=1000)
df_filtered = df[df["symbol"].isin(selected_stocks)] if selected_stocks else df

# ── Header ────────────────────────────────────────────────────
st.markdown("# Stock Market Platform")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ── KPI metric cards ─────────────────────────────────────────
latest = df_filtered.groupby("symbol").last().reset_index()
cols   = st.columns(max(len(latest), 1))
for i, row in latest.iterrows():
    sym  = row["symbol"]
    hist = df_filtered[df_filtered["symbol"] == sym]["price"]
    prev = hist.iloc[0] if len(hist) > 1 else row["price"]
    chg  = ((row["price"] - prev) / prev * 100) if prev else 0
    arrow = "↑" if chg >= 0 else "↓"
    with cols[i % len(cols)]:
        st.metric(label=f"**{sym}**", value=f"${row['price']:.2f}",
                  delta=f"{arrow} {abs(chg):.2f}%")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Prices", "Candlestick", "ML Predictions", "Analytics", "Spike Alerts"
])

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 – Live Prices                                        ║
# ╚══════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<p class="section-header">Real-Time Price Streams</p>', unsafe_allow_html=True)
    if df_filtered.empty:
        st.warning("No data available yet.")
    else:
        # ── Candlestick mode in tab1 ──────────────────────────
        if chart_type == "Candlestick":
            csym = st.selectbox("Select stock", selected_stocks or STOCKS, key="candle_t1")
            cdf  = df_filtered[df_filtered["symbol"] == csym].tail(200).copy()

            if cdf.empty or not all(c in cdf.columns for c in ["open","high","low","close"]):
                st.warning("Not enough OHLC data.")
            else:
                # FIX: aggregate raw ticks into proper candle bars
                bars = aggregate_ohlcv(cdf, candle_freq)

                fig_t1 = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03, row_heights=[0.75, 0.25],
                )
                fig_t1.add_trace(go.Candlestick(
                    x=bars["ts"], open=bars["open"], high=bars["high"],
                    low=bars["low"], close=bars["close"], name=csym,
                    increasing_line_color="#00e676", increasing_fillcolor="#00e676",
                    decreasing_line_color="#ff5252", decreasing_fillcolor="#ff5252",
                    whiskerwidth=0.4, line_width=1,
                ), row=1, col=1)

                bar_colors = ["#00e676" if c >= o else "#ff5252"
                              for c, o in zip(bars["close"], bars["open"])]
                fig_t1.add_trace(go.Bar(
                    x=bars["ts"], y=bars["volume"], name="Volume",
                    marker_color=bar_colors, opacity=0.6,
                ), row=2, col=1)

                fig_t1.update_layout(
                    **DARK_LAYOUT,
                    height=540,
                    xaxis_rangeslider_visible=False,
                    title=dict(text=f"{csym} · {candle_freq} candles", x=0.01, font=dict(size=14, color="#90caf9")),
                )
                # FIX: format volume axis with K/M suffix so numbers don't overflow
                fig_t1.update_yaxes(
                    title_text="Price (USD)", row=1, col=1,
                    tickprefix="$", showgrid=True, gridcolor="#1e2d50",
                )
                fig_t1.update_yaxes(
                    title_text="Volume", row=2, col=1,
                    tickformat=".2s",               # e.g. 2.5M instead of 2500000
                    showgrid=True, gridcolor="#1e2d50",
                )
                fig_t1.update_xaxes(showgrid=False)
                st.plotly_chart(fig_t1, use_container_width=True)

        # ── Line / Area mode in tab1 ──────────────────────────
        else:
            # ── Time-range selector ───────────────────────────
            tr_col, mode_col = st.columns([3, 1])
            with tr_col:
                time_range = st.radio(
                    "Time range",
                    ["15m", "30m", "1h", "2h", "6h", "All"],
                    index=2,
                    horizontal=True,
                    label_visibility="collapsed",
                )
            with mode_col:
                view_mode = st.radio(
                    "View",
                    ["% Change", "Price"],
                    index=0,
                    horizontal=True,
                    label_visibility="collapsed",
                )

            # Filter data to selected time window
            now_utc = pd.Timestamp.utcnow()
            range_map = {
                "15m": timedelta(minutes=15),
                "30m": timedelta(minutes=30),
                "1h":  timedelta(hours=1),
                "2h":  timedelta(hours=2),
                "6h":  timedelta(hours=6),
                "All": None,
            }
            delta = range_map[time_range]
            if delta is not None:
                cutoff = now_utc - delta
                df_window = df_filtered[df_filtered["timestamp"] >= cutoff]
            else:
                df_window = df_filtered

            if df_window.empty:
                st.info(f"No data in the last {time_range} window. Try a wider range or wait for more ticks.")
                df_window = df_filtered   # fall back to all data

            use_pct = (view_mode == "% Change")
            colors  = px.colors.qualitative.Plotly

            # ── Build chart ───────────────────────────────────
            # When showing raw prices with very different scales, use two Y-axes:
            # left axis for high-price stocks (MSFT, NVDA), right for lower ones.
            HIGH_PRICE_THRESHOLD = 300  # stocks above this go on the left axis

            fig = go.Figure()

            for idx, sym in enumerate(selected_stocks):
                sdf = df_window[df_window["symbol"] == sym].copy()
                if sdf.empty:
                    continue

                color      = colors[idx % len(colors)]
                last_price = sdf["price"].iloc[-1]
                first_price = sdf["price"].iloc[0]
                chg_pct    = ((last_price - first_price) / first_price * 100) if first_price else 0
                chg_sign   = "▲" if chg_pct >= 0 else "▼"
                label      = f"{sym}  {chg_sign}{abs(chg_pct):.2f}%  ${last_price:.2f}"

                if use_pct:
                    y_vals = pct_change_series(sdf["price"])
                    yaxis  = "y"
                else:
                    y_vals = sdf["price"]
                    # Separate Y axes only in raw Price mode
                    yaxis  = "y" if last_price >= HIGH_PRICE_THRESHOLD else "y2"

                common = dict(
                    x=sdf["timestamp"], y=y_vals, name=label,
                    line=dict(width=2, color=color),
                    yaxis=yaxis,
                    hovertemplate=f"<b>{sym}</b><br>%{{x|%H:%M:%S}}<br>" +
                                  ("Δ %{y:.2f}%" if use_pct else "$%{y:.2f}") +
                                  "<extra></extra>",
                )

                if chart_type == "Area":
                    # Semi-transparent fill only below its own line (tonexty stacks weirdly)
                    r, g, b = (99,110,250) if idx==0 else (239,85,59) if idx==1 else \
                              (0,204,150) if idx==2 else (171,99,250) if idx==3 else \
                              (255,161,90) if idx==4 else (25,211,243)
                    fig.add_trace(go.Scatter(
                        **common,
                        mode="lines",
                        fill="tozeroy",
                        fillcolor=f"rgba({r},{g},{b},0.06)",
                    ))
                else:
                    fig.add_trace(go.Scatter(**common, mode="lines"))

            # X-axis: clip to actual data range (no dead whitespace)
            x_min = df_window["timestamp"].min()
            x_max = df_window["timestamp"].max()

            layout_extra = {}
            if not use_pct:
                layout_extra = dict(
                    yaxis2=dict(
                        title="Price — lower-priced stocks (USD)",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        tickprefix="$",
                        tickfont=dict(color="#aaa"),
                    )
                )

            fig.update_layout(
                **DARK_LAYOUT,
                height=480,
                hovermode="x unified",
                xaxis=dict(
                    title="Time",
                    range=[x_min, x_max],          # no dead whitespace on the right
                    showgrid=False,
                    rangeslider=dict(visible=False),
                    tickformat="%H:%M\n%b %d",
                ),
                yaxis=dict(
                    title="% Change" if use_pct else "Price — high-priced stocks (USD)",
                    showgrid=True,
                    gridcolor="#1e2d50",
                    ticksuffix="%" if use_pct else "",
                    tickprefix="" if use_pct else "$",
                    zeroline=use_pct,
                    zerolinecolor="#ffffff30",
                    zerolinewidth=1,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="left",   x=0,
                    font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)",
                ),
                **layout_extra,
            )
            st.plotly_chart(fig, use_container_width=True)

            if use_pct:
                st.caption("% change normalises all stocks to the same baseline so different price levels are directly comparable.")
            else:
                st.caption("Stocks ≥ $300 (MSFT, NVDA) use the left axis. Lower-priced stocks use the right axis.")

    # ── Latest prices table + volume bar ─────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-header">Latest Prices Table</p>', unsafe_allow_html=True)
        disp = latest[["symbol","price","open","high","low","close","volume"]].copy()
        disp.columns = ["Symbol","Price","Open","High","Low","Close","Volume"]
        st.dataframe(
            disp.style.format({"Price":"${:.2f}","Open":"${:.2f}","High":"${:.2f}",
                               "Low":"${:.2f}","Close":"${:.2f}","Volume":"{:,.0f}"}),
            use_container_width=True,
        )
    with c2:
        st.markdown('<p class="section-header">Volume Distribution</p>', unsafe_allow_html=True)
        fig_v = px.bar(
            latest[["symbol","volume"]], x="symbol", y="volume",
            template="plotly_dark", color="symbol",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            labels={"volume":"Volume","symbol":"Symbol"},
        )
        # FIX: format Y-axis so "5000000" becomes "5M"
        fig_v.update_yaxes(tickformat=".2s")
        fig_v.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
            showlegend=False, margin=dict(l=50, r=10, t=10, b=30), height=280,
        )
        st.plotly_chart(fig_v, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 – Candlestick (dedicated, full width)               ║
# ╚══════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<p class="section-header">Candlestick Charts</p>', unsafe_allow_html=True)

    col_sym, col_freq = st.columns([2, 1])
    with col_sym:
        candle_sym = st.selectbox("Select stock", selected_stocks or STOCKS, key="candle_sym")
    with col_freq:
        freq2 = st.select_slider("Interval", options=["1min","5min","15min","30min"],
                                 value=candle_freq, key="freq2")

    candle_df = df_filtered[df_filtered["symbol"] == candle_sym].tail(300).copy()

    if candle_df.empty or not all(c in candle_df.columns for c in ["open","high","low","close"]):
        st.warning("Not enough OHLC data for candlestick chart.")
    else:
        # FIX: aggregate raw ticks into candles
        bars2 = aggregate_ohlcv(candle_df, freq2)

        if bars2.empty or len(bars2) < 2:
            st.warning("Not enough data points for the selected interval. Try a shorter interval.")
        else:
            # Moving averages on aggregated bars
            bars2["ma10"] = bars2["close"].rolling(10, min_periods=1).mean()
            bars2["ma20"] = bars2["close"].rolling(20, min_periods=1).mean()

            fig_c = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.03, row_heights=[0.75, 0.25],
            )
            fig_c.add_trace(go.Candlestick(
                x=bars2["ts"], open=bars2["open"], high=bars2["high"],
                low=bars2["low"], close=bars2["close"], name=candle_sym,
                increasing_line_color="#00e676", increasing_fillcolor="#00e676",
                decreasing_line_color="#ff5252", decreasing_fillcolor="#ff5252",
                whiskerwidth=0.4, line_width=1,
            ), row=1, col=1)

            fig_c.add_trace(go.Scatter(
                x=bars2["ts"], y=bars2["ma10"], name="MA10",
                line=dict(color="#ffd54f", width=1.5, dash="dot"),
            ), row=1, col=1)
            fig_c.add_trace(go.Scatter(
                x=bars2["ts"], y=bars2["ma20"], name="MA20",
                line=dict(color="#ce93d8", width=1.5, dash="dot"),
            ), row=1, col=1)

            bar_colors2 = ["#00e676" if c >= o else "#ff5252"
                           for c, o in zip(bars2["close"], bars2["open"])]
            fig_c.add_trace(go.Bar(
                x=bars2["ts"], y=bars2["volume"], name="Volume",
                marker_color=bar_colors2, opacity=0.6,
            ), row=2, col=1)

            fig_c.update_layout(
                **DARK_LAYOUT,
                height=600,
                xaxis_rangeslider_visible=False,
                title=dict(text=f"{candle_sym} · {freq2} candles  |  {len(bars2)} bars",
                           x=0.01, font=dict(size=14, color="#90caf9")),
            )
            fig_c.update_yaxes(title_text="Price (USD)", row=1, col=1,
                               tickprefix="$", showgrid=True, gridcolor="#1e2d50")
            fig_c.update_yaxes(title_text="Volume", row=2, col=1,
                               tickformat=".2s", showgrid=True, gridcolor="#1e2d50")
            fig_c.update_xaxes(showgrid=False)
            st.plotly_chart(fig_c, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 3 – ML Predictions                                     ║
# ╚══════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<p class="section-header">Machine Learning Predictions</p>', unsafe_allow_html=True)
    metrics = load_ml_metrics()

    if metrics:
        st.markdown("**Model Performance (MAE & R²)**")
        met_rows = []
        for sym, m in metrics.items():
            for mname, vals in m.items():
                if isinstance(vals, dict) and "mae" in vals:
                    met_rows.append({"Symbol":sym,"Model":mname,"MAE":vals["mae"],"R²":vals["r2"]})
        met_df = pd.DataFrame(met_rows)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_mae = px.bar(met_df, x="Symbol", y="MAE", color="Model", barmode="group",
                             template="plotly_dark", title="Mean Absolute Error (lower = better)")
            fig_mae.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                                  margin=dict(l=50,r=10,t=40,b=30), height=300)
            st.plotly_chart(fig_mae, use_container_width=True)
        with col_m2:
            fig_r2 = px.bar(met_df, x="Symbol", y="R²", color="Model", barmode="group",
                            template="plotly_dark", title="R² Score (higher = better)")
            fig_r2.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                                 margin=dict(l=50,r=10,t=40,b=30), height=300)
            st.plotly_chart(fig_r2, use_container_width=True)
    else:
        st.info("Model metrics not found. Models will be trained automatically on first run.")

    st.markdown("---")
    st.markdown("**Run a Prediction**")
    pred_col1, pred_col2 = st.columns(2)
    with pred_col1:
        pred_sym   = st.selectbox("Symbol", STOCKS, key="pred_sym")
        pred_model = st.radio("Model", ["random_forest", "linear_regression"], horizontal=True)
        last_row   = df[df["symbol"]==pred_sym].tail(1)
        def_open = float(last_row["open"].iloc[0])   if not last_row.empty and "open"   in last_row.columns else BASE_PRICES.get(pred_sym, 100)
        def_high = float(last_row["high"].iloc[0])   if not last_row.empty and "high"   in last_row.columns else def_open * 1.02
        def_low  = float(last_row["low"].iloc[0])    if not last_row.empty and "low"    in last_row.columns else def_open * 0.98
        def_vol  = int(last_row["volume"].iloc[0])   if not last_row.empty and "volume" in last_row.columns else 2_000_000

        open_p = st.number_input("Open price", value=def_open, step=0.5)
        high_p = st.number_input("High",       value=def_high, step=0.5)
        low_p  = st.number_input("Low",        value=def_low,  step=0.5)
        vol_p  = st.number_input("Volume",     value=def_vol,  step=100_000)

        if st.button("Predict Close Price", use_container_width=True):
            pred_val = run_prediction(pred_sym, open_p, high_p, low_p, int(vol_p), pred_model)
            st.session_state["last_pred"] = {"symbol":pred_sym,"predicted":pred_val,"model":pred_model}

    with pred_col2:
        if "last_pred" in st.session_state:
            p = st.session_state["last_pred"]
            st.markdown(f"### Prediction for **{p['symbol']}**")
            st.markdown(f"<h1 class='price-up'>${p['predicted']:.2f}</h1>", unsafe_allow_html=True)
            st.caption(f"Model: {p['model'].replace('_',' ').title()}")
            actual_row = df[df["symbol"]==p["symbol"]].tail(1)
            if not actual_row.empty:
                actual = float(actual_row["price"].iloc[0])
                diff   = p["predicted"] - actual
                st.metric("vs current market price", f"${actual:.2f}", delta=f"${diff:+.2f}")

        st.markdown("**All symbols – latest predictions**")
        pred_rows = []
        for sym in STOCKS:
            lr = df[df["symbol"] == sym].tail(1)
            if lr.empty:
                pred_rows.append({"Symbol":sym,"Current":"—","Predicted Close":"—","Delta":"—"})
                continue
            o  = float(lr["open"].iloc[0])   if "open"   in lr.columns else BASE_PRICES[sym]
            h  = float(lr["high"].iloc[0])   if "high"   in lr.columns else o * 1.02
            l  = float(lr["low"].iloc[0])    if "low"    in lr.columns else o * 0.98
            v  = int(lr["volume"].iloc[0])   if "volume" in lr.columns else 1_000_000
            pv      = run_prediction(sym, o, h, l, v)
            current = float(lr["price"].iloc[0])
            pred_rows.append({
                "Symbol":          sym,
                "Current":         f"${current:.2f}",
                "Predicted Close": f"${pv:.2f}",
                "Delta":           f"${pv - current:+.2f}",
            })
        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 4 – Analytics                                          ║
# ╚══════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown('<p class="section-header">Historical Analytics</p>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Price trend — % change (all stocks)**")
        # FIX: normalise each stock to % change so they're all visible on one axis
        fig_t = go.Figure()
        colors = px.colors.qualitative.Plotly
        for idx, sym in enumerate(selected_stocks or STOCKS):
            sdf = df_filtered[df_filtered["symbol"] == sym].copy()
            if sdf.empty:
                continue
            y_pct = pct_change_series(sdf["price"])
            fig_t.add_trace(go.Scatter(
                x=sdf["timestamp"], y=y_pct, name=sym,
                mode="lines", line=dict(width=1.5, color=colors[idx % len(colors)]),
            ))
        fig_t.update_layout(
            **DARK_LAYOUT, height=320,
            yaxis_title="% Change", xaxis_title="Time",
            yaxis=dict(ticksuffix="%", showgrid=True, gridcolor="#1e2d50"),
            xaxis=dict(showgrid=False),
            hovermode="x unified",
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with a2:
        st.markdown("**Average daily gain (close − open)**")
        gain_df = (
            df_filtered.groupby("symbol")
            .apply(lambda g: pd.Series({"avg_gain":(g["close"]-g["open"]).mean()}), include_groups=False)
            .reset_index()
        )
        gc = ["#00e676" if v >= 0 else "#ff5252" for v in gain_df["avg_gain"]]
        fig_g = go.Figure(go.Bar(
            x=gain_df["symbol"], y=gain_df["avg_gain"],
            marker_color=gc, text=gain_df["avg_gain"].round(3),
            texttemplate="%{text:+.3f}", textposition="outside",
        ))
        fig_g.update_layout(
            **DARK_LAYOUT, height=320,
            yaxis=dict(title="Avg Gain (USD)", tickprefix="$", showgrid=True, gridcolor="#1e2d50"),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown('<p class="section-header">Volatility (Price Std Dev per Symbol)</p>', unsafe_allow_html=True)
    vol_stats = df_filtered.groupby("symbol")["price"].std().reset_index()
    vol_stats.columns = ["symbol","volatility"]
    fig_vol = px.bar(
        vol_stats, x="symbol", y="volatility", template="plotly_dark",
        color="volatility", color_continuous_scale="Reds",
        labels={"volatility":"Std Dev (USD)"},
    )
    fig_vol.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        height=280, margin=dict(l=60,r=10,t=10,b=30),
    )
    # FIX: add $ prefix on Y axis
    fig_vol.update_yaxes(tickprefix="$", showgrid=True, gridcolor="#1e2d50")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown('<p class="section-header">Price Correlation Heatmap</p>', unsafe_allow_html=True)
    pivot = df_filtered.pivot_table(index="timestamp", columns="symbol", values="price")
    corr  = pivot.corr()
    fig_h = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
    ))
    fig_h.update_layout(
        **DARK_LAYOUT, height=340,
    )
    st.plotly_chart(fig_h, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 5 – Spike Alerts                                       ║
# ╚══════════════════════════════════════════════════════════════╝
with tab5:
    sp1, sp2 = st.columns([2, 1])
    with sp1:
        spike_threshold = st.slider(
            "Spike alert threshold (%)",
            min_value=0.5, max_value=10.0, value=3.0, step=0.5,
            help="Alert triggers when price changes by more than this % from open",
        )
    with sp2:
        st.metric("Current threshold", f"+/- {spike_threshold}%")

    st.markdown(
        f'<p class="section-header">Real-Time Spike Alerts  (>{spike_threshold}% change)</p>',
        unsafe_allow_html=True,
    )
    alerts = []
    for sym in selected_stocks:
        sdf = df_filtered[df_filtered["symbol"] == sym].copy()
        if sdf.empty or "open" not in sdf.columns:
            continue
        sdf["chg_pct"] = ((sdf["price"] - sdf["open"]) / sdf["open"].replace(0, 1)) * 100
        alerts.append(
            sdf[sdf["chg_pct"].abs() > spike_threshold]
            [["timestamp","symbol","price","open","chg_pct"]].tail(10)
        )

    if alerts:
        alert_df = pd.concat(alerts).sort_values("timestamp", ascending=False)
        alert_df.columns = ["Time","Symbol","Price","Open","Change %"]
        alert_df["Change %"] = alert_df["Change %"].round(2)
        st.dataframe(
            alert_df.style.map(
                lambda v: "color: #ff5252" if isinstance(v, float) and v < 0 else "color: #00e676",
                subset=["Change %"],
            ),
            use_container_width=True,
        )
        fig_a = px.scatter(
            alert_df, x="Time", y="Change %", color="Symbol",
            size=alert_df["Change %"].abs(),
            template="plotly_dark",
            title=f"Spike events (threshold +/- {spike_threshold}%)",
        )
        fig_a.add_hline(y= spike_threshold, line_dash="dash", line_color="#00e676",
                        annotation_text=f"+{spike_threshold}%")
        fig_a.add_hline(y=-spike_threshold, line_dash="dash", line_color="#ff5252",
                        annotation_text=f"-{spike_threshold}%")
        fig_a.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
            height=380, margin=dict(l=60,r=10,t=40,b=40),
        )
        st.plotly_chart(fig_a, use_container_width=True)
    else:
        st.success(f"No spikes detected beyond +/- {spike_threshold}% in current data window.")


# ── Auto-refresh ──────────────────────────────────────────────
if auto_refresh:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_interval * 1000, key="autorefresh")
    else:
        import streamlit.components.v1 as components
        components.html(
            f"<script>setTimeout(()=>window.parent.location.reload(),{refresh_interval*1000});</script>",
            height=0,
        )