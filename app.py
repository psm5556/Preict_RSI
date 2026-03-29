import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

try:
    from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

st.set_page_config(
    page_title="RSI 타겟 가격 계산기",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #000000;
    }
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 RSI 타겟 가격 계산기")
st.caption("종목/지수를 입력하면 현재 RSI와 타겟 RSI 달성에 필요한 가격 및 등락률을 계산합니다.")


# ── RSI 계산 함수 ────────────────────────────────────────────────────────────────

def _gains_losses(prices: pd.Series):
    delta = prices.diff()
    return delta.clip(lower=0), -delta.clip(upper=0)


def calc_rsi_wilder(prices: pd.Series, period: int):
    """Wilder's Smoothing (표준 RSI): alpha = 1/period"""
    gain, loss = _gains_losses(prices)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi, avg_gain, avg_loss


def calc_rsi_sma(prices: pd.Series, period: int):
    """Cutler's RSI (SMA): 단순 이동평균"""
    gain, loss = _gains_losses(prices)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    oldest_gain = gain.shift(period - 1)
    oldest_loss = loss.shift(period - 1)
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi, avg_gain, avg_loss, oldest_gain, oldest_loss


def calc_rsi_ema(prices: pd.Series, period: int):
    """EMA RSI: span = period (alpha = 2/(period+1))"""
    gain, loss = _gains_losses(prices)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi, avg_gain, avg_loss


# ── 타겟 가격 역산 함수 ──────────────────────────────────────────────────────────

def _current_rsi(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def target_wilder(
    current_price: float, avg_gain: float, avg_loss: float,
    target_rsi: float, period: int,
) -> float | None:
    """
    Wilder: avg_new = avg_prev*(n-1)/n + new_val/n

    Up:   P = P_cur + (n-1)*(RS_t * avg_loss - avg_gain)
    Down: P = P_cur + (n-1)*(avg_loss - avg_gain / RS_t)
    """
    if target_rsi <= 0 or target_rsi >= 100:
        return None
    n = period
    rs_t = target_rsi / (100 - target_rsi)
    cur = _current_rsi(avg_gain, avg_loss)
    if target_rsi >= cur:
        return current_price + (n - 1) * (rs_t * avg_loss - avg_gain)
    else:
        if rs_t == 0:
            return None
        return current_price + (n - 1) * (avg_loss - avg_gain / rs_t)


def target_sma(
    current_price: float, avg_gain: float, avg_loss: float,
    oldest_gain: float, oldest_loss: float,
    target_rsi: float, period: int,
) -> float | None:
    """
    SMA: avg_new = (avg_prev*n - oldest + new_val) / n
    ΣG = avg_gain*n,  ΣL = avg_loss*n

    Up:   P = P_cur + RS_t*(ΣL - L_old) - ΣG + G_old
    Down: P = P_cur + (ΣL - L_old) - (ΣG - G_old) / RS_t
    """
    if target_rsi <= 0 or target_rsi >= 100:
        return None
    n = period
    rs_t = target_rsi / (100 - target_rsi)
    sg = avg_gain * n
    sl = avg_loss * n
    cur = _current_rsi(avg_gain, avg_loss)
    if target_rsi >= cur:
        return current_price + rs_t * (sl - oldest_loss) - sg + oldest_gain
    else:
        if rs_t == 0:
            return None
        return current_price + (sl - oldest_loss) - (sg - oldest_gain) / rs_t


def target_ema(
    current_price: float, avg_gain: float, avg_loss: float,
    target_rsi: float, period: int,
) -> float | None:
    """
    EMA: alpha = 2/(n+1),  avg_new = avg_prev*(n-1)/(n+1) + new_val*2/(n+1)

    Up:   P = P_cur + (n-1)/2 * (RS_t * avg_loss - avg_gain)
    Down: P = P_cur + (n-1)/2 * (avg_loss - avg_gain / RS_t)
    """
    if target_rsi <= 0 or target_rsi >= 100:
        return None
    n = period
    rs_t = target_rsi / (100 - target_rsi)
    k = (n - 1) / 2
    cur = _current_rsi(avg_gain, avg_loss)
    if target_rsi >= cur:
        return current_price + k * (rs_t * avg_loss - avg_gain)
    else:
        if rs_t == 0:
            return None
        return current_price + k * (avg_loss - avg_gain / rs_t)


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────

def build_table(
    target_rsi_list, current_price, current_rsi, calc_fn
) -> pd.DataFrame:
    rows = []
    for t_rsi in target_rsi_list:
        t_price = calc_fn(t_rsi)
        if t_price is None or t_price <= 0:
            rows.append({"타겟 RSI": t_rsi, "예상 가격": None, "등락률 (%)": None, "방향": "-"})
        else:
            pct = (t_price - current_price) / current_price * 100
            direction = "▲ 상승" if t_price > current_price else ("▼ 하락" if t_price < current_price else "─")
            rows.append({
                "타겟 RSI": t_rsi,
                "예상 가격": round(t_price, 4),
                "등락률 (%)": round(pct, 2),
                "방향": direction,
            })
    return pd.DataFrame(rows)


def style_table(df: pd.DataFrame):
    def _row(row):
        rsi = row["타겟 RSI"]
        if rsi <= 30:
            return ["color: #4fc3f7"] * len(row)
        if rsi >= 70:
            return ["color: #ef9a9a"] * len(row)
        return [""] * len(row)

    return df.style.apply(_row, axis=1).format(
        {"예상 가격": lambda v: f"{v:,.4g}" if v is not None else "계산 불가",
         "등락률 (%)": lambda v: f"{v:+.2f}%" if v is not None else "-"},
        na_rep="-",
    )


# ── 피처 엔지니어링 ───────────────────────────────────────────────────────────

def build_features(hist: pd.DataFrame, period_rsi: int = 14) -> pd.DataFrame:
    """기술 지표를 계산해 공변량 DataFrame 반환."""
    close  = hist["Close"]
    high   = hist["High"]
    low    = hist["Low"]
    volume = hist.get("Volume", pd.Series(np.nan, index=close.index))

    f = pd.DataFrame(index=close.index)

    # RSI (Wilder)
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period_rsi, adjust=False).mean()
    al = loss.ewm(alpha=1 / period_rsi, adjust=False).mean()
    f["rsi"] = 100 - 100 / (1 + ag / al)

    # MA 비율 (price / MA - 1)
    for p in [20, 60, 125, 200]:
        ma = close.rolling(p).mean()
        f[f"ma_ratio_{p}"] = (close / ma - 1).replace([np.inf, -np.inf], np.nan)

    # MACD (normalized)
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    f["macd"]       = (macd   / close).replace([np.inf, -np.inf], np.nan)
    f["macd_sig"]   = (signal / close).replace([np.inf, -np.inf], np.nan)
    f["macd_hist"]  = ((macd - signal) / close).replace([np.inf, -np.inf], np.nan)

    # Bollinger Bands %B, Width
    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_u  = ma20 + 2 * std20
    bb_l  = ma20 - 2 * std20
    bb_rng = (bb_u - bb_l).replace(0, np.nan)
    f["bb_pct"]   = (close - bb_l) / bb_rng
    f["bb_width"] = bb_rng / ma20

    # ATR 비율
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    f["atr_ratio"] = (tr.ewm(alpha=1/14, adjust=False).mean() / close).replace([np.inf, -np.inf], np.nan)

    # Stochastic %K / %D
    lo14 = low.rolling(14).min()
    hi14 = high.rolling(14).max()
    rng14 = (hi14 - lo14).replace(0, np.nan)
    stoch_k = (close - lo14) / rng14 * 100
    f["stoch_k"] = stoch_k
    f["stoch_d"] = stoch_k.rolling(3).mean()

    # Log returns (1·5·20봉)
    for lag in [1, 5, 20]:
        f[f"log_ret_{lag}"] = np.log(close / close.shift(lag))

    # Realized volatility (20봉)
    f["vol_20"] = np.log(close / close.shift()).rolling(20).std()

    # 거래량 비율
    if volume.notna().any():
        vol_ma = volume.rolling(20).mean().replace(0, np.nan)
        f["vol_ratio"] = (volume / vol_ma).replace([np.inf, -np.inf], np.nan)

    return f


# ── TabPFN-TS 예측 ─────────────────────────────────────────────────────────────

def run_tabpfn_forecast(
    close: pd.Series,
    features: pd.DataFrame,
    horizon: int,
    token: str,
) -> pd.DataFrame | None:
    """
    TabPFN-TS CLIENT 모드로 예측 실행.
    반환: 분위수(0.1~0.9) + target 컬럼을 가진 DataFrame (index=timestamp)
    """
    import os
    os.environ["TABPFN_API_TOKEN"] = token

    # 공통 인덱스 정렬 및 NaN 제거
    common = close.index.intersection(features.index)
    c = close.loc[common].copy()
    f = features.loc[common].copy()
    valid = f.notna().all(axis=1) & c.notna()
    c, f = c[valid], f[valid]

    if len(c) < 30:
        return None

    # context_df 구성
    timestamps = pd.to_datetime(c.index).tz_localize(None)  # tz-naive 필요
    ctx = pd.DataFrame({"item_id": "price", "timestamp": timestamps, "target": c.values})
    for col in f.columns:
        ctx[col] = f[col].values

    pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.CLIENT)
    pred = pipeline.predict_df(context_df=ctx, prediction_length=horizon)

    # Multi-index (item_id, timestamp) → timestamp index
    if isinstance(pred.index, pd.MultiIndex):
        pred = pred.xs("price", level="item_id") if "price" in pred.index.get_level_values(0) else pred.droplevel(0)
    pred.index = pd.to_datetime(pred.index)
    return pred


# ── 한국 티커 자동 처리 ────────────────────────────────────────────────────────

import re

@st.cache_data(ttl=3600, show_spinner=False)
def resolve_ticker(ticker: str) -> str:
    """
    6자리 한국 주식 티커(숫자/영문 혼합, 예: 005930, 0046A0)에
    yfinance 접미사(.KS 또는 .KQ)를 자동으로 붙여 반환.
    - 이미 접미사가 있거나(. 포함) 지수(^ 시작)면 그대로 반환.
    - .KS 조회 성공 → .KS, .KQ 조회 성공 → .KQ, 둘 다 실패 → 원본 반환.
    """
    if "." in ticker or ticker.startswith("^"):
        return ticker
    if not re.match(r'^\d[0-9A-Za-z]{5}$', ticker):
        return ticker
    for suffix in [".KS", ".KQ"]:
        try:
            h = yf.Ticker(ticker + suffix).history(period="5d")
            if not h.empty:
                return ticker + suffix
        except Exception:
            continue
    return ticker


# ── 구글시트 종목 로드 ─────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def load_sheet_tickers() -> tuple[pd.DataFrame, str]:
    """구글시트에서 티커·기업명 목록을 읽어 (DataFrame, 에러메시지) 반환."""
    from urllib.parse import quote
    try:
        sheet_id   = st.secrets["GOOGLE_SHEET_ID"]
        sheet_name = st.secrets["GOOGLE_SHEET_NAME"]
    except KeyError as e:
        return pd.DataFrame(), f"secrets 키 없음: {e}"

    encoded_name = quote(sheet_name)
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/gviz/tq?tqx=out:csv&sheet={encoded_name}"
    )
    try:
        df = pd.read_csv(url)
    except Exception as e:
        return pd.DataFrame(), f"CSV 읽기 실패: {e}"

    needed = {"티커", "기업명"}
    if not needed.issubset(df.columns):
        return pd.DataFrame(), f"필요한 컬럼 없음. 실제 컬럼: {list(df.columns)}"

    result = df[["티커", "기업명"]].dropna(subset=["티커"]).reset_index(drop=True)
    return result, ""


# ── 사이드바 ──────────────────────────────────────────────────────────────────

PRESET_TICKERS = {
    "S&P 500": "^GSPC",
    "나스닥 종합": "^IXIC",
    "나스닥 100": "^NDX",
    "코스피": "^KS11",
    "코스닥": "^KQ11",
}

# 세션 상태 초기화
if "_ticker" not in st.session_state:
    st.session_state["_ticker"] = "^GSPC"
if "_sheet_sel" not in st.session_state:
    st.session_state["_sheet_sel"] = "— 선택 —"

with st.sidebar:
    st.header("설정")

    # 주요 지수 빠른 선택
    st.caption("주요 지수 빠른 선택")
    preset_cols = st.columns(5)
    for col, (label, symbol) in zip(preset_cols, PRESET_TICKERS.items()):
        if col.button(label, use_container_width=True):
            st.session_state["_ticker"] = symbol
            st.session_state["_sheet_sel"] = "— 선택 —"  # selectbox 초기화

    # 구글시트 종목 선택
    st.divider()
    st.caption("구글시트 종목 목록에서 선택")
    refresh_col, _ = st.columns([1, 2])
    if refresh_col.button("🔄 새로고침", use_container_width=True):
        st.cache_data.clear()

    sheet_df, sheet_err = load_sheet_tickers()
    if sheet_df.empty:
        msg = sheet_err if sheet_err else "종목 목록이 비어 있습니다."
        st.error(msg)
    else:
        options = ["— 선택 —"] + [
            f"{row['기업명']}  ({row['티커']})" for _, row in sheet_df.iterrows()
        ]
        # 현재 selectbox 위치 계산 (초기화 후에는 0)
        cur_sel = st.session_state["_sheet_sel"]
        sel_idx = options.index(cur_sel) if cur_sel in options else 0

        selected = st.selectbox(
            "종목 선택",
            options,
            index=sel_idx,
            label_visibility="collapsed",
        )
        # 이전 값과 달라졌을 때만 티커 업데이트
        if selected != st.session_state["_sheet_sel"]:
            st.session_state["_sheet_sel"] = selected
            if selected != "— 선택 —":
                st.session_state["_ticker"] = selected.split("(")[-1].rstrip(")")

    st.divider()
    ticker_input = st.text_input(
        "종목/지수 심볼 직접 입력",
        value=st.session_state["_ticker"],
        help="예) AAPL, TSLA, ^GSPC (S&P500), ^IXIC (나스닥), ^NDX (나스닥100), ^KS11 (KOSPI), 005930.KS (삼성전자)",
    )
    # 직접 입력 시 상태 동기화
    if ticker_input != st.session_state["_ticker"]:
        st.session_state["_ticker"] = ticker_input
        st.session_state["_sheet_sel"] = "— 선택 —"

    period_rsi = st.number_input(
        "RSI 기간 (봉 수)", min_value=2, max_value=50, value=14, step=1,
    )

    interval_options = {
        "일봉": "1d", "주봉": "1wk", "월봉": "1mo",
        "1시간": "1h", "4시간": "4h", "15분": "15m",
    }
    interval_label = st.selectbox(
        "차트 간격", list(interval_options.keys()),
        index=list(interval_options.keys()).index("주봉"),
    )
    interval = interval_options[interval_label]

    lookback_options = {
        "3개월": "3mo", "6개월": "6mo", "1년": "1y", "2년": "2y",
        "5년": "5y", "10년": "10y", "15년": "15y",
    }
    lookback_label = st.selectbox(
        "조회 기간", list(lookback_options.keys()),
        index=list(lookback_options.keys()).index("5년"),
    )
    lookback = lookback_options[lookback_label]

    st.divider()
    st.caption("타겟 RSI: 고정값 25·30·50·70·75 + 기간 내 RSI min/max 자동 추가")

    run_btn = st.button("계산하기", type="primary", use_container_width=True)

FIXED_RSI = [25, 30, 50, 70, 75]

# ── 메인 ──────────────────────────────────────────────────────────────────────

_ticker_val = st.session_state["_ticker"].strip()

if run_btn or _ticker_val:
    resolved = resolve_ticker(_ticker_val)
    if resolved != _ticker_val:
        st.sidebar.caption(f"티커 변환: `{_ticker_val}` → `{resolved}`")

    with st.spinner(f"{resolved} 데이터 로딩 중..."):
        try:
            hist = yf.Ticker(resolved).history(period=lookback, interval=interval)
        except Exception as e:
            st.error(f"데이터 로딩 실패: {e}")
            st.stop()

    if hist.empty:
        st.error(f"'{_ticker_val}' 데이터를 가져오지 못했습니다. 심볼을 확인해주세요.")
        st.stop()

    close = hist["Close"].dropna()

    if len(close) < period_rsi + 1:
        st.error(f"데이터 부족. 최소 {period_rsi + 1}개 봉이 필요합니다.")
        st.stop()

    # 세 방식 계산
    rsi_w, ag_w, al_w = calc_rsi_wilder(close, period_rsi)
    rsi_s, ag_s, al_s, og_s, ol_s = calc_rsi_sma(close, period_rsi)
    rsi_e, ag_e, al_e = calc_rsi_ema(close, period_rsi)

    # 방식별 기간 내 RSI min/max (반올림 없이 실제값 사용)
    def make_target_list(rsi_series):
        mn = float(rsi_series.dropna().min())
        mx = float(rsi_series.dropna().max())
        return sorted({mn, *map(float, FIXED_RSI), mx})

    current_price = float(close.iloc[-1])
    last_date = close.index[-1]
    date_str = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)

    cr_w = float(rsi_w.iloc[-1])
    cr_s = float(rsi_s.iloc[-1])
    cr_e = float(rsi_e.iloc[-1])

    # ── 헤더 ────────────────────────────────────────────────────────────────
    st.subheader(f"{resolved.upper()}  |  {interval_label}  |  {date_str}")

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        st.metric("현재 가격", f"{current_price:,.4g}")

    def rsi_label(v):
        if v >= 70:
            return "🔴 과매수"
        if v <= 30:
            return "🟢 과매도"
        return "⚪ 중립"

    with c1:
        st.metric("Wilder RSI", f"{cr_w:.2f}", rsi_label(cr_w))
    with c2:
        st.metric("Cutler RSI (SMA)", f"{cr_s:.2f}", rsi_label(cr_s))
    with c3:
        st.metric("EMA RSI", f"{cr_e:.2f}", rsi_label(cr_e))

    # ── 타겟 가격 테이블 (탭) ─────────────────────────────────────────────
    st.subheader("타겟 RSI별 예상 가격")

    def fmt_list(lst):
        return ", ".join(f"{v:.2f}" if v != int(v) else str(int(v)) for v in lst)

    tl_w = make_target_list(rsi_w)
    tl_s = make_target_list(rsi_s)
    tl_e = make_target_list(rsi_e)

    tab_w, tab_s, tab_e = st.tabs(
        ["📊 Wilder (표준)", "📊 Cutler (SMA)", "📊 EMA RSI"]
    )

    with tab_w:
        st.caption(f"타겟 목록: {fmt_list(tl_w)}  (기간 min **{tl_w[0]:.2f}** / max **{tl_w[-1]:.2f}**)")
        df_w = build_table(
            tl_w, current_price, cr_w,
            lambda t: target_wilder(current_price, float(ag_w.iloc[-1]), float(al_w.iloc[-1]), t, period_rsi),
        )
        st.dataframe(style_table(df_w), use_container_width=True, hide_index=True)

    with tab_s:
        st.caption(f"타겟 목록: {fmt_list(tl_s)}  (기간 min **{tl_s[0]:.2f}** / max **{tl_s[-1]:.2f}**)")
        df_s = build_table(
            tl_s, current_price, cr_s,
            lambda t: target_sma(
                current_price, float(ag_s.iloc[-1]), float(al_s.iloc[-1]),
                float(og_s.iloc[-1]), float(ol_s.iloc[-1]), t, period_rsi,
            ),
        )
        st.dataframe(style_table(df_s), use_container_width=True, hide_index=True)

    with tab_e:
        st.caption(f"타겟 목록: {fmt_list(tl_e)}  (기간 min **{tl_e[0]:.2f}** / max **{tl_e[-1]:.2f}**)")
        df_e = build_table(
            tl_e, current_price, cr_e,
            lambda t: target_ema(current_price, float(ag_e.iloc[-1]), float(al_e.iloc[-1]), t, period_rsi),
        )
        st.dataframe(style_table(df_e), use_container_width=True, hide_index=True)

    # ── 차트 ──────────────────────────────────────────────────────────────
    st.subheader("차트")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.04,
    )

    # 캔들차트
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"],
            name="가격",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # 이동평균선
    MA_PERIODS = [20, 60, 125, 200, 240, 365]
    MA_COLORS  = ["#ff0000", "#00cc00", "#3399ff", "#ffff00", "#ff8800", "#cccccc"]
    for ma_p, ma_c in zip(MA_PERIODS, MA_COLORS):
        ma = close.rolling(ma_p).mean()
        if ma.dropna().empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=ma.index, y=ma.values,
                line=dict(color=ma_c, width=1),
                name=f"MA{ma_p}",
                opacity=0.7,
            ),
            row=1, col=1,
        )

    # 현재가 수평선
    fig.add_hline(
        y=current_price, line_dash="solid", line_color="#ffeb3b", line_width=1.5,
        annotation_text=f"현재 {current_price:,.4g}", annotation_position="right",
        row=1, col=1,
    )

    # Wilder 타겟 가격 수평선
    for t_rsi in tl_w:
        t_price = target_wilder(
            current_price, float(ag_w.iloc[-1]), float(al_w.iloc[-1]), t_rsi, period_rsi
        )
        if t_price is None or t_price <= 0:
            continue
        is_fixed = t_rsi in FIXED_RSI
        line_color = "#ef9a9a" if t_rsi >= 70 else ("#4fc3f7" if t_rsi <= 30 else "#bdbdbd")
        pct = (t_price - current_price) / current_price * 100
        fig.add_hline(
            y=t_price,
            line_dash="dot" if is_fixed else "dash",
            line_color=line_color,
            line_width=1,
            annotation_text=f"RSI {t_rsi:.1f}  ({pct:+.1f}%)",
            annotation_position="right",
            annotation_font_size=10,
            row=1, col=1,
        )

    # RSI 세 라인
    rsi_traces = [
        (rsi_w, "#4fc3f7", f"Wilder({period_rsi})"),
        (rsi_s, "#a5d6a7", f"Cutler SMA({period_rsi})"),
        (rsi_e, "#ce93d8", f"EMA({period_rsi})"),
    ]
    for rsi_ser, color, name in rsi_traces:
        fig.add_trace(
            go.Scatter(x=rsi_ser.index, y=rsi_ser.values,
                       line=dict(color=color, width=1.5), name=name),
            row=2, col=1,
        )

    fig.add_hline(y=70, line_dash="dash", line_color="#ef9a9a", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#4fc3f7", line_width=1, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#9e9e9e", line_width=1, row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        height=750,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=80, t=30, b=10),
    )
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)

    # ── TabPFN-TS 가격 예측 ───────────────────────────────────────────────
    with st.expander("🔮 TabPFN-TS 가격 예측 (AI)"):
        if not TABPFN_AVAILABLE:
            st.warning("`tabpfn-time-series` 미설치. `pip install tabpfn-time-series` 후 재실행하세요.")
        else:
            try:
                tabpfn_token = st.secrets["TABPFN_API_TOKEN"]
            except KeyError:
                tabpfn_token = None
                st.error("secrets에 `TABPFN_API_TOKEN` 이 없습니다.")

            if tabpfn_token:
                col_h, col_btn = st.columns([3, 1])
                horizon_opts = {
                    f"4봉  (~{4}주)" if interval == "1wk" else f"4봉": 4,
                    f"8봉  (~{8}주)" if interval == "1wk" else f"8봉": 8,
                    f"12봉 (~3개월)" if interval == "1wk" else f"12봉": 12,
                    f"26봉 (~6개월)" if interval == "1wk" else f"26봉": 26,
                    f"52봉 (~1년)"   if interval == "1wk" else f"52봉": 52,
                }
                horizon_label = col_h.selectbox("예측 기간", list(horizon_opts.keys()), index=2)
                horizon = horizon_opts[horizon_label]

                run_forecast = col_btn.button("예측 실행", type="primary", use_container_width=True)

                # 피처 설명
                with st.expander("사용 피처 목록", expanded=False):
                    st.markdown("""
| 카테고리 | 피처 |
|---|---|
| 모멘텀 | RSI(Wilder), Stochastic %K/%D |
| 추세 | MA 비율(20·60·125·200), MACD/Signal/Hist |
| 변동성 | Bollinger %B·Width, ATR 비율, 실현변동성(20봉) |
| 수익률 | 로그 수익률(1·5·20봉) |
| 거래량 | 거래량 / MA20 비율 |
""")

                if run_forecast:
                    with st.spinner("피처 계산 및 TabPFN-TS 예측 중..."):
                        feats = build_features(hist, period_rsi)
                        pred_df = run_tabpfn_forecast(close, feats, horizon, tabpfn_token)

                    if pred_df is None:
                        st.error("유효 데이터 부족으로 예측 실패.")
                    else:
                        # ── 예측 테이블
                        q_cols = [c for c in pred_df.columns if str(c) in [str(q/10) for q in range(1,10)]]
                        has_q  = len(q_cols) >= 3

                        tbl_rows = []
                        for i, (ts, row) in enumerate(pred_df.iterrows()):
                            tgt = row.get("target", row.get("0.5", np.nan))
                            lo  = row.get(0.1, row.get("0.1", np.nan)) if has_q else np.nan
                            hi  = row.get(0.9, row.get("0.9", np.nan)) if has_q else np.nan
                            pct = (tgt - current_price) / current_price * 100 if not np.isnan(tgt) else np.nan
                            tbl_rows.append({
                                "봉": i + 1,
                                "예상 날짜": ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts),
                                "예측가 (중앙값)": round(tgt, 4) if not np.isnan(tgt) else None,
                                "하단 10%": round(lo, 4)  if not np.isnan(lo)  else None,
                                "상단 90%": round(hi, 4)  if not np.isnan(hi)  else None,
                                "등락률 (%)": round(pct, 2)  if not np.isnan(pct) else None,
                            })

                        st.dataframe(
                            pd.DataFrame(tbl_rows).style.format({
                                "예측가 (중앙값)": lambda v: f"{v:,.4g}" if v else "-",
                                "하단 10%":  lambda v: f"{v:,.4g}" if v else "-",
                                "상단 90%":  lambda v: f"{v:,.4g}" if v else "-",
                                "등락률 (%)": lambda v: f"{v:+.2f}%" if v else "-",
                            }, na_rep="-"),
                            use_container_width=True,
                            hide_index=True,
                        )

                        # ── 예측 차트 (fan chart)
                        fig_fc = go.Figure()

                        # 최근 120봉 실제가
                        hist_show = close.iloc[-120:]
                        fig_fc.add_trace(go.Scatter(
                            x=hist_show.index, y=hist_show.values,
                            name="실제 가격", line=dict(color="#ffffff", width=1.5),
                        ))

                        pred_x = pred_df.index

                        # 신뢰대역 (10%~90%)
                        if has_q:
                            def _col(name):
                                for k in [name, float(name)]:
                                    if k in pred_df.columns:
                                        return pred_df[k].values
                                return None

                            q10 = _col("0.1"); q90 = _col("0.9")
                            q25 = _col("0.25"); q75 = _col("0.75")

                            if q10 is not None and q90 is not None:
                                fig_fc.add_trace(go.Scatter(
                                    x=list(pred_x) + list(pred_x[::-1]),
                                    y=list(q90) + list(q10[::-1]),
                                    fill="toself", fillcolor="rgba(100,160,255,0.15)",
                                    line=dict(color="rgba(0,0,0,0)"), name="10%~90%",
                                ))
                            if q25 is not None and q75 is not None:
                                fig_fc.add_trace(go.Scatter(
                                    x=list(pred_x) + list(pred_x[::-1]),
                                    y=list(q75) + list(q25[::-1]),
                                    fill="toself", fillcolor="rgba(100,160,255,0.30)",
                                    line=dict(color="rgba(0,0,0,0)"), name="25%~75%",
                                ))

                        # 중앙값 예측선
                        med_col = "target" if "target" in pred_df.columns else ("0.5" if "0.5" in pred_df.columns else None)
                        if med_col:
                            fig_fc.add_trace(go.Scatter(
                                x=pred_x, y=pred_df[med_col].values,
                                name="예측 중앙값", line=dict(color="#64a0ff", width=2, dash="dash"),
                            ))

                        # 현재가 연결선
                        fig_fc.add_shape(
                            type="line",
                            x0=close.index[-1], x1=pred_x[0] if len(pred_x) else close.index[-1],
                            y0=current_price, y1=current_price,
                            line=dict(color="#ffeb3b", width=1, dash="dot"),
                        )

                        fig_fc.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#000000", plot_bgcolor="#000000",
                            height=450, title="TabPFN-TS 가격 예측",
                            margin=dict(l=10, r=10, t=40, b=10),
                            legend=dict(orientation="h"),
                        )
                        st.plotly_chart(fig_fc, use_container_width=True)

                        st.caption("⚠️ 예측 결과는 참고용이며 투자 판단의 근거로 사용할 수 없습니다.")

    # ── pandas_ta 검증 ────────────────────────────────────────────────────
    with st.expander("🔬 pandas_ta 수치 비교 검증"):
        if not PANDAS_TA_AVAILABLE:
            st.warning("`pandas_ta` 가 설치되지 않았습니다. `pip install pandas_ta` 후 재실행하세요.")
        else:
            n_rows = st.slider("비교할 최근 봉 수", min_value=5, max_value=100, value=20)

            # pandas_ta 계산
            delta = close.diff()
            gain_ser = delta.clip(lower=0)
            loss_ser = -delta.clip(upper=0)

            # Wilder: ta.rsi() 내부적으로 RMA(alpha=1/n) 사용
            pta_rsi_w = ta.rsi(close, length=period_rsi)

            # SMA: ta.sma() 으로 gain/loss 평균
            pta_ag_s = ta.sma(gain_ser, length=period_rsi)
            pta_al_s = ta.sma(loss_ser, length=period_rsi)
            pta_rsi_s = 100 - 100 / (1 + pta_ag_s / pta_al_s)

            # EMA: ta.ema() span=period (alpha=2/(period+1))
            pta_ag_e = ta.ema(gain_ser, length=period_rsi)
            pta_al_e = ta.ema(loss_ser, length=period_rsi)
            pta_rsi_e = 100 - 100 / (1 + pta_ag_e / pta_al_e)

            idx = close.index[-n_rows:]

            def fmt_diff(diff_val):
                """차이가 1e-6 이하면 ✅, 아니면 ⚠️"""
                return f"✅ {diff_val:.2e}" if diff_val < 1e-6 else f"⚠️ {diff_val:.6f}"

            methods = [
                ("Wilder", rsi_w, pta_rsi_w),
                ("Cutler (SMA)", rsi_s, pta_rsi_s),
                ("EMA", rsi_e, pta_rsi_e),
            ]

            tab_vw, tab_vs, tab_ve = st.tabs(
                ["📊 Wilder 검증", "📊 Cutler(SMA) 검증", "📊 EMA 검증"]
            )

            for tab, (label, our_rsi, pta_rsi) in zip(
                [tab_vw, tab_vs, tab_ve], methods
            ):
                with tab:
                    our = our_rsi.reindex(idx).rename("직접 구현")
                    pta = pta_rsi.reindex(idx).rename("pandas_ta")
                    diff = (our - pta).abs().rename("절대 오차")

                    comp_df = pd.DataFrame({"직접 구현": our, "pandas_ta": pta, "절대 오차": diff})
                    comp_df.index = comp_df.index.strftime("%Y-%m-%d %H:%M") if hasattr(comp_df.index, "strftime") else comp_df.index

                    max_diff = diff.max()
                    mean_diff = diff.mean()

                    verdict = "✅ 일치 (오차 < 1e-6)" if max_diff < 1e-6 else f"⚠️ 최대 오차: {max_diff:.6f}"
                    st.markdown(f"**{label}** — {verdict} &nbsp;|&nbsp; 평균 오차: `{mean_diff:.2e}`")

                    def highlight_diff(col):
                        if col.name != "절대 오차":
                            return [""] * len(col)
                        return [
                            "color: #ef9a9a" if v >= 1e-6 else "color: #a5d6a7"
                            for v in col
                        ]

                    st.dataframe(
                        comp_df.style.apply(highlight_diff).format("{:.6f}"),
                        use_container_width=True,
                    )

                    # 비교 차트
                    fig_v = go.Figure()
                    fig_v.add_trace(go.Scatter(
                        x=our_rsi.index, y=our_rsi.values,
                        name="직접 구현", line=dict(color="#4fc3f7", width=2),
                    ))
                    fig_v.add_trace(go.Scatter(
                        x=pta_rsi.index, y=pta_rsi.values,
                        name="pandas_ta", line=dict(color="#ffb74d", width=1.5, dash="dot"),
                    ))
                    fig_v.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#000000",
                        plot_bgcolor="#000000",
                        height=300,
                        title=f"{label} RSI 비교",
                        margin=dict(l=10, r=10, t=40, b=10),
                        legend=dict(orientation="h"),
                    )
                    st.plotly_chart(fig_v, use_container_width=True)

    # ── 계산 방식 설명 ─────────────────────────────────────────────────────
    with st.expander("계산 방식 설명"):
        st.markdown(f"""
### RSI 계산 방식 비교

| 방식 | 평활화 | alpha |
|------|--------|-------|
| **Wilder (표준)** | EWM (adjust=False) | α = 1/{period_rsi} |
| **Cutler (SMA)** | 단순 이동평균 | — |
| **EMA RSI** | EWM (adjust=False) | α = 2/{period_rsi+1} |

$$RSI = 100 - \\frac{{100}}{{1 + RS}}, \\quad RS = \\frac{{\\overline{{Gain}}}}{{\\overline{{Loss}}}}$$

---

### 타겟 가격 역산 공식

다음 봉 종가 $P$ 가 타겟 RSI를 달성하는 값. $RS_t = \\dfrac{{RSI_t}}{{100 - RSI_t}}$

**Wilder** (α = 1/n → avg_new = avg × (n-1)/n + val/n)

$$P_{{\\uparrow}} = P_{{cur}} + (n-1)(RS_t \\cdot \\overline{{L}} - \\overline{{G}})$$
$$P_{{\\downarrow}} = P_{{cur}} + (n-1)\\left(\\overline{{L}} - \\frac{{\\overline{{G}}}}{{RS_t}}\\right)$$

**Cutler / SMA** (rolling window: oldest $G_0, L_0$ 이탈, $\\Sigma G = \\overline{{G}} \\cdot n$)

$$P_{{\\uparrow}} = P_{{cur}} + RS_t(\\Sigma L - L_0) - \\Sigma G + G_0$$
$$P_{{\\downarrow}} = P_{{cur}} + (\\Sigma L - L_0) - \\frac{{\\Sigma G - G_0}}{{RS_t}}$$

**EMA RSI** (α = 2/(n+1) → avg_new = avg × (n-1)/(n+1) + val × 2/(n+1))

$$P_{{\\uparrow}} = P_{{cur}} + \\frac{{n-1}}{{2}}(RS_t \\cdot \\overline{{L}} - \\overline{{G}})$$
$$P_{{\\downarrow}} = P_{{cur}} + \\frac{{n-1}}{{2}}\\left(\\overline{{L}} - \\frac{{\\overline{{G}}}}{{RS_t}}\\right)$$

> **주의**: 단일 봉 기준 역산이며 실제 가격은 시장 상황에 따라 달라집니다.
""")
