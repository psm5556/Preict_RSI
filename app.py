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
    import tabpfn_client  # noqa: F401 (tabpfn-time-series 의존성)
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

try:
    import lightgbm  # noqa: F401
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

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


# ── 미래 날짜 생성 헬퍼 ───────────────────────────────────────────────────────

def _future_dates(last_date, pred_len: int, interval: str) -> pd.DatetimeIndex:
    """봉 간격에 맞는 미래 날짜 시퀀스 생성."""
    freq_map = {
        "1d":  "B",       # 영업일
        "1wk": "W-FRI",   # 매주 금요일
        "1mo": "MS",      # 매월 1일
        "1h":  "h",
        "4h":  "4h",
        "15m": "15min",
    }
    freq = freq_map.get(interval, "B")
    last = pd.Timestamp(last_date)
    if last.tzinfo is not None:
        last = last.tz_convert(None)
    return pd.date_range(start=last, periods=pred_len + 1, freq=freq)[1:]


# ── TabPFN-TS 예측 (tabpfn-time-series) ───────────────────────────────────────

def _patch_tabpfn_token_path():
    """
    tabpfn_client 토큰 저장 경로를 쓰기 가능한 위치로 자동 설정.
    로컬 PC: ~/.tabpfn/config  /  Streamlit Cloud: /tmp/tabpfn_auth/config
    """
    import pathlib, os
    candidates = [
        pathlib.Path.home() / ".tabpfn",
        pathlib.Path("/tmp") / "tabpfn_auth",
        pathlib.Path(os.getcwd()) / ".tabpfn",
    ]
    chosen_dir = None
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / "_write_probe"
            probe.write_text("ok")
            probe.unlink()
            chosen_dir = d
            break
        except Exception:
            continue
    if chosen_dir is None:
        return None
    token_file = chosen_dir / "config"
    try:
        from tabpfn_client.service_wrapper import UserAuthenticationClient
        import tabpfn_client.service_wrapper as sw
        UserAuthenticationClient.CACHED_TOKEN_FILE = token_file
        sw.CACHE_DIR = chosen_dir
    except Exception:
        pass
    try:
        import tabpfn_client.constants as const
        const.CACHE_DIR = chosen_dir
    except Exception:
        pass
    return token_file


@st.cache_data(ttl=3600, show_spinner=False)
def _get_tabpfn_ts_version() -> tuple:
    """tabpfn-time-series 설치 버전 반환 (major, minor, patch)"""
    try:
        import importlib.metadata
        ver = importlib.metadata.version("tabpfn-time-series")
        parts = [int(x) for x in ver.split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts), ver
    except Exception:
        return (0, 0, 0), "unknown"


def _clean_timeseries_for_tabpfn(values, timestamps):
    """
    TabPFN-TS 입력 전처리: 정렬, 중복 제거, NaN 보간, 일별 리샘플링.
    Returns (values_list, timestamps_list)
    """
    dates = pd.to_datetime(list(timestamps))
    vals  = pd.to_numeric(list(values), errors="coerce")
    s = pd.Series(vals, index=dates, name="target").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.interpolate(method="time").ffill().bfill()
    s = s.resample("D").last().ffill()
    s = s.dropna()
    return s.tolist(), s.index.strftime("%Y-%m-%d").tolist()


def _run_tabpfn_v1(values, timestamps, pred_len, item_id, token):
    """tabpfn-time-series >= 1.0.0 API"""
    import tabpfn_client
    from tabpfn_time_series import (
        TimeSeriesDataFrame,
        FeatureTransformer,
        TabPFNTimeSeriesPredictor,
        TabPFNMode,
    )
    from tabpfn_time_series.data_preparation import generate_test_X
    from tabpfn_time_series.features import RunningIndexFeature, CalendarFeature

    if token:
        _patch_tabpfn_token_path()
        tabpfn_client.set_access_token(token)

    clean_vals, clean_dates = _clean_timeseries_for_tabpfn(values, timestamps)
    if len(clean_vals) < pred_len + 10:
        raise ValueError(
            f"전처리 후 데이터 부족 ({len(clean_vals)}행). 조회 기간을 늘려주세요."
        )
    clean_ts = pd.to_datetime(clean_dates)

    df = pd.DataFrame(
        {"target": clean_vals},
        index=pd.MultiIndex.from_arrays(
            [[item_id] * len(clean_ts), clean_ts],
            names=["item_id", "timestamp"],
        ),
    )
    full_tsdf = TimeSeriesDataFrame(df)
    train = full_tsdf
    test  = generate_test_X(train, pred_len)

    # 과거 검증용 backtesting
    train_bt, test_bt = None, None
    try:
        if len(clean_vals) > pred_len:
            bt_vals  = clean_vals[:-pred_len]
            bt_dates = clean_ts[:-pred_len]
            df_bt = pd.DataFrame(
                {"target": bt_vals},
                index=pd.MultiIndex.from_arrays(
                    [[item_id] * len(bt_dates), bt_dates],
                    names=["item_id", "timestamp"],
                ),
            )
            train_bt = TimeSeriesDataFrame(df_bt)
            test_bt  = generate_test_X(train_bt, pred_len)
    except Exception:
        train_bt, test_bt = None, None

    # 피처 공학
    features = [RunningIndexFeature(), CalendarFeature()]
    try:
        from tabpfn_time_series.features import AutoSeasonalFeature
        asf = AutoSeasonalFeature()
        _, _ = FeatureTransformer([asf]).transform(train.copy(), test.copy())
        features.append(AutoSeasonalFeature())
    except Exception:
        pass

    train, test = FeatureTransformer(features).transform(train, test)
    if train_bt is not None:
        try:
            train_bt, test_bt = FeatureTransformer(features).transform(train_bt, test_bt)
        except Exception:
            train_bt, test_bt = None, None

    predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.CLIENT)
    pred = predictor.predict(train, test)

    def _normalize_pred(df: pd.DataFrame) -> pd.DataFrame:
        """timestamp 컬럼 정규화 + 분위수 컬럼을 float → string 으로 통일."""
        ts_cands = [c for c in df.columns if "time" in str(c).lower() and c != "item_id"]
        if ts_cands and "timestamp" not in df.columns:
            df = df.rename(columns={ts_cands[0]: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # float 분위수 컬럼(0.1, 0.25 …) → string("0.1", "0.25" …)
        rename_map = {c: str(c) for c in df.columns if isinstance(c, float)}
        if rename_map:
            df = df.rename(columns=rename_map)
        # "mean" 컬럼이 있으면 "0.5" 또는 "target" 으로 alias
        if "mean" in df.columns and "0.5" not in df.columns:
            df = df.rename(columns={"mean": "0.5"})
        return df

    pred_df = _normalize_pred(pred.reset_index())

    hist_pred_df = None
    if train_bt is not None:
        try:
            bt_pred = predictor.predict(train_bt, test_bt)
            hist_pred_df = _normalize_pred(bt_pred.reset_index())
        except Exception:
            hist_pred_df = None

    return pred_df, hist_pred_df


def _run_tabpfn_v0(values, timestamps, pred_len, token):  # noqa: ARG001
    """tabpfn-time-series 0.x (구버전) API"""
    from tabpfn_time_series import TabPFNTimeSeriesPredictor
    train_series = pd.Series(
        values[:-pred_len] if len(values) > pred_len else values,
        index=pd.DatetimeIndex(timestamps[:-pred_len] if len(timestamps) > pred_len else timestamps),
    )
    predictor = TabPFNTimeSeriesPredictor()
    if hasattr(predictor, "fit_predict"):
        preds = predictor.fit_predict(train_series, prediction_length=pred_len)
    elif hasattr(predictor, "fit") and hasattr(predictor, "predict"):
        predictor.fit(train_series)
        preds = predictor.predict(prediction_length=pred_len)
    else:
        raise RuntimeError("v0.x에서 예측 메서드를 찾을 수 없습니다.")
    if isinstance(preds, pd.Series):
        pred_df = preds.reset_index()
        pred_df.columns = ["timestamp", "target"]
    elif isinstance(preds, pd.DataFrame):
        pred_df = preds.reset_index() if preds.index.name else preds.copy()
        if "timestamp" not in pred_df.columns:
            pred_df = pred_df.rename(columns={pred_df.columns[0]: "timestamp"})
        if "target" not in pred_df.columns and len(pred_df.columns) >= 2:
            pred_df = pred_df.rename(columns={pred_df.columns[1]: "target"})
    else:
        raise RuntimeError(f"예상치 못한 예측 결과 타입: {type(preds)}")
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    return pred_df


def run_tabpfn_forecast(
    close: pd.Series,
    pred_len_bars: int,
    interval: str,
    token: str,
) -> tuple:
    """
    TabPFN-TS 시계열 예측.
    Returns (pred_df, hist_pred_df, error_msg)
    pred_df 컬럼: timestamp, 0.1, 0.25, 0.5, 0.75, 0.9
    """
    (major, minor, _), ver_str = _get_tabpfn_ts_version()
    if major == 0 and minor == 0:
        return None, None, (
            "❌ tabpfn-time-series 패키지를 찾을 수 없습니다.\n"
            "requirements.txt: `tabpfn-time-series>=1.0.9`"
        )

    close_clean = close.dropna()
    ts_tz = close_clean.index
    try:
        timestamps = pd.DatetimeIndex(ts_tz).tz_localize(None)
    except TypeError:
        timestamps = pd.DatetimeIndex(ts_tz).tz_convert(None)
    values = close_clean.values.tolist()

    def _fix_timestamps(pred_df, hist_pred_df):
        """TabPFN 내부 일별 타임스탬프 → 봉 간격 타임스탬프로 교체."""
        future = _future_dates(close_clean.index[-1], len(pred_df), interval)
        pred_df = pred_df.copy()
        pred_df["timestamp"] = future[:len(pred_df)]
        if hist_pred_df is not None and len(hist_pred_df) > 0:
            hist_pred_df = hist_pred_df.copy()
            actual = close_clean.index[-len(hist_pred_df):]
            try:
                hd = pd.DatetimeIndex(actual).tz_localize(None)
            except TypeError:
                hd = pd.DatetimeIndex(actual).tz_convert(None)
            hist_pred_df["timestamp"] = hd[:len(hist_pred_df)]
        return pred_df, hist_pred_df

    if major >= 1:
        try:
            pred_df, hist_pred_df = _run_tabpfn_v1(
                values, timestamps, pred_len_bars, "price", token
            )
            pred_df, hist_pred_df = _fix_timestamps(pred_df, hist_pred_df)
            return pred_df, hist_pred_df, None
        except ImportError as e:
            return None, None, f"❌ 모듈 임포트 실패 (tabpfn-time-series {ver_str}): {e}"
        except Exception as e:
            return None, None, f"❌ 예측 실패 (v{ver_str}): {e}"

    try:
        pred_df = _run_tabpfn_v0(values, timestamps, pred_len_bars, token)
        pred_df, _ = _fix_timestamps(pred_df, None)
        return pred_df, None, None
    except Exception as e:
        return None, None, f"❌ 예측 실패 (tabpfn-time-series {ver_str} 구버전): {e}"


# ── LightGBM 분위수 회귀 예측 ─────────────────────────────────────────────────

def run_lgbm_forecast(
    close: pd.Series,
    hist: pd.DataFrame,
    pred_len: int,
    interval: str,
    period_rsi: int = 14,
) -> tuple:
    """
    LightGBM 분위수 회귀 예측.
    Returns (pred_df, hist_pred_df, error_msg)
    pred_df 컬럼: timestamp, 0.1, 0.25, 0.5, 0.75, 0.9
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return None, None, "❌ lightgbm 미설치. requirements.txt에 `lightgbm>=4.0.0` 추가 필요."

    QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

    close_clean = close.dropna()
    if len(close_clean) < 50:
        return None, None, "❌ 데이터 부족 (최소 50봉 필요)."

    # 피처 생성 (기존 build_features 재사용)
    feat_df = build_features(hist, period_rsi)
    common = close_clean.index.intersection(feat_df.index)
    c = close_clean.loc[common]
    f = feat_df.loc[common]

    # NaN 50% 이상 컬럼 제거 (조회 기간 짧을 때 MA200 등 대응)
    f = f.dropna(axis=1, thresh=max(1, len(f) // 2))
    f = f.fillna(0)

    # 최근 2000봉으로 제한 (인트라데이 대용량 대응)
    if len(c) > 2000:
        c = c.iloc[-2000:]
        f = f.iloc[-2000:]

    c_arr = c.values.astype(np.float64)
    f_arr = f.values.astype(np.float32)
    n = len(c_arr)

    if n < pred_len + 10:
        return None, None, f"❌ 학습 데이터 부족 ({n}봉). 조회 기간을 늘려주세요."

    # Stacked horizon 학습셋 구성
    # y = log(close[t+h] / close[t]), horizon=h 를 피처로 추가
    X_rows, y_rows = [], []
    for h in range(1, pred_len + 1):
        for i in range(n - h):
            if c_arr[i] <= 0:
                continue
            log_ret = np.log(c_arr[i + h] / c_arr[i])
            if not np.isfinite(log_ret):
                continue
            X_rows.append(np.append(f_arr[i], float(h)))
            y_rows.append(log_ret)

    if len(X_rows) < 30:
        return None, None, "❌ 유효 학습 샘플 부족."

    X_train = np.array(X_rows, dtype=np.float32)
    y_train = np.array(y_rows, dtype=np.float32)

    # 5개 분위수 모델 학습
    params_base = dict(
        objective="quantile",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=-1,
        n_jobs=-1,
    )
    models = {}
    for q in QUANTILES:
        m = lgb.LGBMRegressor(**{**params_base, "alpha": q})
        m.fit(X_train, y_train)
        models[q] = m

    # 미래 pred_len 스텝 예측
    last_feat = f_arr[-1].astype(np.float32)
    last_price = float(c_arr[-1])
    pred_rows = []
    for h in range(1, pred_len + 1):
        x_pred = np.append(last_feat, float(h)).reshape(1, -1)
        raw = [float(models[q].predict(x_pred)[0]) for q in QUANTILES]
        raw.sort()  # Quantile crossing 방지
        pred_rows.append([last_price * np.exp(v) for v in raw])

    future = _future_dates(close_clean.index[-1], pred_len, interval)
    pred_df = pd.DataFrame(pred_rows, columns=["0.1", "0.25", "0.5", "0.75", "0.9"])
    pred_df.insert(0, "timestamp", future[:pred_len])

    # 과거 검증: 마지막 pred_len 구간을 재예측해 실제가와 비교
    hist_pred_df = None
    if n > pred_len + 10:
        hist_rows = []
        start_idx = n - pred_len - 1
        for h in range(1, pred_len + 1):
            idx = start_idx + (h - 1)
            if idx < 0 or idx >= n - 1:
                break
            x_p = np.append(f_arr[idx], float(h)).reshape(1, -1)
            med_ret = float(models[0.5].predict(x_p)[0])
            hist_rows.append(float(c_arr[idx]) * np.exp(med_ret))

        if hist_rows:
            hist_dates = close_clean.index[-pred_len: -pred_len + len(hist_rows)]
            if len(hist_dates) == 0:
                hist_dates = close_clean.index[-len(hist_rows):]
            try:
                hd = pd.DatetimeIndex(hist_dates).tz_localize(None)
            except TypeError:
                hd = pd.DatetimeIndex(hist_dates).tz_convert(None)
            hist_pred_df = pd.DataFrame({
                "timestamp": hd[:len(hist_rows)],
                "0.5": hist_rows,
            })

    return pred_df, hist_pred_df, None


# ── Trend Vision / Multi Lens 지표 헬퍼 ──────────────────────────────────────

def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14, multiplier: float = 3.0):
    """ATR 기반 Supertrend 계산. Returns (supertrend_series, is_up_series)."""
    n = len(close)
    h = high.values.astype(float)
    l = low.values.astype(float)
    c = close.values.astype(float)

    # True Range
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

    # Wilder ATR
    alpha = 1.0 / period
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    hl2 = (h + l) / 2
    raw_upper = hl2 + multiplier * atr
    raw_lower = hl2 - multiplier * atr

    upper = raw_upper.copy()
    lower = raw_lower.copy()
    supertrend = np.zeros(n)
    is_up_arr = np.ones(n, dtype=bool)

    supertrend[0] = lower[0]
    for i in range(1, n):
        # Adjust lower band (only moves up)
        lower[i] = raw_lower[i] if (raw_lower[i] > lower[i - 1] or c[i - 1] < lower[i - 1]) else lower[i - 1]
        # Adjust upper band (only moves down)
        upper[i] = raw_upper[i] if (raw_upper[i] < upper[i - 1] or c[i - 1] > upper[i - 1]) else upper[i - 1]

        if is_up_arr[i - 1]:
            is_up_arr[i] = c[i] >= lower[i]
        else:
            is_up_arr[i] = c[i] > upper[i]

        supertrend[i] = lower[i] if is_up_arr[i] else upper[i]

    return (
        pd.Series(supertrend, index=close.index),
        pd.Series(is_up_arr, index=close.index),
    )


def _add_bg_bands(fig, dates, is_up: pd.Series, row: int = 1, col: int = 1):
    """연속 상승/하락 구간을 초록/빨간 배경 vrect으로 추가."""
    date_list = list(dates)
    n = len(date_list)
    if n == 0:
        return
    i = 0
    while i < n:
        color = "rgba(0,160,0,0.13)" if bool(is_up.iloc[i]) else "rgba(200,0,0,0.13)"
        j = i + 1
        while j < n and bool(is_up.iloc[j]) == bool(is_up.iloc[i]):
            j += 1
        fig.add_vrect(
            x0=date_list[i], x1=date_list[j - 1],
            fillcolor=color, opacity=1.0, layer="below", line_width=0,
            row=row, col=col,
        )
        i = j


def render_trend_vision(ticker: str, hist: pd.DataFrame):
    """주봉 Trend Vision: Supertrend 배경 + MACD 기반 UP/DOWN/BOTTOM 신호 + 요약."""
    close = hist["Close"]
    high  = hist["High"]
    low   = hist["Low"]
    dates = close.index

    # Supertrend (주봉 기본: period=20, multiplier=2.5)
    st_line, is_up = _supertrend(high, low, close, period=20, multiplier=2.5)

    # MACD
    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    sig_line  = macd_line.ewm(span=9, adjust=False).mean()
    hist_macd = macd_line - sig_line

    # RSI (Wilder 14)
    delta = close.diff()
    ag = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + ag / al.replace(0, np.nan))

    # MA
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    # 신호 감지 (너무 촘촘하지 않게 최소 간격 적용)
    up_raw     = (hist_macd > 0) & (hist_macd.shift(1) <= 0) & is_up
    down_raw   = (hist_macd < 0) & (hist_macd.shift(1) >= 0) & (~is_up)
    bottom_raw = (~is_up.shift(1).fillna(True)) & is_up & (rsi < 40)

    def _filter_min_gap(mask: pd.Series, gap: int = 4) -> pd.Series:
        result = mask.copy()
        last = -gap
        for i, (idx, v) in enumerate(mask.items()):
            if v:
                if i - last < gap:
                    result[idx] = False
                else:
                    last = i
        return result

    up_sig     = _filter_min_gap(up_raw, gap=4)
    down_sig   = _filter_min_gap(down_raw, gap=4)
    bottom_sig = _filter_min_gap(bottom_raw, gap=4)

    # ── 차트 (캔들 + MACD 하단)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    _add_bg_bands(fig, dates, is_up, row=1, col=1)

    fig.add_trace(go.Candlestick(
        x=dates, open=hist["Open"], high=high, low=low, close=close,
        name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=ma20, name="MA20",
                             line=dict(color="#f59e0b", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ma60, name="MA60",
                             line=dict(color="#60a5fa", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=st_line, name="Supertrend",
                             line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dot"),
                             showlegend=False), row=1, col=1)

    # 신호 어노테이션
    for idx in dates[up_sig]:
        fig.add_annotation(x=idx, y=float(low[idx]) * 0.993,
                           text="▲ UP", showarrow=False,
                           font=dict(color="#00e676", size=10, family="Arial Black"),
                           row=1, col=1)
    for idx in dates[down_sig]:
        fig.add_annotation(x=idx, y=float(high[idx]) * 1.007,
                           text="▼ DOWN", showarrow=False,
                           font=dict(color="#ff5252", size=10, family="Arial Black"),
                           row=1, col=1)
    for idx in dates[bottom_sig]:
        fig.add_annotation(x=idx, y=float(low[idx]) * 0.986,
                           text="◆ BOTTOM", showarrow=False,
                           font=dict(color="#ffd700", size=11, family="Arial Black"),
                           row=1, col=1)

    # MACD 패널
    bar_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in hist_macd.fillna(0)]
    fig.add_trace(go.Bar(x=dates, y=hist_macd, marker_color=bar_colors,
                         name="MACD Hist", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=macd_line, name="MACD",
                             line=dict(color="#60a5fa", width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=sig_line, name="Signal",
                             line=dict(color="#f59e0b", width=1), showlegend=False), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{ticker} — Trend Vision (주봉)", font=dict(size=15)),
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0d1117",
        font_color="white", height=560,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
        margin=dict(t=55, b=10, l=10, r=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1e1e2e")

    # Streamlit 열: 차트(좌) + 요약(우)
    chart_col, sum_col = st.columns([5, 1])
    with chart_col:
        st.plotly_chart(fig, use_container_width=True)
    with sum_col:
        st.markdown("#### 현재 상태")
        trend_txt = "📈 상승" if bool(is_up.iloc[-1]) else "📉 하락"
        st.metric("추세", trend_txt)
        st.metric("RSI(14)", f"{rsi.iloc[-1]:.1f}")
        macd_val = macd_line.iloc[-1]
        st.metric("MACD", f"{macd_val:.3g}", delta=f"{hist_macd.iloc[-1]:.3g}")
        if not np.isnan(ma20.iloc[-1]):
            st.metric("vs MA20", f"{(close.iloc[-1]/ma20.iloc[-1]-1)*100:+.2f}%")
        if not np.isnan(ma60.iloc[-1]):
            st.metric("vs MA60", f"{(close.iloc[-1]/ma60.iloc[-1]-1)*100:+.2f}%")


def render_multi_lens(ticker: str, hist: pd.DataFrame):
    """일봉 Multi Lens: Supertrend 배경 + 다중신호 스코어 + 6개 지표 패널."""
    close  = hist["Close"]
    high   = hist["High"]
    low    = hist["Low"]
    volume = hist.get("Volume", pd.Series(np.nan, index=close.index))
    dates  = close.index

    # Supertrend (일봉: period=14, multiplier=3.0)
    st_line, is_up = _supertrend(high, low, close, period=14, multiplier=3.0)

    # RSI
    delta = close.diff()
    ag = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + ag / al.replace(0, np.nan))

    # MACD
    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    sig_line  = macd_line.ewm(span=9, adjust=False).mean()
    hist_macd = macd_line - sig_line

    # Bollinger Bands
    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_u  = ma20 + 2 * std20
    bb_l  = ma20 - 2 * std20
    bb_pct = ((close - bb_l) / (bb_u - bb_l).replace(0, np.nan)).clip(-0.5, 1.5)

    # Stochastic
    lo14    = low.rolling(14).min()
    hi14    = high.rolling(14).max()
    stoch_k = ((close - lo14) / (hi14 - lo14).replace(0, np.nan) * 100).clip(0, 100)
    stoch_d = stoch_k.rolling(3).mean()

    # CCI
    tp    = (high + low + close) / 3
    ma_tp = tp.rolling(20).mean()
    md_tp = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci   = (tp - ma_tp) / (0.015 * md_tp.replace(0, np.nan))

    # ADX / ±DI
    h_diff   = high.diff()
    l_diff   = -low.diff()
    plus_dm  = pd.Series(np.where((h_diff > l_diff) & (h_diff > 0), h_diff, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((l_diff > h_diff) & (l_diff > 0), l_diff, 0.0), index=close.index)
    prev_c   = close.shift(1)
    tr_raw   = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr14    = tr_raw.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx      = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # MA60, Volume ratio
    ma60     = close.rolling(60).mean()
    vol_ma20 = volume.rolling(20).mean().replace(0, np.nan)
    vol_ratio = (volume / vol_ma20).replace([np.inf, -np.inf], np.nan)

    # 다중신호 스코어 (-7 ~ +7)
    score = (
        np.sign(rsi - 50).fillna(0)
        + np.sign(hist_macd).fillna(0)
        + np.sign(stoch_k - 50).fillna(0)
        + np.sign(close / ma20.replace(0, np.nan) - 1).fillna(0)
        + np.sign(close / ma60.replace(0, np.nan) - 1).fillna(0)
        + np.sign(bb_pct - 0.5).fillna(0)
        + np.sign(vol_ratio - 1).fillna(0)
    )

    # 신호: Supertrend 전환 또는 스코어 임계 돌파
    up_flip    = is_up & (~is_up.shift(1).fillna(True))
    dn_flip    = (~is_up) & (is_up.shift(1).fillna(False))
    score_up   = (score >= 4) & (score.shift(1).fillna(0) < 4)
    score_dn   = (score <= -4) & (score.shift(1).fillna(0) > -4)
    up_signals = (up_flip | score_up)
    dn_signals = (dn_flip | score_dn)

    def _filter_min_gap(mask: pd.Series, gap: int = 5) -> pd.Series:
        result = mask.copy()
        last = -gap
        for i, (idx, v) in enumerate(mask.items()):
            if v:
                if i - last < gap:
                    result[idx] = False
                else:
                    last = i
        return result

    up_signals = _filter_min_gap(up_signals, gap=5)
    dn_signals = _filter_min_gap(dn_signals, gap=5)

    # ── 서브플롯 (캔들 + RSI + MACD + BB%B + Stoch + CCI + ADX)
    fig = make_subplots(
        rows=7, cols=1,
        shared_xaxes=True,
        row_heights=[0.36, 0.10, 0.10, 0.10, 0.10, 0.12, 0.12],
        subplot_titles=["", "RSI(14)", "MACD", "Bollinger %B", "Stochastic(14)", "CCI(20)", "ADX(14)"],
        vertical_spacing=0.025,
    )

    _add_bg_bands(fig, dates, is_up, row=1, col=1)

    # 캔들
    fig.add_trace(go.Candlestick(
        x=dates, open=hist["Open"], high=high, low=low, close=close,
        name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=ma20, name="MA20",
                             line=dict(color="#f59e0b", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ma60, name="MA60",
                             line=dict(color="#60a5fa", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bb_u, name="BB",
                             line=dict(color="rgba(147,51,234,0.55)", width=1, dash="dot"),
                             showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bb_l,
                             line=dict(color="rgba(147,51,234,0.55)", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(147,51,234,0.05)",
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=st_line, name="ST",
                             line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
                             showlegend=False), row=1, col=1)

    # UP/DOWN 어노테이션 (font size ∝ |score|)
    for idx in dates[up_signals]:
        sz = int(np.clip(9 + abs(score[idx]), 9, 16))
        fig.add_annotation(x=idx, y=float(low[idx]) * 0.995,
                           text="▲ UP", showarrow=False,
                           font=dict(color="#00e676", size=sz, family="Arial Black"),
                           row=1, col=1)
    for idx in dates[dn_signals]:
        sz = int(np.clip(9 + abs(score[idx]), 9, 16))
        fig.add_annotation(x=idx, y=float(high[idx]) * 1.005,
                           text="▼ DOWN", showarrow=False,
                           font=dict(color="#ff5252", size=sz, family="Arial Black"),
                           row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=dates, y=rsi, line=dict(color="#a78bfa", width=1.5),
                             name="RSI", showlegend=False), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.5)",  row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(38,166,154,0.5)", row=2, col=1)

    # MACD
    bar_clr = ["#26a69a" if v >= 0 else "#ef5350" for v in hist_macd.fillna(0)]
    fig.add_trace(go.Bar(x=dates, y=hist_macd, marker_color=bar_clr,
                         name="Hist", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=macd_line, line=dict(color="#60a5fa", width=1),
                             name="MACD", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=sig_line, line=dict(color="#f59e0b", width=1),
                             name="Signal", showlegend=False), row=3, col=1)

    # Bollinger %B
    fig.add_trace(go.Scatter(x=dates, y=bb_pct, line=dict(color="#c084fc", width=1.5),
                             name="BB%B", showlegend=False), row=4, col=1)
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(239,83,80,0.5)",  row=4, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=4, col=1)
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(38,166,154,0.5)", row=4, col=1)

    # Stochastic
    fig.add_trace(go.Scatter(x=dates, y=stoch_k, line=dict(color="#f472b6", width=1.5),
                             name="%K", showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=dates, y=stoch_d, line=dict(color="#fb923c", width=1),
                             name="%D", showlegend=False), row=5, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="rgba(239,83,80,0.5)",  row=5, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="rgba(38,166,154,0.5)", row=5, col=1)

    # CCI
    cci_clr = ["#26a69a" if v >= 0 else "#ef5350" for v in cci.fillna(0)]
    fig.add_trace(go.Bar(x=dates, y=cci, marker_color=cci_clr,
                         name="CCI", showlegend=False), row=6, col=1)
    fig.add_hline(y=100,  line_dash="dot", line_color="rgba(239,83,80,0.5)",  row=6, col=1)
    fig.add_hline(y=-100, line_dash="dot", line_color="rgba(38,166,154,0.5)", row=6, col=1)

    # ADX
    fig.add_trace(go.Scatter(x=dates, y=adx,      line=dict(color="#fbbf24", width=1.8),
                             name="ADX", showlegend=False), row=7, col=1)
    fig.add_trace(go.Scatter(x=dates, y=plus_di,  line=dict(color="#34d399", width=1),
                             name="+DI", showlegend=False), row=7, col=1)
    fig.add_trace(go.Scatter(x=dates, y=minus_di, line=dict(color="#f87171", width=1),
                             name="-DI", showlegend=False), row=7, col=1)
    fig.add_hline(y=25, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=7, col=1)

    fig.update_layout(
        title=dict(text=f"{ticker} — Multi Lens (일봉)", font=dict(size=15)),
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0d1117",
        font_color="white", height=920,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.01, x=0, font=dict(size=11)),
        margin=dict(t=55, b=10, l=10, r=10),
    )
    for r in range(1, 8):
        fig.update_xaxes(showgrid=False, row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#1e1e2e", row=r, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=5, col=1)
    fig.update_xaxes(rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)


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

    # ── Trend Vision / Multi Lens 탭 ─────────────────────────────────────────
    st.divider()
    tab_tv, tab_ml = st.tabs(["📈 Trend Vision (주봉)", "📉 Multi Lens (일봉)"])

    with tab_tv:
        with st.spinner("Trend Vision 차트 생성 중..."):
            if interval == "1wk":
                hist_tv = hist
            else:
                hist_tv = yf.Ticker(resolved).history(period="5y", interval="1wk")
            if hist_tv.empty:
                st.warning("주봉 데이터를 가져올 수 없습니다.")
            else:
                render_trend_vision(resolved, hist_tv)

    with tab_ml:
        with st.spinner("Multi Lens 차트 생성 중..."):
            if interval == "1d":
                hist_ml = hist
            else:
                hist_ml = yf.Ticker(resolved).history(period="2y", interval="1d")
            if hist_ml.empty:
                st.warning("일봉 데이터를 가져올 수 없습니다.")
            else:
                render_multi_lens(resolved, hist_ml)

    # ── TabPFN-TS + LightGBM 가격 예측 ──────────────────────────────────────
    with st.expander("🔮 TabPFN-TS + LightGBM 가격 예측 (AI)"):
        col_h, col_btn = st.columns([3, 1])
        horizon_opts_bars = {"4봉": 4, "8봉": 8, "12봉": 12, "26봉": 26, "52봉": 52}
        horizon_label = col_h.selectbox("예측 기간 (봉 수)", list(horizon_opts_bars.keys()), index=2)
        horizon_bars = horizon_opts_bars[horizon_label]
        run_forecast = col_btn.button("예측 실행", type="primary", use_container_width=True)

        if run_forecast:
            # ── LightGBM 예측 (항상 실행)
            lgbm_pred_df = lgbm_hist_df = lgbm_err = None
            with st.spinner("LightGBM 예측 중..."):
                lgbm_pred_df, lgbm_hist_df, lgbm_err = run_lgbm_forecast(
                    close, hist, horizon_bars, interval, period_rsi
                )

            # ── TabPFN-TS 예측 (토큰 있을 때만)
            tabpfn_pred_df = tabpfn_hist_df = tabpfn_err = None
            tabpfn_token = None
            try:
                tabpfn_token = st.secrets["TABPFN_API_TOKEN"]
            except KeyError:
                pass
            (ts_major, _, _), ts_ver_str = _get_tabpfn_ts_version()
            if tabpfn_token and ts_major >= 1:
                with st.spinner(f"TabPFN-TS {ts_ver_str} 예측 중..."):
                    tabpfn_pred_df, tabpfn_hist_df, tabpfn_err = run_tabpfn_forecast(
                        close, horizon_bars, interval, tabpfn_token
                    )

            # ── 헬퍼: 예측 테이블 렌더링
            def _render_table(p_df):
                has_q = all(c in p_df.columns for c in ["0.1", "0.9"])
                med = "0.5" if "0.5" in p_df.columns else "target"
                rows = []
                for i, row in p_df.iterrows():
                    ts_  = row.get("timestamp", i)
                    tgt  = float(row.get(med, np.nan))
                    lo   = float(row.get("0.1", np.nan)) if has_q else np.nan
                    hi   = float(row.get("0.9", np.nan)) if has_q else np.nan
                    pct  = (tgt - current_price) / current_price * 100 if np.isfinite(tgt) else np.nan
                    rows.append({
                        "#": i + 1,
                        "날짜": ts_.strftime("%Y-%m-%d") if hasattr(ts_, "strftime") else str(ts_),
                        "예측가 (중앙값)": round(tgt, 4) if np.isfinite(tgt) else None,
                        "하단 10%": round(lo,  4) if np.isfinite(lo)  else None,
                        "상단 90%": round(hi,  4) if np.isfinite(hi)  else None,
                        "등락률 (%)": round(pct, 2) if np.isfinite(pct) else None,
                    })
                st.dataframe(
                    pd.DataFrame(rows).style.format({
                        "예측가 (중앙값)": lambda v: f"{v:,.4g}" if v else "-",
                        "하단 10%":        lambda v: f"{v:,.4g}" if v else "-",
                        "상단 90%":        lambda v: f"{v:,.4g}" if v else "-",
                        "등락률 (%)":      lambda v: f"{v:+.2f}%" if v else "-",
                    }, na_rep="-"),
                    use_container_width=True, hide_index=True,
                )
                return has_q, med

            # ── 헬퍼: 팬차트 트레이스 추가
            def _add_fan(fig, p_df, h_df, line_color, fill_color, fill_color2, model_name):
                has_q = all(c in p_df.columns for c in ["0.1", "0.9"])
                med   = "0.5" if "0.5" in p_df.columns else "target"
                px_   = p_df["timestamp"] if "timestamp" in p_df.columns else p_df.index
                if h_df is not None and "timestamp" in h_df.columns:
                    hm = "0.5" if "0.5" in h_df.columns else "target"
                    if hm in h_df.columns:
                        fig.add_trace(go.Scatter(
                            x=h_df["timestamp"], y=h_df[hm],
                            name=f"{model_name} 과거검증",
                            line=dict(color=line_color, width=1.2, dash="dot"),
                        ))
                if has_q:
                    fig.add_trace(go.Scatter(
                        x=list(px_) + list(px_[::-1]),
                        y=list(p_df["0.9"]) + list(p_df["0.1"][::-1]),
                        fill="toself", fillcolor=fill_color,
                        line=dict(color="rgba(0,0,0,0)"), name=f"{model_name} 10%~90%",
                    ))
                    if "0.25" in p_df.columns:
                        fig.add_trace(go.Scatter(
                            x=list(px_) + list(px_[::-1]),
                            y=list(p_df["0.75"]) + list(p_df["0.25"][::-1]),
                            fill="toself", fillcolor=fill_color2,
                            line=dict(color="rgba(0,0,0,0)"), name=f"{model_name} 25%~75%",
                        ))
                fig.add_trace(go.Scatter(
                    x=px_, y=p_df[med],
                    name=f"{model_name} 중앙값",
                    line=dict(color=line_color, width=2, dash="dash"),
                ))
                return px_

            hist_show = close.iloc[-120:]

            # ── 탭별 표시
            tab_lgbm, tab_tabpfn = st.tabs(["📊 LightGBM", "🌐 TabPFN-TS"])

            with tab_lgbm:
                if lgbm_err:
                    st.error(lgbm_err)
                elif lgbm_pred_df is None:
                    st.error("LightGBM 예측 결과가 없습니다.")
                else:
                    _render_table(lgbm_pred_df)
                    fig_l = go.Figure()
                    fig_l.add_trace(go.Scatter(x=hist_show.index, y=hist_show.values,
                                               name="실제가", line=dict(color="#ffffff", width=1.5)))
                    _lx = _add_fan(fig_l, lgbm_pred_df, lgbm_hist_df,
                                   "#64a0ff", "rgba(100,160,255,0.15)", "rgba(100,160,255,0.30)", "LightGBM")
                    fig_l.add_shape(type="line",
                                    x0=close.index[-1],
                                    x1=_lx.iloc[0] if hasattr(_lx, "iloc") else _lx[0],
                                    y0=current_price, y1=current_price,
                                    line=dict(color="#ffeb3b", width=1, dash="dot"))
                    fig_l.update_layout(template="plotly_dark", paper_bgcolor="#000000",
                                        plot_bgcolor="#000000", height=420,
                                        title=f"LightGBM 예측 ({horizon_bars}봉)",
                                        margin=dict(l=10, r=10, t=40, b=10),
                                        legend=dict(orientation="h"))
                    st.plotly_chart(fig_l, use_container_width=True)

            with tab_tabpfn:
                if not tabpfn_token:
                    st.info("Streamlit Secrets에 `TABPFN_API_TOKEN`을 추가하면 TabPFN-TS 예측을 사용할 수 있습니다.")
                elif ts_major < 1:
                    st.warning(f"tabpfn-time-series {ts_ver_str} — v1.0.9 이상 필요.")
                elif tabpfn_err:
                    st.error(tabpfn_err)
                elif tabpfn_pred_df is None:
                    st.error("TabPFN-TS 예측 결과가 없습니다.")
                else:
                    _render_table(tabpfn_pred_df)
                    fig_t = go.Figure()
                    fig_t.add_trace(go.Scatter(x=hist_show.index, y=hist_show.values,
                                               name="실제가", line=dict(color="#ffffff", width=1.5)))
                    _tx = _add_fan(fig_t, tabpfn_pred_df, tabpfn_hist_df,
                                   "#69f0ae", "rgba(100,255,160,0.15)", "rgba(100,255,160,0.30)", "TabPFN-TS")
                    fig_t.add_shape(type="line",
                                    x0=close.index[-1],
                                    x1=_tx.iloc[0] if hasattr(_tx, "iloc") else _tx[0],
                                    y0=current_price, y1=current_price,
                                    line=dict(color="#ffeb3b", width=1, dash="dot"))
                    fig_t.update_layout(template="plotly_dark", paper_bgcolor="#000000",
                                        plot_bgcolor="#000000", height=420,
                                        title=f"TabPFN-TS 예측 ({horizon_bars}봉)",
                                        margin=dict(l=10, r=10, t=40, b=10),
                                        legend=dict(orientation="h"))
                    st.plotly_chart(fig_t, use_container_width=True)

            # ── 두 모델 모두 성공 시 통합 비교 차트
            if lgbm_pred_df is not None and tabpfn_pred_df is not None and not tabpfn_err:
                st.markdown("#### 📈 LightGBM vs TabPFN-TS 비교")
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(x=hist_show.index, y=hist_show.values,
                                             name="실제가", line=dict(color="#ffffff", width=1.5)))
                _lx2 = _add_fan(fig_cmp, lgbm_pred_df, None,
                                 "#64a0ff", "rgba(100,160,255,0.12)", "rgba(100,160,255,0.22)", "LightGBM")
                _add_fan(fig_cmp, tabpfn_pred_df, None,
                         "#69f0ae", "rgba(100,255,160,0.12)", "rgba(100,255,160,0.22)", "TabPFN-TS")
                fig_cmp.add_shape(type="line",
                                  x0=close.index[-1],
                                  x1=_lx2.iloc[0] if hasattr(_lx2, "iloc") else _lx2[0],
                                  y0=current_price, y1=current_price,
                                  line=dict(color="#ffeb3b", width=1, dash="dot"))
                fig_cmp.update_layout(template="plotly_dark", paper_bgcolor="#000000",
                                      plot_bgcolor="#000000", height=450,
                                      title=f"LightGBM vs TabPFN-TS ({horizon_bars}봉)",
                                      margin=dict(l=10, r=10, t=40, b=10),
                                      legend=dict(orientation="h"))
                st.plotly_chart(fig_cmp, use_container_width=True)

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
