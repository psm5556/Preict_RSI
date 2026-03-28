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

st.set_page_config(
    page_title="RSI 타겟 가격 계산기",
    page_icon="📈",
    layout="wide",
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


# ── 사이드바 ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("설정")

    ticker_input = st.text_input(
        "종목/지수 심볼",
        value="^GSPC",
        help="예) AAPL, TSLA, ^GSPC (S&P500), ^KS11 (KOSPI), 005930.KS (삼성전자)",
    )

    period_rsi = st.number_input(
        "RSI 기간 (봉 수)", min_value=2, max_value=50, value=14, step=1,
    )

    interval_options = {
        "일봉": "1d", "주봉": "1wk", "월봉": "1mo",
        "1시간": "1h", "4시간": "4h", "15분": "15m",
    }
    interval_label = st.selectbox("차트 간격", list(interval_options.keys()), index=0)
    interval = interval_options[interval_label]

    lookback_options = {
        "3개월": "3mo", "6개월": "6mo", "1년": "1y", "2년": "2y", "5년": "5y",
    }
    lookback_label = st.selectbox("조회 기간", list(lookback_options.keys()), index=1)
    lookback = lookback_options[lookback_label]

    st.divider()
    st.subheader("타겟 RSI 목록")
    target_rsi_str = st.text_input(
        "쉼표로 구분 입력",
        value="20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80",
    )

    run_btn = st.button("계산하기", type="primary", use_container_width=True)

# ── 메인 ──────────────────────────────────────────────────────────────────────

if run_btn or ticker_input:
    try:
        target_rsi_list = sorted(
            {float(x.strip()) for x in target_rsi_str.split(",") if x.strip()}
        )
    except ValueError:
        st.error("타겟 RSI 목록에 숫자만 입력해주세요.")
        st.stop()

    with st.spinner(f"{ticker_input} 데이터 로딩 중..."):
        try:
            hist = yf.Ticker(ticker_input).history(period=lookback, interval=interval)
        except Exception as e:
            st.error(f"데이터 로딩 실패: {e}")
            st.stop()

    if hist.empty:
        st.error(f"'{ticker_input}' 데이터를 가져오지 못했습니다. 심볼을 확인해주세요.")
        st.stop()

    close = hist["Close"].dropna()

    if len(close) < period_rsi + 1:
        st.error(f"데이터 부족. 최소 {period_rsi + 1}개 봉이 필요합니다.")
        st.stop()

    # 세 방식 계산
    rsi_w, ag_w, al_w = calc_rsi_wilder(close, period_rsi)
    rsi_s, ag_s, al_s, og_s, ol_s = calc_rsi_sma(close, period_rsi)
    rsi_e, ag_e, al_e = calc_rsi_ema(close, period_rsi)

    current_price = float(close.iloc[-1])
    last_date = close.index[-1]
    date_str = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)

    cr_w = float(rsi_w.iloc[-1])
    cr_s = float(rsi_s.iloc[-1])
    cr_e = float(rsi_e.iloc[-1])

    # ── 헤더 ────────────────────────────────────────────────────────────────
    st.subheader(f"{ticker_input.upper()}  |  {interval_label}  |  {date_str}")

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
    tab_w, tab_s, tab_e = st.tabs(
        ["📊 Wilder (표준)", "📊 Cutler (SMA)", "📊 EMA RSI"]
    )

    with tab_w:
        df_w = build_table(
            target_rsi_list, current_price, cr_w,
            lambda t: target_wilder(current_price, float(ag_w.iloc[-1]), float(al_w.iloc[-1]), t, period_rsi),
        )
        st.dataframe(style_table(df_w), use_container_width=True, hide_index=True)

    with tab_s:
        df_s = build_table(
            target_rsi_list, current_price, cr_s,
            lambda t: target_sma(
                current_price, float(ag_s.iloc[-1]), float(al_s.iloc[-1]),
                float(og_s.iloc[-1]), float(ol_s.iloc[-1]), t, period_rsi,
            ),
        )
        st.dataframe(style_table(df_s), use_container_width=True, hide_index=True)

    with tab_e:
        df_e = build_table(
            target_rsi_list, current_price, cr_e,
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

    # 현재가 수평선
    fig.add_hline(
        y=current_price, line_dash="solid", line_color="#ffeb3b", line_width=1.5,
        annotation_text=f"현재 {current_price:,.4g}", annotation_position="right",
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
        height=750,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=80, t=30, b=10),
    )
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)

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
                        template="plotly_dark", height=300,
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
