import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="RSI 타겟 가격 계산기",
    page_icon="📈",
    layout="wide",
)

st.title("📈 RSI 타겟 가격 계산기")
st.caption("종목/지수를 입력하면 현재 RSI와 타겟 RSI 달성에 필요한 가격 및 등락률을 계산합니다.")


def calculate_rsi(prices: pd.Series, period: int = 14):
    """단순 이동평균(SMA) 방식으로 RSI 계산."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # 다음 봉 역산에 필요한 현재 윈도우의 가장 오래된 값
    oldest_gain = gain.shift(period - 1)
    oldest_loss = loss.shift(period - 1)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi, avg_gain, avg_loss, oldest_gain, oldest_loss


def target_price_for_rsi(
    current_price: float,
    avg_gain: float,
    avg_loss: float,
    oldest_gain: float,
    oldest_loss: float,
    target_rsi: float,
    period: int = 14,
) -> float | None:
    """
    SMA 방식에서 다음 봉 하나로 target_rsi 에 도달하는 가격을 역산.

    SMA rolling: avg_new = (avg_prev * n - oldest + new_val) / n

    target_rs = target_rsi / (100 - target_rsi)
    sum_gain = avg_gain * n,  sum_loss = avg_loss * n

    [상승 케이스] new_gain = P - current_price, new_loss = 0
        avg_gain_new = (sum_gain - oldest_gain + P - current_price) / n
        avg_loss_new = (sum_loss - oldest_loss) / n
        -> P = current_price + target_rs*(sum_loss - oldest_loss) - sum_gain + oldest_gain

    [하락 케이스] new_gain = 0, new_loss = current_price - P
        avg_gain_new = (sum_gain - oldest_gain) / n
        avg_loss_new = (sum_loss - oldest_loss + current_price - P) / n
        -> P = current_price + sum_loss - oldest_loss - (sum_gain - oldest_gain) / target_rs
    """
    if target_rsi <= 0 or target_rsi >= 100:
        return None

    n = period
    target_rs = target_rsi / (100 - target_rsi)

    sum_gain = avg_gain * n
    sum_loss = avg_loss * n

    # 현재 RSI
    if avg_loss == 0:
        current_rsi = 100.0
    elif avg_gain == 0:
        current_rsi = 0.0
    else:
        current_rsi = 100 - 100 / (1 + avg_gain / avg_loss)

    if target_rsi >= current_rsi:
        # 상승 케이스
        price = current_price + target_rs * (sum_loss - oldest_loss) - sum_gain + oldest_gain
    else:
        # 하락 케이스
        if target_rs == 0:
            return None
        price = current_price + (sum_loss - oldest_loss) - (sum_gain - oldest_gain) / target_rs

    return price


# ── 사이드바: 입력 ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("설정")

    ticker_input = st.text_input(
        "종목/지수 심볼",
        value="^GSPC",
        help="예) AAPL, TSLA, ^GSPC (S&P500), ^KS11 (KOSPI), 005930.KS (삼성전자)",
    )

    period_rsi = st.number_input(
        "RSI 기간 (봉 수)",
        min_value=2,
        max_value=50,
        value=14,
        step=1,
    )

    interval_options = {
        "일봉": "1d",
        "주봉": "1wk",
        "월봉": "1mo",
        "1시간": "1h",
        "4시간": "4h",
        "15분": "15m",
    }
    interval_label = st.selectbox("차트 간격", list(interval_options.keys()), index=0)
    interval = interval_options[interval_label]

    lookback_options = {
        "3개월": "3mo",
        "6개월": "6mo",
        "1년": "1y",
        "2년": "2y",
        "5년": "5y",
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

# ── 메인: 계산 및 출력 ──────────────────────────────────────────────────────────
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
            ticker_obj = yf.Ticker(ticker_input)
            hist = ticker_obj.history(period=lookback, interval=interval)
        except Exception as e:
            st.error(f"데이터 로딩 실패: {e}")
            st.stop()

    if hist.empty:
        st.error(f"'{ticker_input}' 에 대한 데이터를 가져오지 못했습니다. 심볼을 확인해주세요.")
        st.stop()

    close = hist["Close"].dropna()

    if len(close) < period_rsi + 1:
        st.error(f"데이터가 부족합니다. 최소 {period_rsi + 1}개 봉이 필요합니다.")
        st.stop()

    rsi_series, avg_gain_series, avg_loss_series, oldest_gain_series, oldest_loss_series = calculate_rsi(close, period=period_rsi)

    current_price = float(close.iloc[-1])
    current_rsi = float(rsi_series.iloc[-1])
    last_avg_gain = float(avg_gain_series.iloc[-1])
    last_avg_loss = float(avg_loss_series.iloc[-1])
    last_oldest_gain = float(oldest_gain_series.iloc[-1])
    last_oldest_loss = float(oldest_loss_series.iloc[-1])
    last_date = close.index[-1]

    # ── 현재 상태 카드 ─────────────────────────────────────────────────────────
    st.subheader(f"{ticker_input.upper()}  |  {interval_label}  |  {last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else last_date}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("현재 가격", f"{current_price:,.4g}")
    with col2:
        rsi_color = (
            "🔴 과매수" if current_rsi >= 70
            else "🟢 과매도" if current_rsi <= 30
            else "⚪ 중립"
        )
        st.metric("현재 RSI", f"{current_rsi:.2f}", rsi_color)
    with col3:
        st.metric("RSI 기간", f"{period_rsi}봉")

    # ── 타겟 RSI 테이블 ────────────────────────────────────────────────────────
    st.subheader("타겟 RSI별 예상 가격")

    rows = []
    for t_rsi in target_rsi_list:
        t_price = target_price_for_rsi(
            current_price, last_avg_gain, last_avg_loss,
            last_oldest_gain, last_oldest_loss, t_rsi, period=period_rsi
        )
        if t_price is None or t_price <= 0:
            rows.append(
                {
                    "타겟 RSI": t_rsi,
                    "예상 가격": "계산 불가",
                    "등락률 (%)": "-",
                    "방향": "-",
                }
            )
        else:
            pct = (t_price - current_price) / current_price * 100
            direction = "▲ 상승" if t_price > current_price else ("▼ 하락" if t_price < current_price else "─ 현재")
            rows.append(
                {
                    "타겟 RSI": t_rsi,
                    "예상 가격": round(t_price, 4),
                    "등락률 (%)": round(pct, 2),
                    "방향": direction,
                }
            )

    df_table = pd.DataFrame(rows)

    def highlight_row(row):
        rsi = row["타겟 RSI"]
        diff = abs(rsi - current_rsi)
        if diff < 1e-6:
            return ["background-color: #444"] * len(row)
        if rsi <= 30:
            return ["color: #4fc3f7"] * len(row)
        if rsi >= 70:
            return ["color: #ef9a9a"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_table.style.apply(highlight_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # ── 차트 ──────────────────────────────────────────────────────────────────
    st.subheader("차트")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
    )

    # 가격 캔들차트
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name="가격",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    # 타겟 가격 수평선
    for row in rows:
        if isinstance(row["예상 가격"], (int, float)):
            t_rsi = row["타겟 RSI"]
            t_price = row["예상 가격"]
            color = "#4fc3f7" if t_rsi <= 30 else ("#ef9a9a" if t_rsi >= 70 else "#bdbdbd")
            fig.add_hline(
                y=t_price,
                line_dash="dot",
                line_color=color,
                line_width=1,
                annotation_text=f"RSI {t_rsi}",
                annotation_position="right",
                row=1,
                col=1,
            )

    # 현재 가격 수평선
    fig.add_hline(
        y=current_price,
        line_dash="solid",
        line_color="#ffeb3b",
        line_width=1.5,
        annotation_text=f"현재 {current_price:,.4g}",
        annotation_position="right",
        row=1,
        col=1,
    )

    # RSI 라인
    fig.add_trace(
        go.Scatter(
            x=rsi_series.index,
            y=rsi_series.values,
            line=dict(color="#ce93d8", width=1.5),
            name=f"RSI({period_rsi})",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#ef9a9a", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#4fc3f7", line_width=1, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#9e9e9e", line_width=1, row=2, col=1)

    # 현재 RSI 표시
    fig.add_hline(
        y=current_rsi,
        line_dash="solid",
        line_color="#ffeb3b",
        line_width=1,
        annotation_text=f"현재 RSI {current_rsi:.1f}",
        annotation_position="right",
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=80, t=30, b=10),
    )
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)

    # ── 계산 방식 설명 ─────────────────────────────────────────────────────────
    with st.expander("계산 방식 설명"):
        st.markdown(
            f"""
**RSI 계산**: 단순 이동평균 (SMA, 기간 = {period_rsi})

$$RSI = 100 - \\frac{{100}}{{1 + RS}}, \\quad RS = \\frac{{SMA_{{Gain}}(n)}}{{SMA_{{Loss}}(n)}}$$

**타겟 가격 역산**: SMA rolling 윈도우에서 가장 오래된 값($G_{{old}}, L_{{old}}$)이 빠지고 새 값이 들어오는 구조를 이용합니다.

- 상승 케이스 (P > 현재가):
$$P = P_{{cur}} + RS_{{target}} \\cdot (\\Sigma L - L_{{old}}) - \\Sigma G + G_{{old}}$$

- 하락 케이스 (P < 현재가):
$$P = P_{{cur}} + (\\Sigma L - L_{{old}}) - \\frac{{\\Sigma G - G_{{old}}}}{{RS_{{target}}}}$$

여기서 $RS_{{target}} = \\dfrac{{RSI_{{target}}}}{{100 - RSI_{{target}}}}$, $\\Sigma G = SMA_{{Gain}} \\times n$, $\\Sigma L = SMA_{{Loss}} \\times n$

> **주의**: 이 계산은 단일 봉 기준이며, 실제 가격은 시장 상황에 따라 달라집니다.
"""
        )
