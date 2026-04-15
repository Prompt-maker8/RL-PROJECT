from evalution_model import get_rl_signal
from real_news_engine import get_real_sentiment
from trend_analyzer import detect_regime
import yfinance as yf
import pandas as pd
import numpy as np

def regime_bias(regime):
    return {"BULL":0.15,"BEAR":-0.15}.get(regime,0)

def map_decision(score):
    if score > 0.3: return "BUY"
    elif score < -0.3: return "SELL"
    return "HOLD"

from sb3_contrib import RecurrentPPO

model = None

def fuse_signals(stock):
    global model
    if model is None:
        model = RecurrentPPO.load("global_macro_agent.zip")

    rl = get_rl_signal(stock, model)
    news = get_real_sentiment(stock, mode="today")

    df = yf.download(stock, period="6mo")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    regime = detect_regime(df)

    # dynamic weights
    if regime == "BEAR":
        w_rl, w_sent = 0.5, 0.3
    elif regime == "BULL":
        w_rl, w_sent = 0.7, 0.15
    else:
        w_rl, w_sent = 0.6, 0.25

    raw_score = (
        w_rl * rl["signal"] * rl["confidence"]
        + w_sent * news["sentiment"]
        + regime_bias(regime)
    )

    # Nonlinear bounded scaling
    score = float(np.tanh(raw_score * 1.5))

    return {
        "stock": stock,
        "final_score": round(score,4),
        "decision": map_decision(score),
        "confidence": abs(score),
        "regime": regime,
        "sentiment": news["sentiment"],
        "headlines": news["headlines"],
        "rl_signal": rl["signal"]
    }