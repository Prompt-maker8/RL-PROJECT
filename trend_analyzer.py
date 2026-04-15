import numpy as np


def analyze_trend(history, window=10):
    """
    Analyzes trend of final_score over recent history.
    Uses normalized slope for stability.
    """

    if len(history) < 3:
        return {
            "trend": "INSUFFICIENT_DATA",
            "slope": 0.0,
            "strength": 0.0
        }

    # Use recent window only
    recent = history[-window:]

    scores = [entry["final_score"] for entry in recent]

    x = np.arange(len(scores))

    # Linear regression
    slope, _ = np.polyfit(x, scores, 1)

    # Normalize slope to reduce scale sensitivity
    std_dev = np.std(scores) + 1e-6
    normalized_slope = slope / std_dev

    # Classification
    if normalized_slope > 0.5:
        trend = "STRONGLY_INCREASING"
    elif normalized_slope > 0.1:
        trend = "INCREASING"
    elif normalized_slope < -0.5:
        trend = "STRONGLY_DECREASING"
    elif normalized_slope < -0.1:
        trend = "DECREASING"
    else:
        trend = "STABLE"

    return {
        "trend": trend,
        "slope": float(round(slope, 4)),
        "strength": float(round(abs(normalized_slope), 3))
    }
def detect_regime(df):
    """
    Detects market regime using moving averages.
    """

    ma50 = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()

    if ma50.iloc[-1] > ma200.iloc[-1]:
        return "BULL"
    elif ma50.iloc[-1] < ma200.iloc[-1]:
        return "BEAR"
    else:
        return "SIDEWAYS"