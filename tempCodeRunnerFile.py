import sys
from fusion_engine import fuse_signals
from trend_analyzer import analyze_trend
from reasoning_engine import generate_reasoning
from report_engine import generate_report

def run_agent(ticker):

    fusion = fuse_signals(ticker)

    # fake history (replace later with DB)
    history = [fusion]*5
    trend_info = analyze_trend(history)

    reasoning = generate_reasoning(
        ticker,
        fusion,
        trend_info,
        fusion["headlines"]
    )

    report_file = generate_report(
        ticker,
        fusion,
        fusion["headlines"],
        reasoning
    )

    print("\n=== FINAL OUTPUT ===")
    print(fusion)
    print("\nTrend:", trend_info)
    print("\nReasoning:\n", reasoning)
    print("\nReport saved:", report_file)
