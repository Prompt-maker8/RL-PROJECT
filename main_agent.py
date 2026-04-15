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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        selected_ticker = sys.argv[1].upper()
    else:
        universe = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOG", "AMZN", "META", "NFLX", "JPM"]
        print(f"Available Universe: {', '.join(universe)}")
        selected_ticker = input(f"Select a robust ticker to analyze [{', '.join(universe[:3])}...]: ").strip().upper()
        if not selected_ticker:
            selected_ticker = "AAPL"
            
    print(f"\n[ SYSTEM ] Engaging full pipeline for {selected_ticker}...")
    run_agent(selected_ticker)