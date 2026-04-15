from fusion_engine import fuse_signals

universe = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOG", "AMZN", "META", "NFLX", "JPM"]
print("Scanning global macro universe for evaluating all positions natively...\n")

results = {"BUY": [], "HOLD": [], "SELL": []}

for ticker in universe:
    try:
        res = fuse_signals(ticker)
        print(f"[{ticker}] Processed -> Action: {res['decision']} | Fusion Score: {res['final_score']}")
        results[res['decision']].append((ticker, res['final_score']))
    except Exception as e:
        print(f"[{ticker}] Scan Blocked: {e}")

print("\n" + "="*50)
print("       OVERALL MARKET CLASSIFICATION")
print("="*50)

for decision in ["BUY", "SELL", "HOLD"]:
    print(f"\n[ ::: {decision} TARGETS ::: ]")
    if not results[decision]:
        print("  -> None detected matching logic constraints.")
    else:
        # Sort logic: Buy descending (best buys), Sell ascending (strongest sells), Hold absolute (flat)
        rev = True if decision in ["BUY", "HOLD"] else False
        sorted_targets = sorted(results[decision], key=lambda x: x[1], reverse=rev)
        
        for t in sorted_targets:
            print(f"  -> {t[0]:<5} (Engine Value: {t[1]:.4f})")

print("\nScan Completed Successfully.")
