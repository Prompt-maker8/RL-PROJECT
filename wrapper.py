import sys
import main_agent

with open("goog_results.log", "w", encoding="utf-8") as f:
    sys.stdout = f
    sys.stderr = f
    try:
        main_agent.run_agent("GOOG")
    except Exception as e:
        import traceback
        traceback.print_exc(file=f)
