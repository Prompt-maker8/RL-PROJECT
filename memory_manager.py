import json
import os
from datetime import datetime

HISTORY_PATH = "data/history.json"


def _ensure_storage():
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "w") as f:
            json.dump({}, f)


def load_memory():
    _ensure_storage()
    with open(HISTORY_PATH, "r") as f:
        return json.load(f)


def save_memory(memory):
    with open(HISTORY_PATH, "w") as f:
        json.dump(memory, f, indent=4)


def store_daily_result(ticker, fusion_result):
    memory = load_memory()
    today = datetime.now().strftime("%Y-%m-%d")

    entry = {
        "date": today,
        "rl_signal": fusion_result["rl_signal"],
        "sentiment": fusion_result["sentiment"],
        "final_score": fusion_result["final_score"],
        "confidence": fusion_result["confidence"],
        "decision": fusion_result["decision"]
    }

    if ticker not in memory:
        memory[ticker] = []

    memory[ticker].append(entry)
    save_memory(memory)


def get_recent_history(ticker, days=7):
    memory = load_memory()
    if ticker not in memory:
        return []
    return memory[ticker][-days:]