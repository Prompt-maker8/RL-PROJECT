import requests
import os
import json
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
CACHE_FILE = "data/news_cache.json"

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()


# ==========================================
# Cache Handling
# ==========================================
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    os.makedirs("data", exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


# ==========================================
# Fetch Company News from Finnhub
# ==========================================
def fetch_news(ticker, mode="weekly"):

    today = datetime.now()

    if mode == "today":
        from_date = today.strftime("%Y-%m-%d")
    else:
        from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")

    to_date = today.strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"

    params = {
        "symbol": ticker.upper(),
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
    except:
        return []

    headlines = []

    for article in data:
        headline = article.get("headline")
        if headline:
            headlines.append(headline)

    return headlines[:5]  # limit to top 5


# ==========================================
# VADER Sentiment Scoring
# ==========================================
def rule_based_sentiment(headlines):
    scores = []
    for headline in headlines:
        compound = vader.polarity_scores(headline)['compound']
        scores.append(float(compound))
    return scores


# ==========================================
# Public Function
# ==========================================
def get_real_sentiment(ticker, mode="weekly"):

    cache = load_cache()
    today_key = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"{ticker}_{mode}_{today_key}"

    if cache_key in cache:
        return cache[cache_key]

    headlines = fetch_news(ticker, mode)

    if not headlines:
        result = {
            "sentiment": 0.0,
            "headlines": [],
            "source": "none",
            "mode": mode
        }
        return result

    try:
        if GOOGLE_API_KEY:
            model = genai.GenerativeModel("gemini-pro")
            prompt = f"Analyze the overall financial sentiment of these headlines. Output ONLY a single floating point number between -1.0 (extremely negative) and 1.0 (extremely positive). Do not provide any explanation, just the numeric float.\n\nHeadlines:\n{headlines}"
            response = model.generate_content(prompt)
            sentiment_value = float(response.text.strip())
            
            result = {
                "sentiment": max(min(sentiment_value, 1.0), -1.0),
                "headlines": headlines,
                "source": "gemini_llm",
                "mode": mode
            }
            cache[cache_key] = result
            save_cache(cache)
            return result
    except Exception as e:
        print(f"\n[!] Gemini LLM Failed: {e}\n")
        pass # Fallback to rule-based directly

    scores = rule_based_sentiment(headlines)

    weights = np.linspace(0.1, 0.3, len(scores))
    weights = weights / weights.sum()

    sentiment = float(np.dot(scores, weights))

    result = {
        "sentiment": sentiment,
        "headlines": headlines,
        "source": "finnhub_rule_based",
        "mode": mode
    }

    cache[cache_key] = result
    save_cache(cache)

    return result


# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":

    ticker = "TSLA"

    result = get_real_sentiment(ticker, mode="weekly")

    print("\n==============================")
    print("FINNHUB NEWS ENGINE TEST")
    print("==============================")
    print(f"Stock       : {ticker}")
    print(f"Sentiment   : {round(result['sentiment'], 4)}")
    print(f"Source      : {result['source']}")
    print(f"Mode        : {result['mode']}")
    print("\nHeadlines:")
    for h in result["headlines"]:
        print(f"- {h}")
    print("==============================\n")