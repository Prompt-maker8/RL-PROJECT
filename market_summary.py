import os
from real_news_engine import fetch_news, GOOGLE_API_KEY
import google.generativeai as genai

def generate_board_summary():
    print("Initiating global macro asset crawl...")
    # Fetch general broad market news by targeting a major market index like the S&P 500
    headlines = fetch_news("SPY", mode="weekly")
    
    if not headlines:
        print("ERROR: Could not fetch real-time market data from Finnhub.")
        return

    if not GOOGLE_API_KEY:
        print("\n[!] Google Generative API Key missing. Showing raw retrieved headlines instead:\n")
        for h in headlines:
            print(f"- {h}")
        return

    try:
        model = genai.GenerativeModel("gemini-pro")
        
        # Explicit NLP generative Prompt to summarize the overall stock condition
        prompt = f"""
Act as an elite global macro financial analyst. I am providing you with the top news headlines driving the overarching stock market (S&P 500) today.

Analyze these headlines rigorously and generate a comprehensive 2-3 paragraph summary describing:
1. The overriding market condition today (Bullish, Bearish, Flat, Anxious).
2. The core economic drivers or corporate situations affecting this posture.
3. Any immediate qualitative insights an active investor should know.

Here are the Headlines:
{headlines}
        """

        print("Querying Global Gemini NLP Evaluator...\n")
        response = model.generate_content(prompt)

        print("="*60)
        print("       TODAY'S MARKET SUMMARY (GEMINI AI)")
        print("="*60)
        print(response.text.strip())
        print("="*60)

    except Exception as e:
        print(f"Generative Core Exception: {e}")
        print("\nFalling back to raw headline printout:")
        for h in headlines:
            print(f"- {h}")

if __name__ == "__main__":
    generate_board_summary()
