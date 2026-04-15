# AI Carbon Asset Analyzer & Orchestrator

Welcome to the AI Carbon / Global Macro Trading Agent platform! This robust financial analysis pipeline natively integrates reinforcement learning (RL) models, rule-based algorithmic trading signals, and Google's Generative AI (Gemini) NLP capabilities to assess real-time market regimes and output strategic valuation paths for the world's top assets.

## 🚀 How to Run the Platform

### 1. Generating a Report for a Specific Company (Interactive)
To evaluate the live posture for an individual asset, execute the main agent sequence:
```bash
python main_agent.py
```
* **Interactive Mode**: If you run it without arguments, the system provides a list of the built-in target universe (`AAPL`, `TSLA`, `NVDA`, etc.) and asks you to type which one you want.
* **Fast Target**: You can also skip the prompt by passing it directly: `python main_agent.py MSFT`.
* **Output**: A comprehensive terminal readout containing the scaled NLP sentiment, the active trend/regime, and a generated visual asset PDF tracking its 6-month momentum cleanly (`[TICKER]_report.pdf`).

### 2. Scanning the Market for Actionable Signals
If you want to evaluate the broader universe automatically and categorize all target states (BUY / HOLD / SELL) natively:
```bash
python scan_market.py
```
This sequentially loops through the entire global macro universe, applies the daily parameters, and lists all assets sorted into optimal strategic brackets dynamically.

### 3. Evaluating and Visualizing Historical Model Training Tests
If you want to view how the embedded Neural Network `RecurrentPPO` agent behaved dynamically vs standard "Buy & Hold" strategies through the 2022 crash year:
```bash
python evalution_model.py
```
This loads up the offline environment natively, renders all 10 unified asset graphs concurrently onto one beautiful sub-plot grid window, and provides an algorithmic table showing exact backtest returns globally.

### 4. Fetching Today's True Market Summary Context
I built a specific sub-script leveraging the Gemini LLM engine designed to help you quickly understand the literal shape and tone of today's market directly outside rigid stock variables. Execute:
```bash
python market_summary.py
```
* **How It Works**: This script polls wide-ranging news via the `real_news_engine`, but instead of squashing the output into structured `-1` to `1` math arrays for the math models, it feeds the headlines dynamically to a configured `gemini-pro` NLP prompt to give you a cohesive conversational briefing covering "today's market and etc."
