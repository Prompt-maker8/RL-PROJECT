# AI Financial Analyzer Architecture

This repository dynamically orchestrates Reinforcement Learning networks securely against generative NLP sentiment logic to output optimized asset evaluations via unified interactive modules.

## Architecture Flow Diagram

```mermaid
graph TD
    %% External APIs
    subgraph "External Data Ingestion"
    YF[("Yahoo Finance\n(Pricing & Historics)")]
    FH[("Finnhub API\n(Live Market News)")]
    end

    %% Core Data Parsing Engines
    subgraph "Core Evaluation Engines"
    EVAL["`evalution_model.py`\nOffline Agent Evaluator"]
    TREND["`trend_analyzer.py`\nTechnical Regime Detection"]
    NEWS["`real_news_engine.py`\nHeadlines Scraper"]
    end
    
    %% NLP Sentiment Parsing
    subgraph "NLP Sentiment Processors"
    GEMINI{"Google Gemini API\n(Primary Contextual LLM)"}
    VADER{"VADER Natural Language\n(Secure Offline Fallback)"}
    end

    %% Network Routing
    YF -->|"Asset Data Streams"| TREND
    YF -->|"Volatility & Volume"| EVAL
    FH -->|"Daily Top Headlines"| NEWS
    
    NEWS -->|"Query Evaluator"| GEMINI
    GEMINI -- "Auth/Quota Block" --> VADER
    GEMINI -- "Live Score [-1 to 1]" --> FUSION
    VADER -- "Compound Score [-1 to 1]" --> FUSION
    
    EVAL -->|"Network Query"| PPO(("Trained `RecurrentPPO` Model\n(`global_macro_agent.zip`)"))
    PPO -->|"Output Action Vector"| FUSION
    TREND -->|"MA Crossover (Bull/Bear/Sideways)"| FUSION

    %% Engine Orchestration Node
    subgraph "Orchestration Layer"
    FUSION{"`fusion_engine.py`\n(System Weights & Tanh Bounding)"}
    REASON["`reasoning_engine.py`\n(Deduplicated Analytical Synthesis)"]
    end
    
    FUSION -->|"Unified Bounded Score"| REASON
    
    %% High-level Triggers and Final Output Tools
    subgraph "Actionable Outputs & Execution Apps"
    MAIN(("`main_agent.py`\n(Single Ticker Run)"))
    SCAN(("`scan_market.py`\n(Broad Universe Sorting)"))
    MARKET(("`market_summary.py`\n(Gemini GenAI Context)"))
    REPORT>"`report_engine.py`\n(Matplotlib Graphing & PDF Emit)"]
    end

    FUSION -->|"Trigger Configuration"| MAIN
    FUSION -->|"Evaluate Global Matrix"| SCAN
    FH -.->|"SPY Core Headlines"| MARKET
    
    REASON -->|"Feed Analysis"| REPORT
    MAIN -->|"Final Emit"| REPORT
```

## Architectural Processing Rules
1. **Model Security:** The `RecurrentPPO` algorithm is explicitly segregated into `evaluation_model.py` to allow live `predict()` inference checks without engaging environment retraining states randomly out of runtime mode.
2. **Confidence Clamping:** Output logic is merged actively via algorithmic matrices normalized identically around a rigorous boundary (`np.tanh(score * 1.5)`) to eliminate arbitrary score scaling and establish consistent boundaries across changing global macro conditions safely.
3. **Redundancy Structuring:** If the primary Google Gemini evaluator disconnects, the native framework instantly defaults to VADER offline analytics without ever halting or executing a 0-signal static trap configuration.
