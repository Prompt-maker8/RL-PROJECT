import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from sklearn.preprocessing import RobustScaler
import yfinance as yf
import matplotlib.pyplot as plt
import random

# ===============================
# 1. Research-Grade Indicators
# ===============================
def compute_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = (100 - (100 / (1 + rs))) / 100
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean() / df['Close']
    
    df['macd'] = (df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()) / df['Close']
    
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_ret'].rolling(10).std()
    
    sma50 = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200).mean()
    df['regime'] = (sma50 > sma200).astype(float)
    
    df['next_ret'] = df['Close'].pct_change().shift(-1)
    return df.dropna()

# ===============================
# 🔥 RL SIGNAL FUNCTION (ADDED)
# ===============================
def get_rl_signal(ticker, model):
    df = yf.download(ticker, period="1y")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = compute_indicators(df)

    scaler = RobustScaler()
    cols = ["log_ret", "rsi", "atr", "macd", "volatility"]
    df[cols] = scaler.fit_transform(df[cols])

    obs = df[["log_ret","rsi","atr","macd","volatility","regime"]].values[-1]

    # ✅ CRUCIAL FIX
    position = 0.0
    obs = np.append(obs, [position]).astype(np.float32)

    action, _ = model.predict(obs, deterministic=True)

    return {
        "signal": float(action[0]),
        "confidence": abs(float(action[0]))
    }

# ===============================
# 2. ENVIRONMENT
# ===============================
class GlobalMacroEnv(gym.Env):
    def __init__(self, data_dict, is_train=True):
        super().__init__()
        self.data_dict = data_dict
        self.tickers = list(data_dict.keys())
        self.is_train = is_train
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None, ticker=None):
        super().reset(seed=seed)
        self.current_ticker = ticker if ticker else random.choice(self.tickers)
        df = self.data_dict[self.current_ticker]
        
        self.features = df[["log_ret", "rsi", "atr", "macd", "volatility", "regime"]].values
        self.returns = df["next_ret"].values
        
        if self.is_train and len(self.features) > 501:
            self.current_step = random.randint(0, len(self.features) - 501)
            self.max_steps = self.current_step + 500 
        else:
            self.current_step = 0
            self.max_steps = len(self.features) - 1
            
        self.position = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        feat = self.features[self.current_step]
        return np.append(feat, [self.position]).astype(np.float32)

    def step(self, action):
        target_pos = np.clip(action[0], -1, 1)
        prev_pos = self.position
        self.position = 0.7 * target_pos + 0.3 * prev_pos 
        
        fee = 0.001
        cost = abs(self.position - prev_pos) * fee
        net_ret = (self.position * self.returns[self.current_step]) - cost
        
        current_vol = self.features[self.current_step][4] 
        reward = net_ret / (current_vol + 1e-4)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), float(reward), done, False, {"net_ret": net_ret}

# ===============================
# 3. MAIN PIPELINE
# ===============================
def run_full_pipeline():
    MODEL_PATH = "global_macro_agent.zip"
    universe = ["AAPL","TSLA","NVDA","AMD","MSFT","GOOG","AMZN","META","NFLX","JPM"]

    if not os.path.exists(MODEL_PATH):
        raise ValueError("❌ Model not found! Train first.")

    train_dict, test_dict = {}, {}
    scaler = RobustScaler()

    print("Downloading data...")

    for t in universe:
        df = yf.download(t, start="2017-01-01", end="2023-01-01")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = compute_indicators(df)

        # 🔥 ONLY FOR SCALING (NO TRAINING)
        train_df = df[df.index < "2022-01-01"]

        # 🔥 TEST ON CRASH YEAR
        test_df = df[(df.index >= "2022-01-01") & (df.index < "2023-01-01")]

        cols = ["log_ret","rsi","atr","macd","volatility"]

        # fit scaler on old data
        train_df[cols] = scaler.fit_transform(train_df[cols])

        # apply on crash year
        test_df[cols] = scaler.transform(test_df[cols])

        train_dict[t], test_dict[t] = train_df, test_df

    # ✅ only needed for loading model structure
    dummy_env = GlobalMacroEnv(train_dict, is_train=True)

    print("✅ Loading trained model (NO retraining)...")
    model = RecurrentPPO.load(MODEL_PATH, env=dummy_env)

    print("\n" + "="*50)
    print("ANALYSIS ON CRASH YEAR (2022)")
    print("="*50)

    summary_results = []

    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()

    for idx, ticker in enumerate(universe):
        eval_env = GlobalMacroEnv(test_dict, is_train=False)
        obs, _ = eval_env.reset(ticker=ticker)

        history = []
        done = False
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
            obs, _, done, _, info = eval_env.step(action)
            history.append(info["net_ret"])
            episode_starts = np.zeros((1,), dtype=bool)

        # 🔥 RL SIGNAL (no retraining)
        rl_signal = get_rl_signal(ticker, model)
        print(f"{ticker} RL Signal:", rl_signal)
        rl_returns = np.cumprod(1 + np.array(history))
        bh_returns = np.cumprod(1 + test_dict[ticker]["next_ret"].values[:-1])

        ax = axes[idx]
        ax.plot(rl_returns, label="RL Agent", linewidth=2)
        ax.plot(bh_returns, label="Buy & Hold", linestyle="--")

        ax.set_title(f"{ticker} (Crash Year 2022)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Returns")

        ax.legend()
        ax.grid(alpha=0.3)

        rl_final = (rl_returns[-1] - 1) * 100
        bh_final = (bh_returns[-1] - 1) * 100

        summary_results.append({
            "Ticker": ticker,
            "RL_Return_%": round(rl_final, 2),
            "B&H_Return_%": round(bh_final, 2),
            "Beat_Market": "YES" if rl_final > bh_final else "NO"
        })

    plt.tight_layout()
    plt.savefig("evaluation_graphs.png")

    df_res = pd.DataFrame(summary_results)

    print("\n--- CRASH YEAR PERFORMANCE ---")
    print(df_res.to_string(index=False))

    print("\nAverage RL:", df_res["RL_Return_%"].mean())
    print("Average B&H:", df_res["B&H_Return_%"].mean())

if __name__ == "__main__":
    run_full_pipeline()