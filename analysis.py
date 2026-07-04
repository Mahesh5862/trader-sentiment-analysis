"""
Trader Performance vs Market Sentiment Analysis
-------------------------------------------------
Joins Hyperliquid historical trade data with the Bitcoin Fear & Greed Index
on date, then compares trader performance (PnL, win rate, position size,
leverage-related behavior) across sentiment regimes.
"""

import pandas as pd
import numpy as np

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
trades = pd.read_csv("/mnt/user-data/uploads/historical_data.csv")
mood = pd.read_csv("/mnt/user-data/uploads/fear_greed_index.csv")

print("Trades shape:", trades.shape)
print("Mood shape:", mood.shape)
print(trades.columns.tolist())

# ---------------------------------------------------------------
# 2. Clean & align dates
# ---------------------------------------------------------------
trades["Timestamp IST"] = pd.to_datetime(trades["Timestamp IST"], format="%d-%m-%Y %H:%M")
trades["date"] = trades["Timestamp IST"].dt.date

mood["date"] = pd.to_datetime(mood["date"]).dt.date

# ---------------------------------------------------------------
# 3. Merge trades with daily sentiment classification
# ---------------------------------------------------------------
merged = trades.merge(mood[["date", "classification", "value"]], on="date", how="left")
print("\nUnmatched trades (no sentiment date match):", merged["classification"].isna().sum())

merged = merged.dropna(subset=["classification"])

# Standardize sentiment ordering for readability in charts/tables
sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
merged["classification"] = pd.Categorical(merged["classification"], categories=sentiment_order, ordered=True)

merged["is_win"] = merged["Closed PnL"] > 0
merged["is_loss"] = merged["Closed PnL"] < 0

# ---------------------------------------------------------------
# 4. Core summary: performance by sentiment
# ---------------------------------------------------------------
summary = merged.groupby("classification", observed=True).agg(
    trade_count=("Closed PnL", "count"),
    total_pnl=("Closed PnL", "sum"),
    avg_pnl=("Closed PnL", "mean"),
    median_pnl=("Closed PnL", "median"),
    win_rate=("is_win", "mean"),
    avg_trade_size_usd=("Size USD", "mean"),
    unique_accounts=("Account", "nunique"),
).round(3)

print("\n=== Performance by Sentiment ===")
print(summary)
summary.to_csv("/home/claude/proj/summary_by_sentiment.csv")

# ---------------------------------------------------------------
# 5. Buy vs Sell behavior by sentiment
# ---------------------------------------------------------------
side_summary = merged.groupby(["classification", "Side"], observed=True).agg(
    trade_count=("Closed PnL", "count"),
    avg_pnl=("Closed PnL", "mean"),
    win_rate=("is_win", "mean"),
).round(3)
print("\n=== Buy vs Sell by Sentiment ===")
print(side_summary)
side_summary.to_csv("/home/claude/proj/side_by_sentiment.csv")

# ---------------------------------------------------------------
# 6. Coin-level behavior (top coins by volume) across sentiment
# ---------------------------------------------------------------
top_coins = merged["Coin"].value_counts().head(5).index.tolist()
coin_summary = (
    merged[merged["Coin"].isin(top_coins)]
    .groupby(["Coin", "classification"], observed=True)
    .agg(avg_pnl=("Closed PnL", "mean"), win_rate=("is_win", "mean"), trade_count=("Closed PnL", "count"))
    .round(3)
)
print("\n=== Top 5 Coins performance by Sentiment ===")
print(coin_summary)
coin_summary.to_csv("/home/claude/proj/coin_by_sentiment.csv")

# ---------------------------------------------------------------
# 7. Account-level segmentation: who does better in Fear vs Greed
# ---------------------------------------------------------------
account_sentiment = merged.groupby(["Account", "classification"], observed=True)["Closed PnL"].mean().unstack()
account_sentiment["n_trades"] = merged.groupby("Account").size()
account_sentiment = account_sentiment[account_sentiment["n_trades"] >= 20]  # active traders only
account_sentiment.to_csv("/home/claude/proj/account_level_sentiment.csv")
print(f"\nActive accounts (20+ trades) analyzed: {len(account_sentiment)}")

# ---------------------------------------------------------------
# 8. Statistical test: is PnL significantly different Fear vs Greed?
# ---------------------------------------------------------------
from scipy import stats

fear_pnl = merged[merged["classification"].isin(["Fear", "Extreme Fear"])]["Closed PnL"].dropna()
greed_pnl = merged[merged["classification"].isin(["Greed", "Extreme Greed"])]["Closed PnL"].dropna()

t_stat, p_val = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False)
print(f"\n=== T-test: Fear vs Greed PnL ===")
print(f"Fear mean PnL: {fear_pnl.mean():.3f} (n={len(fear_pnl)})")
print(f"Greed mean PnL: {greed_pnl.mean():.3f} (n={len(greed_pnl)})")
print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.5f}")

with open("/home/claude/proj/stats_test.txt", "w") as f:
    f.write(f"Fear mean PnL: {fear_pnl.mean():.4f} (n={len(fear_pnl)})\n")
    f.write(f"Greed mean PnL: {greed_pnl.mean():.4f} (n={len(greed_pnl)})\n")
    f.write(f"t-statistic: {t_stat:.4f}\np-value: {p_val:.6f}\n")

# Save merged dataset for reference / further exploration
merged.to_csv("/home/claude/proj/merged_trades_with_sentiment.csv", index=False)

print("\nDone. Files written to /home/claude/proj/")
