import pandas as pd
import matplotlib.pyplot as plt

sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
colors = ["#8B0000", "#E07B39", "#B8B8B8", "#4E9F3D", "#1B5E20"]

summary = pd.read_csv("summary_by_sentiment.csv", index_col=0)
summary = summary.reindex(sentiment_order)

side = pd.read_csv("side_by_sentiment.csv")

# Chart 1: Average PnL by sentiment
fig, ax = plt.subplots(figsize=(7,4.5))
ax.bar(summary.index, summary["avg_pnl"], color=colors)
ax.set_title("Average Closed PnL per Trade by Market Sentiment")
ax.set_ylabel("Avg PnL (USD)")
ax.set_xlabel("Market Sentiment")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("chart1_avg_pnl.png", dpi=150)
plt.close()

# Chart 2: Win rate by sentiment
fig, ax = plt.subplots(figsize=(7,4.5))
ax.bar(summary.index, summary["win_rate"]*100, color=colors)
ax.set_title("Win Rate (%) by Market Sentiment")
ax.set_ylabel("Win Rate (%)")
ax.set_xlabel("Market Sentiment")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("chart2_win_rate.png", dpi=150)
plt.close()

# Chart 3: Avg trade size by sentiment
fig, ax = plt.subplots(figsize=(7,4.5))
ax.bar(summary.index, summary["avg_trade_size_usd"], color=colors)
ax.set_title("Average Trade Size (USD) by Market Sentiment")
ax.set_ylabel("Avg Trade Size (USD)")
ax.set_xlabel("Market Sentiment")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("chart3_trade_size.png", dpi=150)
plt.close()

# Chart 4: Buy vs Sell win rate by sentiment
pivot = side.pivot(index="classification", columns="Side", values="win_rate").reindex(sentiment_order)
fig, ax = plt.subplots(figsize=(7,4.5))
x = range(len(pivot))
width = 0.35
ax.bar([i-width/2 for i in x], pivot["BUY"]*100, width, label="Buy", color="#4E9F3D")
ax.bar([i+width/2 for i in x], pivot["SELL"]*100, width, label="Sell", color="#8B0000")
ax.set_xticks(list(x))
ax.set_xticklabels(pivot.index, rotation=20)
ax.set_ylabel("Win Rate (%)")
ax.set_title("Buy vs Sell Win Rate by Market Sentiment")
ax.legend()
plt.tight_layout()
plt.savefig("chart4_buy_sell_winrate.png", dpi=150)
plt.close()

print("Charts saved.")
