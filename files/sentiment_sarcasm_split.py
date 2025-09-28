import pandas as pd

df = pd.read_csv("/Users/varunpradeep/Desktop/sentiment-analysis-bert/data/cleaned_dataset.csv")
# print(df.head())
# Sentiment dataset
sentiment_df = df[["text", "target"]]
sentiment_df.to_csv("sentiment.csv", index=False)

# Sarcasm dataset - placeholder (no real labels)
# You’d need to annotate sarcasm manually or find a sarcasm dataset
sarcasm_df = pd.DataFrame({"text": df["text"], "sarcasm": [0]*len(df)})
sarcasm_df.to_csv("sarcasm.csv", index=False)