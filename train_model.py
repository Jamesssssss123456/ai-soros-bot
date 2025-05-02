import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 讀取CSV資料
data_path = "data/data_ALPACAUSDT.csv"
df = pd.read_csv(data_path)

# 特徵與標籤
features = [
    "oi_change_pct",
    "basis_percent_negative",
    "top_trader_account_ls_ratio",
    "top_trader_position_ls_ratio"
]
X = df[features]
y = (df["label"] != 0).astype(int)

# 模型訓練與儲存
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, "model/ai_soros_model.pkl")
print("✅ 模型訓練完成並儲存為 model/ai_soros_model.pkl")