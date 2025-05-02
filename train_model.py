import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. 讀取 CSV 檔案
data_path = "data/data_ALPACAUSDT.csv"
df = pd.read_csv(data_path)

# 2. 特徵欄位與標籤欄位
features = [
    "oi_change_pct",
    "basis_percent_negative",
    "top_trader_account_ls_ratio",
    "top_trader_position_ls_ratio"
]
target = "label"

# 3. 確保資料正確
df = df.dropna(subset=features + [target])
X = df[features]
y = df[target]

# 4. 分割訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 建立與訓練模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 模型評估
y_pred = model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 7. 儲存模型
model_path = "model/ai_soros_model.pkl"
os.makedirs("model", exist_ok=True)
joblib.dump(model, model_path)
print(f"✅ 模型已儲存至 {model_path}")
