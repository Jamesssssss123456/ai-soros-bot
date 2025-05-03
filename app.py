
import os
import time
import joblib
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from utils.binance_api import fetch_all_symbols_data
from utils.feature_engineering import prepare_features
from utils.telegram_bot_helper import send_telegram_alert
from sklearn.metrics import classification_report

MODEL_PATH = "model/ai_soros_model.pkl"
DATA_PATH = "data/data_ALPACAUSDT.csv"
model = joblib.load(MODEL_PATH)

def monitor_job():
    print("⏱️ 每分鐘監控中...")
    try:
        data = fetch_all_symbols_data()
        for symbol, features in data.items():
            X = prepare_features(features)
            prob = model.predict_proba([X])[0][1]
            if prob > 0.7:
                send_telegram_alert(symbol, prob, X)
    except Exception as e:
        print(f"錯誤: {e}")

def backtest(update: Update, context: CallbackContext) -> None:
    try:
        df = pd.read_csv(DATA_PATH)
        features = [
            "oi_change_pct",
            "basis_percent_negative",
            "top_trader_account_ls_ratio",
            "top_trader_position_ls_ratio"
        ]
        X = df[features]
        y = (df["label"] != 0).astype(int)
        y_pred = model.predict(X)
        y_pred_signal = [1 if p != 0 else 0 for p in y_pred]
        report = classification_report(y, y_pred_signal, digits=3, output_dict=False)
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"📊 回測結果：\n{report}")
    except Exception as e:
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ 回測錯誤：{e}")

if __name__ == "__main__":
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("backtest", backtest))

    scheduler = BackgroundScheduler(timezone=pytz.utc)
    scheduler.add_job(monitor_job, 'interval', minutes=1)
    scheduler.start()

    updater.start_polling()
    print("✅ Bot 已啟動，可使用 /backtest 並每分鐘推理")
    updater.idle()
