
import os
import joblib
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from utils.binance_api import fetch_all_symbols_data
from utils.feature_engineering import prepare_features, calculate_tp_sl_risk
from utils.telegram_bot_helper import send_telegram_alert
from sklearn.metrics import classification_report
import pytz

MODEL_PATH = "model/ai_soros_model.pkl"
DATA_PATH = "data/data_ALPACAUSDT.csv"
model = joblib.load(MODEL_PATH)

def monitor_job():
    print("⏱️ 每分鐘監控中...")
    try:
        data = fetch_all_symbols_data()
        for symbol, features in data.items():
            X = prepare_features(features)
            probs = model.predict_proba([X])
            if probs.shape[1] > 1:
                prob = probs[0][1]
                if prob > 0.7:
                    tp, sl, rr = calculate_tp_sl_risk(features)
                    send_telegram_alert(symbol, prob, X, tp, sl, rr)
    except Exception as e:
        print(f"❌ 錯誤: {e}")

def backtest(update: Update, context: CallbackContext):
    update.message.reply_text("📊 回測功能待實作中，請稍後...")

if __name__ == "__main__":
    TOKEN = os.getenv("TELEGRAM_TOKEN")  # Render 上請設定 TELEGRAM_TOKEN 環境變量

    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # 加入 /backtest 指令
    dispatcher.add_handler(CommandHandler("backtest", backtest))

    # 啟動監控排程
    scheduler = BackgroundScheduler(timezone=pytz.utc)
    scheduler.add_job(monitor_job, 'interval', minutes=1)
    scheduler.start()

    # ✅ 使用 polling 模式
    updater.start_polling()
    print("✅ Bot 已啟動，Polling 模式監聽中...")
    updater.idle()




