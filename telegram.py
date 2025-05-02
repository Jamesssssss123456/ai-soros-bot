import os
import requests

def send_telegram_alert(symbol, prob, features):
    TOKEN = os.getenv("BOT_TOKEN")
    CHAT_ID = os.getenv("CHAT_ID")
    message = f"🚨 爆拉信號：{symbol}\n概率：{prob:.2%}\n特徵：{features}"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": message})