
from telegram import Bot
import os

def send_telegram_alert(symbol, prob, features, tp, sl, rr):
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    bot = Bot(token=TOKEN)

    msg = (
        f"🚨 轧空警報：{symbol}\n"
        f"📊 機率：{round(prob * 100, 2)}%\n"
        f"🎯 TP：{tp}, 🛡 SL：{sl}, ⚖ RR：{rr}\n"
        f"📈 特徵：{features}"
    )
    bot.send_message(chat_id=CHAT_ID, text=msg)
