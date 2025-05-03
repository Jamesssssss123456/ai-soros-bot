
import os
from telegram import Bot

bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))

def send_telegram_alert(symbol, prob, features, tp, sl, rr):
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    message = (
        f"🚨 <b>轧空預測警報</b>\n"
        f"📈 幣種: <code>{symbol}</code>\n"
        f"📊 預測機率: <b>{round(prob * 100, 2)}%</b>\n"
        f"💡 技術指標:\n"
        f" - OI變化率: {features[0]}\n"
        f" - 負基差%: {features[1]}\n"
        f" - 大戶多空帳戶比: {features[2]}\n"
        f" - 大戶持倉比: {features[3]}\n"
        f"🎯 建議入場點: 現價\n"
        f"🟢 止盈 TP: <b>{tp}</b>\n"
        f"🔴 止損 SL: <b>{sl}</b>\n"
        f"📐 風報比 R:R = <b>{rr}</b>"
    )
    bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")

