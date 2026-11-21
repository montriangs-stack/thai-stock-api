from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(title="Thai Stock API", version="1.0")

# อนุญาตให้เว็บ Frontend บน Vercel เรียกใช้งาน API นี้ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "success", "message": "Thai Stock API is running with real market data!"}

@app.get("/api/stock/{symbol}")
def get_stock_data(symbol: str):
    try:
        # เติม .BK ให้หุ้นไทยอัตโนมัติหากยังไม่มี
        if not symbol.endswith(".BK"):
            ticker_symbol = f"{symbol.upper()}.BK"
        else:
            ticker_symbol = symbol.upper()

        # ดึงข้อมูลย้อนหลัง 6 เดือน
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period="6mo")

        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"ไม่พบข้อมูลของหุ้น {symbol} หรือข้อมูลน้อยเกินไป")

        # คำนวณค่าทางเทคนิคพื้นฐาน (Indicators)
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        current_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        change_pct = float(((current_price - prev_price) / prev_price) * 100)

        # EMA 20 & 50
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

        # RSI 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs))).iloc[-1]

        # ATR 14 (สำหรับคำนวณ Entry / TP / SL)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])

        # คำนวณจุดเข้า-ออกเบื้องต้นจาก ATR
        entry_price = current_price
        stop_loss = round(entry_price - (1.5 * atr14), 2)
        tp1 = round(entry_price + (2 * atr14), 2)
        tp2 = round(entry_price + (3.5 * atr14), 2)

        # สรุปผลลัพธ์ส่งกลับไปที่หน้าเว็บ
        return {
            "symbol": symbol.upper(),
            "price": round(current_price, 2),
            "change_pct": round(change_pct, 2),
            "high": round(float(high.max()), 2),
            "low": round(float(low.min()), 2),
            "volume": int(volume.iloc[-1]),
            "indicators": {
                "ema20": round(ema20, 2),
                "ema50": round(ema50, 2),
                "rsi": round(rsi, 2),
                "atr14": round(atr14, 2)
            },
            "trade_plan": {
                "entry": round(entry_price, 2),
                "stop_loss": stop_loss,
                "tp1": tp1,
                "tp2": tp2
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
