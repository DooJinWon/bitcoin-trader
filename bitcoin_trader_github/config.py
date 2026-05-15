import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 바이낸스 API (.env 파일 또는 환경변수에 설정)
    BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

    # 텔레그램 (.env 파일 또는 환경변수에 설정)
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

    # 거래 설정
    SYMBOL            = os.getenv("SYMBOL", "BTCUSDT")
    TIMEFRAME         = os.getenv("TIMEFRAME", "1m")
    LEVERAGE          = int(os.getenv("LEVERAGE", "20"))
    POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "0.9"))

    # 수수료 (바이낸스 선물 테이커 0.04%)
    FEE_RATE = 0.0004

    # ML 모델
    CONFIDENCE_THRESHOLD = 0.65
    PREDICTION_HORIZON   = 30     # 30분 후 가격 예측
    FUTURE_RETURN_TARGET = 0.002  # 0.2% 이상 = 신호
    TRAIN_RATIO          = 0.70
    VAL_RATIO            = 0.15

    # 리스크 관리
    STOP_LOSS_PCT   = 0.010  # 긴급 손절 1.0% → 20x 시 -20%
    TAKE_PROFIT_PCT = 0.005  # 익절 0.5% → 20x 시 +10%
    MAX_DAILY_LOSS  = 0.05
    MIN_ORDER_USDT  = 5.0
