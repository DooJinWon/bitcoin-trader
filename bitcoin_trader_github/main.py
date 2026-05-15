#!/usr/bin/env python3
"""
바이낸스 BTC/USDT 선물 자동 트레이딩 시스템

사용법:
  python3 main.py fetch    # 과거 데이터 수집 (최초 1회, 약 10~15분)
  python3 main.py train    # ML 모델 학습
  python3 main.py backtest # 백테스팅 실행
  python3 main.py live     # 실거래 봇 실행
  python3 main.py all      # fetch → train → backtest 순서 실행
"""

import sys
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("btc_trader.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def cmd_fetch(args):
    from config import Config
    from data.fetcher import BinanceFetcher

    fetcher = BinanceFetcher()  # 공개 데이터는 키 불필요
    logger.info(f"바이낸스 {args.years}년 데이터 수집 시작...")
    df = fetcher.fetch_historical(Config.SYMBOL, Config.TIMEFRAME, years=args.years)
    if not df.empty:
        print(f"\n수집 완료: {len(df):,} 캔들 ({df.index[0]} ~ {df.index[-1]})\n")
    else:
        print("데이터 수집 실패.")


def cmd_train(args):
    from config import Config
    from data.fetcher import BinanceFetcher
    from analysis.ml_model import BTCSignalModel

    fetcher = BinanceFetcher()
    df = fetcher.fetch_historical(Config.SYMBOL, Config.TIMEFRAME, years=args.years)
    if df.empty:
        print("데이터 없음. 먼저 fetch를 실행하세요.")
        return
    model = BTCSignalModel()
    model.train(df)
    print("\n모델 학습 완료!\n")


def cmd_backtest(args):
    from config import Config
    from data.fetcher import BinanceFetcher
    from analysis.ml_model import BTCSignalModel
    from backtest.engine import BacktestEngine

    fetcher = BinanceFetcher()
    df = fetcher.fetch_historical(Config.SYMBOL, Config.TIMEFRAME, years=args.years)
    if df.empty:
        print("데이터 없음.")
        return
    model = BTCSignalModel()
    model.load()
    n_start    = int(len(df) * (Config.TRAIN_RATIO + Config.VAL_RATIO))
    df_test    = df.iloc[n_start:]
    df_signals = model.predict_batch(df_test)
    engine = BacktestEngine(initial_usdt=args.capital)
    result = engine.run(df_signals)
    engine.print_report(result)
    engine.plot(result, save_path="backtest_report.png")
    print("차트 저장: backtest_report.png\n")


def cmd_live(_args):
    from trading.bot import BinanceFuturesBot
    bot = BinanceFuturesBot()
    bot.run()


def cmd_all(args):
    cmd_fetch(args)
    cmd_train(args)
    cmd_backtest(args)


def main():
    parser = argparse.ArgumentParser(description="바이낸스 BTC 선물 자동 트레이딩 시스템")
    parser.add_argument("command", choices=["fetch", "train", "backtest", "live", "all"])
    parser.add_argument("--years",   type=int,   default=2,       help="수집 연수 (기본값: 2)")
    parser.add_argument("--capital", type=float, default=1000.0,  help="백테스팅 초기 자본 USDT (기본값: 1000)")
    args = parser.parse_args()
    {
        "fetch":    cmd_fetch,
        "train":    cmd_train,
        "backtest": cmd_backtest,
        "live":     cmd_live,
        "all":      cmd_all,
    }[args.command](args)


if __name__ == "__main__":
    main()
