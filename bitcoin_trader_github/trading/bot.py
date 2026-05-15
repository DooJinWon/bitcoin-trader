from __future__ import annotations

import time
import logging
import signal
import sys
from datetime import datetime, timezone, timedelta

from config import Config
from data.fetcher import BinanceFetcher
from analysis.ml_model import BTCSignalModel
from notifications.telegram_bot import TelegramNotifier
from notifications.telegram_commands import TelegramCommandBot

logger = logging.getLogger(__name__)


class BinanceFuturesBot:
    """
    바이낸스 BTC/USDT 선물 자동 트레이딩 봇 (10x 레버리지).

    - LONG  시그널 → 롱 포지션 진입
    - SHORT 시그널 → 숏 포지션 진입
    - 반대 시그널 → 기존 포지션 청산 후 반대 진입
    - 손절(SL) / 익절(TP) 자동 관리
    """

    POLL_INTERVAL = 10  # 10초마다 새 캔들 체크

    def __init__(self):
        self.fetcher  = BinanceFetcher(Config.BINANCE_API_KEY, Config.BINANCE_SECRET_KEY)
        self.model    = BTCSignalModel()
        self.notifier = TelegramNotifier(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)

        self.position: dict | None = None  # 봇이 추적하는 포지션 정보
        self.daily_pnl    = 0.0
        self.daily_trades = 0
        self.last_candle_ts = None
        self._running       = True
        self.trading_active = True

        self.cmd_bot = TelegramCommandBot(
            Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID, self
        )

        # 레버리지 설정
        self.fetcher.set_leverage(Config.SYMBOL, Config.LEVERAGE)

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    # ── 메인 루프 ─────────────────────────────────────────────────────────

    def run(self):
        logger.info("=" * 55)
        logger.info("  바이낸스 BTC/USDT 선물 자동 트레이딩 봇 시작")
        logger.info(f"  심볼: {Config.SYMBOL} | 타임프레임: {Config.TIMEFRAME}")
        logger.info(f"  레버리지: {Config.LEVERAGE}x | 신뢰도 임계값: {Config.CONFIDENCE_THRESHOLD*100:.0f}%")
        logger.info(f"  손절: -{Config.STOP_LOSS_PCT*100:.1f}% | 익절: +{Config.TAKE_PROFIT_PCT*100:.1f}%")
        logger.info("=" * 55)

        self.model.load()
        self.cmd_bot.start()

        self.notifier.send_text(
            f"바이낸스 BTC 선물 트레이딩 봇 시작\n"
            f"심볼: {Config.SYMBOL} | {Config.TIMEFRAME} | {Config.LEVERAGE}x\n"
            f"손절: -{Config.STOP_LOSS_PCT*100:.1f}% | 익절: +{Config.TAKE_PROFIT_PCT*100:.1f}%\n"
            f"명령어: /help 를 입력하세요"
        )

        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"봇 오류: {e}", exc_info=True)
                self.notifier.send_error(str(e))
                time.sleep(30)
            time.sleep(self.POLL_INTERVAL)

    # ── 단일 틱 처리 ──────────────────────────────────────────────────────

    def _tick(self):
        df = self.fetcher.fetch_latest(Config.SYMBOL, Config.TIMEFRAME, count=300)
        if df.empty:
            return

        latest_ts = df.index[-1]
        if latest_ts == self.last_candle_ts:
            return
        self.last_candle_ts = latest_ts

        signal_info = self.model.predict(df)
        signal      = signal_info["signal"]
        price       = self.fetcher.get_current_price(Config.SYMBOL)
        signal_info["price"] = price

        # 실제 거래소 포지션 동기화
        real_pos = self.fetcher.get_position(Config.SYMBOL)
        if real_pos and not self.position:
            self.position = {
                "side":      real_pos["side"],
                "entry":     real_pos["entry_price"],
                "usdt_used": 0.0,
                "size":      real_pos["size"],
                "open_ts":   datetime.now(timezone.utc),
                "confidence": 0.0,
            }

        trade_flag = "거래중" if self.trading_active else "감시중(정지)"
        side_str   = f"[{self.position['side']}]" if self.position else "[없음]"
        logger.info(
            f"[{latest_ts}] ${price:,.2f} | "
            f"LONG={signal_info['long_prob']:.3f} SHORT={signal_info['short_prob']:.3f} | "
            f"시그널={signal} | 포지션{side_str} | {trade_flag}"
        )

        # ── SL/TP 체크 (자동매매 정지 중에도 청산 수행) ──────────────────
        if self.position:
            should_close, reason = self._check_exit(price, signal)
            if should_close:
                self._close_position(price, reason)

        if not self.trading_active:
            return

        # ── 일일 손실 한도 체크 ───────────────────────────────────────────
        total_usdt = self.fetcher.get_total_usdt()
        if total_usdt > 0 and self.daily_pnl / total_usdt < -Config.MAX_DAILY_LOSS:
            logger.warning("일일 손실 한도 도달 — 신규 진입 중단")
            return

        # ── 신규 진입 (포지션 없을 때만) ─────────────────────────────────
        if not self.position:
            if signal == "LONG":
                self._open_position("LONG", price, signal_info)
                self.notifier.send_signal(signal_info)
            elif signal == "SHORT":
                self._open_position("SHORT", price, signal_info)
                self.notifier.send_signal(signal_info)

    # ── 포지션 관리 ───────────────────────────────────────────────────────

    def _open_position(self, side: str, price: float, signal_info: dict):
        usdt_balance = self.fetcher.get_usdt_balance()
        usdt_to_use  = usdt_balance * Config.POSITION_SIZE_PCT

        if usdt_to_use < Config.MIN_ORDER_USDT:
            logger.warning(f"USDT 부족: {usdt_to_use:.2f} USDT")
            return

        # 수수료 커버 여부 확인 (왕복 수수료 = 0.08%)
        fee_cost = usdt_to_use * Config.LEVERAGE * Config.FEE_RATE * 2
        min_profit_usdt = fee_cost * 1.5  # 수수료의 1.5배 이상 수익 가능해야 진입
        logger.info(f"예상 왕복 수수료: ${fee_cost:.4f} | 최소 필요 수익: ${min_profit_usdt:.4f}")

        try:
            if side == "LONG":
                order = self.fetcher.open_long(Config.SYMBOL, usdt_to_use * Config.LEVERAGE)
            else:
                order = self.fetcher.open_short(Config.SYMBOL, usdt_to_use * Config.LEVERAGE)

            self.position = {
                "side":       side,
                "entry":      price,
                "usdt_used":  usdt_to_use,
                "size":       self.fetcher._calc_quantity(Config.SYMBOL, usdt_to_use * Config.LEVERAGE, price),
                "open_ts":    datetime.now(timezone.utc),
                "confidence": signal_info["confidence"],
            }
            self.daily_trades += 1

            emoji = "🟢" if side == "LONG" else "🔴"
            self.notifier.send_order_filled(
                side, price,
                self.position["size"],
                order.get("orderId", "") if order else ""
            )
            sl = price * (1 - Config.STOP_LOSS_PCT) if side == "LONG" else price * (1 + Config.STOP_LOSS_PCT)
            tp = price * (1 + Config.TAKE_PROFIT_PCT) if side == "LONG" else price * (1 - Config.TAKE_PROFIT_PCT)
            logger.info(f"{emoji} {side} 진입: ${price:,.2f} | SL=${sl:,.2f} | TP=${tp:,.2f}")

        except Exception as e:
            logger.error(f"진입 실패: {e}")
            self.notifier.send_error(f"{side} 진입 실패: {e}")

    def _close_position(self, price: float, reason: str):
        if not self.position:
            return
        try:
            self.fetcher.close_position(Config.SYMBOL)

            entry   = self.position["entry"]
            side    = self.position["side"]
            if side == "LONG":
                ret_pct = (price - entry) / entry
            else:
                ret_pct = (entry - price) / entry

            # 레버리지 적용 실제 수익률
            pnl_pct  = ret_pct * Config.LEVERAGE
            # 수수료 차감
            fee_pct  = Config.FEE_RATE * 2 * Config.LEVERAGE
            net_pct  = pnl_pct - fee_pct
            pnl_usdt = net_pct * self.position["usdt_used"]

            self.daily_pnl += pnl_usdt
            self.notifier.send_position_closed(side, entry, price, pnl_usdt, reason)
            logger.info(
                f"청산 {reason} | {side} | 진입=${entry:,.2f} 청산=${price:,.2f} | "
                f"레버리지수익={pnl_pct*100:+.2f}% | 순수익={net_pct*100:+.2f}% | ${pnl_usdt:+.4f}"
            )
            self.position = None

        except Exception as e:
            logger.error(f"청산 실패: {e}")
            self.notifier.send_error(f"청산 실패: {e}")

    def _check_exit(self, price: float, new_signal: str) -> tuple[bool, str]:
        if not self.position:
            return False, ""
        entry = self.position["entry"]
        side  = self.position["side"]

        if side == "LONG":
            ret = (price - entry) / entry
        else:
            ret = (entry - price) / entry

        # 긴급 손절 (3% 안전망)
        if ret <= -Config.STOP_LOSS_PCT:
            return True, "STOP_LOSS"

        # 예측 보유 기간 경과 → 청산
        open_ts = self.position.get("open_ts")
        if open_ts:
            elapsed = datetime.now(timezone.utc) - open_ts
            if elapsed >= timedelta(minutes=Config.PREDICTION_HORIZON):
                if ret >= Config.TAKE_PROFIT_PCT:
                    return True, "TAKE_PROFIT"
                return True, "TIMEOUT"

        return False, ""

    # ── 안전 종료 ─────────────────────────────────────────────────────────

    def _shutdown(self, *_):
        logger.info("봇 종료 중...")
        self._running = False
        if self.position:
            price = self.fetcher.get_current_price(Config.SYMBOL)
            self._close_position(price, "BOT_SHUTDOWN")
        self.notifier.send_text("바이낸스 선물 트레이딩 봇 종료")
        sys.exit(0)
