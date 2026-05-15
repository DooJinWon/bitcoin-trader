from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes

from config import Config

if TYPE_CHECKING:
    from trading.bot import BinanceFuturesBot

logger = logging.getLogger(__name__)


class TelegramCommandBot:
    """텔레그램 양방향 커맨드 봇 (바이낸스 선물 버전)."""

    def __init__(self, token: str, chat_id: str, trading_bot: "BinanceFuturesBot"):
        self.token   = token
        self.chat_id = str(chat_id)
        self.bot     = trading_bot
        self._app    = None
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("텔레그램 커맨드 봇 시작")

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._start_app())

    async def _start_app(self):
        self._app = Application.builder().token(self.token).build()

        handlers = [
            ("start",       self._cmd_help),
            ("help",        self._cmd_help),
            ("balance",     self._cmd_balance),
            ("price",       self._cmd_price),
            ("status",      self._cmd_status),
            ("profit",      self._cmd_profit),
            ("signal",      self._cmd_signal),
            ("trading_on",  self._cmd_trading_on),
            ("trading_off", self._cmd_trading_off),
            ("long",        self._cmd_long),
            ("short",       self._cmd_short),
            ("close",       self._cmd_close),
        ]
        for cmd, handler in handlers:
            self._app.add_handler(CommandHandler(cmd, handler))

        await self._app.bot.set_my_commands([
            BotCommand("long",        "수동 롱 진입 — /long 100"),
            BotCommand("short",       "수동 숏 진입 — /short 100"),
            BotCommand("close",       "포지션 전량 청산"),
            BotCommand("trading_on",  "자동매매 시작"),
            BotCommand("trading_off", "자동매매 정지"),
            BotCommand("balance",     "잔고 조회 (USDT)"),
            BotCommand("price",       "BTC 현재가"),
            BotCommand("status",      "봇 상태 & 포지션"),
            BotCommand("profit",      "현재 포지션 수익률"),
            BotCommand("signal",      "ML 시그널 확인"),
            BotCommand("help",        "전체 명령어"),
        ])

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(allowed_updates=["message"])
        logger.info("텔레그램 폴링 시작")

        while self.bot._running:
            await asyncio.sleep(1)

        await self._app.updater.stop()
        await self._app.stop()

    # ── 명령어 핸들러 ─────────────────────────────────────────────────────

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update): return
        active = "자동매매 중" if self.bot.trading_active else "자동매매 정지"
        await update.message.reply_text(
            f"BTC 선물 봇 명령어  [{active}]\n"
            f"레버리지: {Config.LEVERAGE}x\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "/long <USDT>  — 수동 롱 진입\n"
            "/short <USDT> — 수동 숏 진입\n"
            "/close        — 포지션 청산\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "/trading_on  — 자동매매 시작\n"
            "/trading_off — 자동매매 정지\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "/balance — 잔고\n"
            "/price   — 현재가\n"
            "/status  — 상태 & 포지션\n"
            "/profit  — 수익률\n"
            "/signal  — ML 시그널\n"
        )

    async def _cmd_balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update): return
        try:
            avail = self.bot.fetcher.get_usdt_balance()
            total = self.bot.fetcher.get_total_usdt()
            price = self.bot.fetcher.get_current_price(Config.SYMBOL)
            await update.message.reply_text(
                "선물 지갑 잔고\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"사용 가능: ${avail:>12,.2f}\n"
                f"총 자산:   ${total:>12,.2f}\n"
                f"BTC 현재가: ${price:>10,.2f}\n"
                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            )
        except Exception as e:
            await update.message.reply_text(f"잔고 조회 실패: {e}")

    async def _cmd_price(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update): return
        try:
            price = self.bot.fetcher.get_current_price(Config.SYMBOL)
            await update.message.reply_text(
                f"BTC/USDT 현재가\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 ${price:,.2f}\n"
                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            )
        except Exception as e:
            await update.message.reply_text(f"현재가 조회 실패: {e}")

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update): return
        try:
            pos   = self.bot.position
            price = self.bot.fetcher.get_current_price(Config.SYMBOL)
            avail = self.bot.fetcher.get_usdt_balance()
            total = self.bot.fetcher.get_total_usdt()
            active = "자동매매 중" if self.bot.trading_active else "정지"

            if pos:
                entry = pos["entry"]
                side  = pos["side"]
                ret   = (price - entry) / entry if side == "LONG" else (entry - price) / entry
                pnl_pct  = ret * Config.LEVERAGE
                fee_pct  = Config.FEE_RATE * 2 * Config.LEVERAGE
                net_pct  = pnl_pct - fee_pct
                pnl_usdt = net_pct * pos["usdt_used"]
                sl = entry * (1 - Config.STOP_LOSS_PCT) if side == "LONG" else entry * (1 + Config.STOP_LOSS_PCT)
                tp = entry * (1 + Config.TAKE_PROFIT_PCT) if side == "LONG" else entry * (1 - Config.TAKE_PROFIT_PCT)
                emoji = "🟢" if side == "LONG" else "🔴"
                pos_str = (
                    f"\n{emoji} {side} 포지션 보유중\n"
                    f"진입가: ${entry:,.2f}\n"
                    f"현재가: ${price:,.2f}\n"
                    f"수익률: {net_pct*100:+.2f}% (${pnl_usdt:+.4f})\n"
                    f"손절가: ${sl:,.2f} | 익절가: ${tp:,.2f}"
                )
            else:
                pos_str = "\n포지션: 없음"

            await update.message.reply_text(
                f"봇 상태  [{active}]\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"레버리지: {Config.LEVERAGE}x\n"
                f"가용 USDT: ${avail:,.2f}\n"
                f"총 자산:   ${total:,.2f}\n"
                f"일일 거래: {self.bot.daily_trades}회\n"
                f"일일 손익: ${self.bot.daily_pnl:+.4f}\n"
                f"━━━━━━━━━━━━━━━━━━━━"
                f"{pos_str}\n"
                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            )
        except Exception as e:
            await update.message.reply_text(f"상태 조회 실패: {e}")

    async def _cmd_profit(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update): return
        try:
            pos = self.bot.position
            if not pos:
                await update.message.reply_text(
                    f"보유 포지션 없음\n일일 누적 손익: ${self.bot.daily_pnl:+.4f}"
                )
                return
            price = self.bot.fetcher.get_current_price(Config.SYMBOL)
            entry = pos["entry"]
            side  = pos["side"]
            ret   = (price - entry) / entry if side == "LONG" else (entry - price) / entry
            pnl_pct  = ret * Config.LEVERAGE
            fee_pct  = Config.FEE_RATE * 2 * Config.LEVERAGE
            net_pct  = pnl_pct - fee_pct
            pnl_usdt = net_pct * pos["usdt_used"]
            sl = entry * (1 - Config.STOP_LOSS_PCT) if side == "LONG" else entry * (1 + Config.STOP_LOSS_PCT)
            tp = entry * (1 + Config.TAKE_PROFIT_PCT) if side == "LONG" else entry * (1 - Config.TAKE_PROFIT_PCT)

            emoji = "📈" if net_pct >= 0 else "📉"
            await update.message.reply_text(
                f"{emoji} {side} 포지션 수익\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"진입가: ${entry:>12,.2f}\n"
                f"현재가: ${price:>12,.2f}\n"
                f"가격변동: {ret*100:>+11.3f}%\n"
                f"레버리지({Config.LEVERAGE}x): {pnl_pct*100:>+7.2f}%\n"
                f"수수료 차감: -{fee_pct*100:.3f}%\n"
                f"순수익률: {net_pct*100:>+10.2f}%\n"
                f"순손익:  ${pnl_usdt:>+11.4f}\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"손절가: ${sl:,.2f}\n"
                f"익절가: ${tp:,.2f}\n"
                f"일일 누적: ${self.bot.daily_pnl:+.4f}"
            )
        except Exception as e:
            await update.message.reply_text(f"수익 조회 실패: {e}")

    async def _cmd_signal(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update): return
        try:
            df = self.bot.fetcher.fetch_latest(Config.SYMBOL, Config.TIMEFRAME, count=300)
            if df.empty:
                await update.message.reply_text("데이터 조회 실패")
                return
            info  = self.bot.model.predict(df)
            price = self.bot.fetcher.get_current_price(Config.SYMBOL)
            sig   = info["signal"]
            lp    = info["long_prob"]  * 100
            sp    = info["short_prob"] * 100
            th_l  = self.bot.model.threshold_long  * 100
            th_s  = self.bot.model.threshold_short * 100
            sig_emoji = {"LONG": "🟢 롱 신호!", "SHORT": "🔴 숏 신호!", "HOLD": "⚪ 대기"}.get(sig, sig)
            await update.message.reply_text(
                "최신 ML 시그널\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"현재가: ${price:,.2f}\n"
                f"시그널: {sig_emoji}\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"롱 확률:  {lp:5.1f}%  (임계값 {th_l:.0f}%)\n"
                f"{self._prob_bar(lp, th_l)}\n"
                f"숏 확률:  {sp:5.1f}%  (임계값 {th_s:.0f}%)\n"
                f"{self._prob_bar(sp, th_s)}\n"
                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            )
        except Exception as e:
            await update.message.reply_text(f"시그널 조회 실패: {e}")

    async def _cmd_trading_on(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update): return
        if self.bot.trading_active:
            await update.message.reply_text("이미 자동매매 실행 중입니다.")
            return
        self.bot.trading_active = True
        logger.info("자동매매 시작 (텔레그램)")
        avail = self.bot.fetcher.get_usdt_balance()
        await update.message.reply_text(
            "자동매매 시작\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"가용 USDT: ${avail:,.2f}\n"
            f"레버리지: {Config.LEVERAGE}x\n"
            "시그널 감지 시 자동 진입합니다."
        )

    async def _cmd_trading_off(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update): return
        if not self.bot.trading_active:
            await update.message.reply_text("이미 자동매매 정지 상태입니다.")
            return
        self.bot.trading_active = False
        logger.info("자동매매 정지 (텔레그램)")
        pos = self.bot.position
        pos_str = ""
        if pos:
            price   = self.bot.fetcher.get_current_price(Config.SYMBOL)
            entry   = pos["entry"]
            ret     = (price - entry) / entry if pos["side"] == "LONG" else (entry - price) / entry
            pnl_pct = ret * Config.LEVERAGE - Config.FEE_RATE * 2 * Config.LEVERAGE
            pos_str = f"\n{pos['side']} 포지션 유지 중 ({pnl_pct*100:+.2f}%)\n손절/익절은 계속 작동합니다."
        await update.message.reply_text(
            "자동매매 정지\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "신규 진입을 하지 않습니다."
            f"{pos_str}\n"
            "재시작: /trading_on"
        )

    async def _cmd_long(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update): return
        if not ctx.args:
            await update.message.reply_text("사용법: /long <USDT금액>\n예시: /long 100")
            return
        try:
            usdt = float(ctx.args[0].replace(",", ""))
        except ValueError:
            await update.message.reply_text("올바른 금액을 입력하세요.")
            return
        if usdt < Config.MIN_ORDER_USDT:
            await update.message.reply_text(f"최소 주문금액: ${Config.MIN_ORDER_USDT} USDT")
            return
        avail = self.bot.fetcher.get_usdt_balance()
        if usdt > avail:
            await update.message.reply_text(f"잔고 부족\n가용: ${avail:.2f} | 요청: ${usdt:.2f}")
            return
        if self.bot.position:
            await update.message.reply_text("이미 포지션 보유 중입니다. /close 후 다시 시도하세요.")
            return
        try:
            await update.message.reply_text(f"🟢 롱 진입 중... ${usdt:.2f} USDT (레버리지 {Config.LEVERAGE}x)")
            self.bot.fetcher.open_long(Config.SYMBOL, usdt * Config.LEVERAGE)
            time.sleep(1)
            price = self.bot.fetcher.get_current_price(Config.SYMBOL)
            sl = price * (1 - Config.STOP_LOSS_PCT)
            tp = price * (1 + Config.TAKE_PROFIT_PCT)
            self.bot.position = {
                "side": "LONG", "entry": price, "usdt_used": usdt,
                "size": self.bot.fetcher._calc_quantity(Config.SYMBOL, usdt * Config.LEVERAGE, price),
                "open_ts": datetime.now(timezone.utc), "confidence": 1.0,
            }
            self.bot.daily_trades += 1
            await update.message.reply_text(
                "🟢 롱 진입 완료\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"진입가:  ${price:,.2f}\n"
                f"마진:    ${usdt:.2f} USDT\n"
                f"노출:    ${usdt*Config.LEVERAGE:.2f} USDT ({Config.LEVERAGE}x)\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"손절가: ${sl:,.2f} (-{Config.STOP_LOSS_PCT*100:.1f}%)\n"
                f"익절가: ${tp:,.2f} (+{Config.TAKE_PROFIT_PCT*100:.1f}%)"
            )
            logger.info(f"수동 롱: ${usdt:.2f} USDT @ ${price:,.2f}")
        except Exception as e:
            await update.message.reply_text(f"롱 진입 실패: {e}")

    async def _cmd_short(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update): return
        if not ctx.args:
            await update.message.reply_text("사용법: /short <USDT금액>\n예시: /short 100")
            return
        try:
            usdt = float(ctx.args[0].replace(",", ""))
        except ValueError:
            await update.message.reply_text("올바른 금액을 입력하세요.")
            return
        if usdt < Config.MIN_ORDER_USDT:
            await update.message.reply_text(f"최소 주문금액: ${Config.MIN_ORDER_USDT} USDT")
            return
        avail = self.bot.fetcher.get_usdt_balance()
        if usdt > avail:
            await update.message.reply_text(f"잔고 부족\n가용: ${avail:.2f} | 요청: ${usdt:.2f}")
            return
        if self.bot.position:
            await update.message.reply_text("이미 포지션 보유 중입니다. /close 후 다시 시도하세요.")
            return
        try:
            await update.message.reply_text(f"🔴 숏 진입 중... ${usdt:.2f} USDT (레버리지 {Config.LEVERAGE}x)")
            self.bot.fetcher.open_short(Config.SYMBOL, usdt * Config.LEVERAGE)
            time.sleep(1)
            price = self.bot.fetcher.get_current_price(Config.SYMBOL)
            sl = price * (1 + Config.STOP_LOSS_PCT)
            tp = price * (1 - Config.TAKE_PROFIT_PCT)
            self.bot.position = {
                "side": "SHORT", "entry": price, "usdt_used": usdt,
                "size": self.bot.fetcher._calc_quantity(Config.SYMBOL, usdt * Config.LEVERAGE, price),
                "open_ts": datetime.now(timezone.utc), "confidence": 1.0,
            }
            self.bot.daily_trades += 1
            await update.message.reply_text(
                "🔴 숏 진입 완료\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"진입가:  ${price:,.2f}\n"
                f"마진:    ${usdt:.2f} USDT\n"
                f"노출:    ${usdt*Config.LEVERAGE:.2f} USDT ({Config.LEVERAGE}x)\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"손절가: ${sl:,.2f} (+{Config.STOP_LOSS_PCT*100:.1f}%)\n"
                f"익절가: ${tp:,.2f} (-{Config.TAKE_PROFIT_PCT*100:.1f}%)"
            )
            logger.info(f"수동 숏: ${usdt:.2f} USDT @ ${price:,.2f}")
        except Exception as e:
            await update.message.reply_text(f"숏 진입 실패: {e}")

    async def _cmd_close(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update): return
        pos = self.bot.position
        if not pos:
            real = self.bot.fetcher.get_position(Config.SYMBOL)
            if not real:
                await update.message.reply_text("청산할 포지션이 없습니다.")
                return
        try:
            price = self.bot.fetcher.get_current_price(Config.SYMBOL)
            await update.message.reply_text(f"포지션 청산 중... ${price:,.2f}")
            self.bot._close_position(price, "MANUAL_CLOSE")
            await update.message.reply_text(
                "청산 완료\n"
                f"청산가: ${price:,.2f}\n"
                f"일일 누적 손익: ${self.bot.daily_pnl:+.4f}"
            )
        except Exception as e:
            await update.message.reply_text(f"청산 실패: {e}")

    # ── 유틸 ──────────────────────────────────────────────────────────────

    def _is_authorized(self, update: Update) -> bool:
        return str(update.effective_chat.id) == self.chat_id

    @staticmethod
    def _prob_bar(prob: float, threshold: float, width: int = 10) -> str:
        filled = int(prob / 10)
        bar    = "█" * filled + "░" * (width - filled)
        marker = "▲" if prob >= threshold else ""
        return f"[{bar}] {marker}"
