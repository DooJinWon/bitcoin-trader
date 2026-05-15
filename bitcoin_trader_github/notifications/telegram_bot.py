import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """텔레그램 봇으로 매매 알림을 전송합니다."""

    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = str(chat_id)
        self._bot    = None
        self.enabled = bool(token and chat_id)
        if not self.enabled:
            logger.warning("텔레그램 토큰/채팅ID 미설정 — 알림 비활성화")

    def _get_bot(self):
        if self._bot is None:
            from telegram import Bot
            self._bot = Bot(token=self.token)
        return self._bot

    # ── 알림 메서드 ───────────────────────────────────────────────────────

    def send_signal(self, signal_info: dict):
        """매매 시그널 알림 (업비트 현물 기준)."""
        if not self.enabled:
            return

        signal     = signal_info["signal"]
        price      = signal_info["price"]
        confidence = signal_info["confidence"] * 100
        long_prob  = signal_info["long_prob"] * 100
        short_prob = signal_info["short_prob"] * 100
        ts         = signal_info.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))

        from config import Config

        if signal == "LONG":
            emoji     = "🟢"
            direction = "매수 (BUY)"
            tp_price  = price * (1 + Config.TAKE_PROFIT_PCT)
            sl_price  = price * (1 - Config.STOP_LOSS_PCT)
            action    = f"✅ 익절가: {tp_price:,.0f}원 (+{Config.TAKE_PROFIT_PCT*100:.1f}%)\n🛡️ 손절가: {sl_price:,.0f}원 (-{Config.STOP_LOSS_PCT*100:.1f}%)"
        else:
            emoji     = "🔴"
            direction = "매도 (SELL)"
            action    = "💰 BTC 전량 매도 → KRW 전환"

        msg = f"""
{emoji} *BTC 매매 시그널 발생*
━━━━━━━━━━━━━━━━━━━━
📊 방향: *{direction}*
💰 현재가: {price:,.0f}원
🎯 신뢰도: *{confidence:.1f}%*

📈 매수 확률: {long_prob:.1f}%
📉 매도 확률: {short_prob:.1f}%

{action}

⏰ {ts}
━━━━━━━━━━━━━━━━━━━━
⚠️ 투자는 본인 책임입니다.
""".strip()
        self._send(msg)

    def send_order_filled(self, side: str, price: float, amount: float, order_id: str):
        """주문 체결 알림."""
        if not self.enabled:
            return
        emoji = "🟢" if side == "BUY" else "🔴"
        msg = f"""
{emoji} *주문 체결 완료*
━━━━━━━━━━━━━━━━━━━━
방향: {side}
체결가: {price:,.0f}원
수량: {amount:.8f} BTC
주문ID: `{order_id}`
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
        self._send(msg)

    def send_position_closed(self, side: str, entry: float, exit_price: float, pnl_krw: float, reason: str):
        """포지션 청산 알림."""
        if not self.enabled:
            return
        ret_pct   = (exit_price - entry) / entry * 100
        emoji     = "✅" if pnl_krw > 0 else "❌"
        reason_kr = {
            "TAKE_PROFIT":  "익절 (Take Profit)",
            "STOP_LOSS":    "손절 (Stop Loss)",
            "SELL_SIGNAL":  "매도 시그널",
            "BOT_SHUTDOWN": "봇 종료",
        }.get(reason, reason)

        msg = f"""
{emoji} *포지션 청산*
━━━━━━━━━━━━━━━━━━━━
사유: {reason_kr}
진입가: {entry:,.0f}원
청산가: {exit_price:,.0f}원
수익률: {ret_pct:+.2f}%
손익: {pnl_krw:+,.0f}원
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
        self._send(msg)

    def send_daily_summary(self, stats: dict):
        """일일 요약 알림."""
        if not self.enabled:
            return
        msg = f"""
📊 *일일 트레이딩 요약*
━━━━━━━━━━━━━━━━━━━━
총 거래: {stats.get('total_trades', 0)}회
승률: {stats.get('win_rate', 0):.1f}%
일일 손익: {stats.get('daily_pnl', 0):+,.0f}원
추정 자산: {stats.get('total_value', 0):,.0f}원
⏰ {datetime.now().strftime('%Y-%m-%d')}
""".strip()
        self._send(msg)

    def send_error(self, error_msg: str):
        """에러 알림."""
        if not self.enabled:
            return
        self._send(f"🚨 *봇 에러*\n{error_msg}\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def send_text(self, text: str):
        if self.enabled:
            self._send(text)

    # ── 내부 전송 ─────────────────────────────────────────────────────────

    def _send(self, text: str):
        try:
            try:
                asyncio.run(self._async_send(text))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._async_send(text))
                loop.close()
        except Exception as e:
            logger.error(f"텔레그램 전송 실패: {e}")

    async def _async_send(self, text: str):
        bot = self._get_bot()
        await bot.send_message(chat_id=self.chat_id, text=text, parse_mode="Markdown")
        logger.info("텔레그램 알림 전송 완료")
