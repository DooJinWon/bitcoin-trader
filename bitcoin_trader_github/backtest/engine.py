from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import logging

from config import Config

logger = logging.getLogger(__name__)


class BacktestEngine:
    """바이낸스 선물 백테스팅 엔진 (레버리지 적용, LONG/SHORT 양방향)."""

    def __init__(
        self,
        initial_usdt: float = 1000.0,
        leverage: int = Config.LEVERAGE,
        position_size_pct: float = Config.POSITION_SIZE_PCT,
        stop_loss_pct: float = Config.STOP_LOSS_PCT,
        take_profit_pct: float = Config.TAKE_PROFIT_PCT,
        fee_rate: float = Config.FEE_RATE,
        hold_bars: int = Config.PREDICTION_HORIZON,
    ):
        self.initial_usdt      = initial_usdt
        self.leverage          = leverage
        self.position_size_pct = position_size_pct
        self.stop_loss_pct     = stop_loss_pct
        self.take_profit_pct   = take_profit_pct
        self.fee_rate          = fee_rate
        self.hold_bars         = hold_bars

    def run(self, df: pd.DataFrame) -> dict:
        usdt, equity, trades = self.initial_usdt, [], []
        position = None
        daily_pnl, last_date = 0.0, None
        bar_idx = 0

        for ts, row in df.iterrows():
            price  = row["close"]
            signal = row["signal"]
            date   = ts.date() if hasattr(ts, "date") else ts
            equity.append(usdt + (self._unrealized(position, price) if position else 0))
            bar_idx += 1

            if date != last_date:
                daily_pnl, last_date = 0.0, date

            if position:
                held_bars = bar_idx - position["open_bar"]
                ret = self._ret(position, price)

                # 익절: 매 바마다 체크 (라이브 봇과 동일)
                if ret >= self.take_profit_pct:
                    pnl, usdt = self._close(position, price, usdt)
                    daily_pnl += pnl
                    trades.append(self._record(position, price, pnl, "TAKE_PROFIT", ts, usdt))
                    position = None

                # 긴급 손절 (안전망): 매 바마다 체크
                elif ret <= -self.stop_loss_pct:
                    pnl, usdt = self._close(position, price, usdt)
                    daily_pnl += pnl
                    trades.append(self._record(position, price, pnl, "STOP_LOSS", ts, usdt))
                    position = None

                # 예측 기간(hold_bars) 경과 → TIMEOUT 청산
                elif held_bars >= self.hold_bars:
                    pnl, usdt = self._close(position, price, usdt)
                    daily_pnl += pnl
                    trades.append(self._record(position, price, pnl, "TIMEOUT", ts, usdt))
                    position = None

            if usdt > 0 and daily_pnl / usdt < -Config.MAX_DAILY_LOSS:
                continue

            if not position and signal in ("LONG", "SHORT"):
                margin = usdt * self.position_size_pct
                if margin >= Config.MIN_ORDER_USDT:
                    position = {"side": signal, "entry": price, "margin": margin, "ts": ts,
                                "conf": row.get("long_prob" if signal == "LONG" else "short_prob", 0),
                                "open_bar": bar_idx}

        if position and not df.empty:
            lp = df["close"].iloc[-1]
            pnl, usdt = self._close(position, lp, usdt)
            trades.append(self._record(position, lp, pnl, "FORCED_CLOSE", df.index[-1], usdt))

        trades_df = pd.DataFrame(trades)
        return {"equity": equity, "trades": trades_df, "stats": self._stats(equity, trades_df)}

    def _ret(self, pos, price):
        return (price - pos["entry"]) / pos["entry"] if pos["side"] == "LONG" else (pos["entry"] - price) / pos["entry"]

    def _unrealized(self, pos, price):
        return self._ret(pos, price) * self.leverage * pos["margin"]

    def _close(self, pos, price, usdt):
        net = self._ret(pos, price) * self.leverage - self.fee_rate * 2 * self.leverage
        pnl = net * pos["margin"]
        return pnl, max(usdt + pnl, 0)

    def _record(self, pos, exit_price, pnl, reason, close_ts, equity):
        ret = self._ret(pos, exit_price)
        return {"open_ts": pos["ts"], "close_ts": close_ts, "side": pos["side"],
                "entry": pos["entry"], "exit": exit_price, "ret_pct": ret * 100,
                "lev_pct": ret * self.leverage * 100, "pnl_usdt": pnl,
                "reason": reason, "confidence": pos["conf"], "equity": equity}

    def _stats(self, equity, trades):
        eq  = np.array(equity)
        ret = np.diff(eq) / np.where(eq[:-1] == 0, 1, eq[:-1])
        base = {"total_return_pct": round((eq[-1]/self.initial_usdt-1)*100, 2),
                "final_usdt": round(float(eq[-1]), 4),
                "max_drawdown_pct": round(self._mdd(eq)*100, 2),
                "sharpe_ratio": round(self._sharpe(ret), 3),
                "total_trades": 0, "win_rate": 0.0, "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0, "profit_factor": 0.0}
        if trades.empty:
            return base
        wins, losses = trades[trades["pnl_usdt"] > 0], trades[trades["pnl_usdt"] <= 0]
        gp, gl = wins["pnl_usdt"].sum(), losses["pnl_usdt"].sum()
        base.update({"total_trades": len(trades),
                     "long_trades": len(trades[trades["side"] == "LONG"]),
                     "short_trades": len(trades[trades["side"] == "SHORT"]),
                     "win_rate": round(len(wins)/len(trades)*100, 2),
                     "avg_win_pct": round(wins["lev_pct"].mean(), 2) if not wins.empty else 0,
                     "avg_loss_pct": round(losses["lev_pct"].mean(), 2) if not losses.empty else 0,
                     "profit_factor": round(gp/abs(gl), 3) if gl != 0 else float("inf"),
                     "total_pnl": round(trades["pnl_usdt"].sum(), 4),
                     "tp_count": len(trades[trades["reason"] == "TAKE_PROFIT"]),
                     "sl_count": len(trades[trades["reason"] == "STOP_LOSS"]),
                     "timeout_count": len(trades[trades["reason"] == "TIMEOUT"]),
                     "reverse_count": len(trades[trades["reason"] == "REVERSE"])})
        return base

    @staticmethod
    def _mdd(eq):
        peak = np.maximum.accumulate(eq)
        return float(-((eq - peak) / np.where(peak == 0, 1, peak)).min())

    @staticmethod
    def _sharpe(ret, periods=525600):
        return float(ret.mean() / ret.std() * np.sqrt(periods)) if ret.std() != 0 else 0.0

    def print_report(self, result):
        s = result["stats"]
        print("\n" + "=" * 60)
        print("     바이낸스 BTC 선물 백테스팅 결과")
        print("=" * 60)
        print(f"  초기 자본      : ${self.initial_usdt:>12,.2f}")
        print(f"  최종 자본      : ${s['final_usdt']:>12,.4f}")
        print(f"  총 수익률      : {s['total_return_pct']:>+13.2f} %")
        print(f"  총 손익        : ${s.get('total_pnl', 0):>+12,.4f}")
        print(f"  최대 낙폭(MDD) : {s['max_drawdown_pct']:>13.2f} %")
        print(f"  Sharpe Ratio   : {s['sharpe_ratio']:>14.3f}")
        print("-" * 60)
        print(f"  레버리지       : {self.leverage}x")
        print(f"  총 거래        : {s['total_trades']:>14} 회")
        print(f"  롱 / 숏        : {s.get('long_trades',0)} / {s.get('short_trades',0)}")
        print(f"  승률           : {s['win_rate']:>13.2f} %")
        print(f"  평균 수익(승)  : {s['avg_win_pct']:>+13.2f} %")
        print(f"  평균 손실(패)  : {s['avg_loss_pct']:>+13.2f} %")
        print(f"  Profit Factor  : {s['profit_factor']:>14.3f}")
        print(f"  익절(TP) / 손절(SL): {s.get('tp_count',0)} / {s.get('sl_count',0)}")
        print(f"  시간만료(TO) / 역전(RV): {s.get('timeout_count',0)} / {s.get('reverse_count',0)}")
        print("=" * 60 + "\n")

    def plot(self, result, save_path="backtest_report.png"):
        sns.set_theme(style="darkgrid")
        fig = plt.figure(figsize=(20, 10), facecolor="#1a1a2e")
        gs  = gridspec.GridSpec(2, 2, figure=fig)
        s, equity, trades = result["stats"], result["equity"], result["trades"]

        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor("#16213e")
        ax1.plot(equity, color="#00d4aa", linewidth=1.2, label="포트폴리오 (USDT)")
        ax1.axhline(self.initial_usdt, color="#aaa", linestyle="--", alpha=0.5)
        ax1.set_title("자본 곡선", fontsize=14, color="white")
        ax1.set_ylabel("USDT", color="white")
        ax1.tick_params(colors="white")
        ax1.legend(facecolor="#1a1a2e", labelcolor="white")

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor("#16213e")
        if not trades.empty:
            colors = ["#00d4aa" if v > 0 else "#ff4757" for v in trades["pnl_usdt"]]
            ax2.bar(range(len(trades)), trades["lev_pct"], color=colors)
            ax2.axhline(0, color="white", linewidth=0.5)
        ax2.set_title("거래별 수익률 (레버리지 적용 %)", fontsize=12, color="white")
        ax2.tick_params(colors="white")

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor("#16213e")
        ax3.axis("off")
        text = (f"수익률: {s['total_return_pct']:+.2f}%  |  MDD: {s['max_drawdown_pct']:.2f}%  |  "
                f"Sharpe: {s['sharpe_ratio']:.3f}\n"
                f"거래: {s['total_trades']}회  |  승률: {s['win_rate']:.1f}%  |  "
                f"PF: {s['profit_factor']:.3f}  |  손익: ${s.get('total_pnl',0):+.4f}")
        ax3.text(0.5, 0.5, text, transform=ax3.transAxes, fontsize=11,
                 ha="center", va="center", color="white",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#2d3436", edgecolor="#00d4aa"))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        logger.info(f"차트 저장: {save_path}")
        plt.close()
