import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """RSI, MACD, 볼린저밴드, 이동평균 등 기술적 지표를 계산합니다."""

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        close = d["close"]
        high  = d["high"]
        low   = d["low"]
        vol   = d["volume"]

        # ── 이동평균 ─────────────────────────────────────────────────────
        for period in [7, 20, 50, 100, 200]:
            d[f"ma{period}"] = close.rolling(period).mean()

        d["ema12"] = close.ewm(span=12, adjust=False).mean()
        d["ema26"] = close.ewm(span=26, adjust=False).mean()

        # 골든/데드 크로스 시그널 (1 = 골든, -1 = 데드, 0 = 중립)
        d["cross_20_50"]  = np.sign(d["ma20"] - d["ma50"])
        d["cross_50_200"] = np.sign(d["ma50"] - d["ma200"])

        # ── RSI ──────────────────────────────────────────────────────────
        for period in [14, 21]:
            d[f"rsi{period}"] = IndicatorEngine._rsi(close, period)

        # Stochastic RSI
        d["stoch_rsi_k"], d["stoch_rsi_d"] = IndicatorEngine._stoch_rsi(close, 14, 14, 3, 3)

        # ── MACD ─────────────────────────────────────────────────────────
        d["macd"]        = d["ema12"] - d["ema26"]
        d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
        d["macd_hist"]   = d["macd"] - d["macd_signal"]
        d["macd_cross"]  = np.sign(d["macd"] - d["macd_signal"])

        # ── 볼린저밴드 ───────────────────────────────────────────────────
        d["bb_mid"]   = close.rolling(20).mean()
        bb_std        = close.rolling(20).std()
        d["bb_upper"] = d["bb_mid"] + 2 * bb_std
        d["bb_lower"] = d["bb_mid"] - 2 * bb_std
        d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"]
        # -1(하단 이탈) ~ 0(중간) ~ 1(상단 이탈) 정규화
        d["bb_pos"]   = (close - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-10)
        d["bb_pos"]   = d["bb_pos"].clip(0, 1)

        # ── ATR (변동성) ─────────────────────────────────────────────────
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        d["atr14"] = tr.ewm(span=14, adjust=False).mean()
        d["atr_pct"] = d["atr14"] / close  # 가격 대비 변동성 %

        # ── 모멘텀 / 수익률 ──────────────────────────────────────────────
        for lag in [1, 4, 12, 24, 48, 168]:  # 1h, 4h, 12h, 1d, 2d, 7d
            d[f"ret_{lag}h"] = close.pct_change(lag)

        # ── 거래량 지표 ──────────────────────────────────────────────────
        d["vol_ma20"]  = vol.rolling(20).mean()
        d["vol_ratio"] = vol / (d["vol_ma20"] + 1e-10)  # 거래량 상대 비율

        # OBV (On-Balance Volume)
        d["obv"] = (np.sign(close.diff()) * vol).cumsum()
        d["obv_ma"] = d["obv"].rolling(20).mean()
        d["obv_signal"] = np.sign(d["obv"] - d["obv_ma"])

        # ── 과매수/과매도 복합 시그널 ────────────────────────────────────
        d["oversold"]  = ((d["rsi14"] < 30) & (d["bb_pos"] < 0.1)).astype(int)
        d["overbought"] = ((d["rsi14"] > 70) & (d["bb_pos"] > 0.9)).astype(int)

        return d

    # ── 내부 계산 헬퍼 ────────────────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs    = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _stoch_rsi(
        close: pd.Series,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_smooth: int = 3,
        d_smooth: int = 3,
    ):
        rsi      = IndicatorEngine._rsi(close, rsi_period)
        rsi_min  = rsi.rolling(stoch_period).min()
        rsi_max  = rsi.rolling(stoch_period).max()
        stoch_k  = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
        k        = stoch_k.rolling(k_smooth).mean()
        d        = k.rolling(d_smooth).mean()
        return k, d

    @staticmethod
    def get_feature_columns() -> list[str]:
        """ML 모델 입력 피처 목록."""
        return [
            "ma7", "ma20", "ma50", "ma100", "ma200",
            "cross_20_50", "cross_50_200",
            "rsi14", "rsi21",
            "stoch_rsi_k", "stoch_rsi_d",
            "macd", "macd_signal", "macd_hist", "macd_cross",
            "bb_width", "bb_pos",
            "atr_pct",
            "ret_1h", "ret_4h", "ret_12h", "ret_24h", "ret_48h", "ret_168h",
            "vol_ratio", "obv_signal",
            "oversold", "overbought",
        ]
