from __future__ import annotations

import pandas as pd
import time
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)


class BinanceFetcher:
    """바이낸스 선물 OHLCV 데이터 수집 및 주문 실행."""

    # 바이낸스 interval → 분(minutes)
    TF_MINUTES = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
    }

    def __init__(self, api_key: str = "", secret_key: str = ""):
        self.client: Optional[Client] = None
        if api_key and secret_key:
            self.client = Client(api_key, secret_key)

    # ── 데이터 수집 ───────────────────────────────────────────────────────

    def fetch_historical(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1m",
        years: int = 2,
        cache_path: str = "data/cache",
    ) -> pd.DataFrame:
        """과거 N년치 OHLCV 수집 (로컬 캐시 활용)."""
        os.makedirs(cache_path, exist_ok=True)
        cache_file = os.path.join(cache_path, f"{symbol}_{timeframe}_{years}y.parquet")

        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            last_dt = df.index[-1].to_pydatetime().replace(tzinfo=None)
            new_df  = self._fetch_range(symbol, timeframe, from_dt=last_dt)
            if new_df is not None and not new_df.empty:
                df = pd.concat([df, new_df]).drop_duplicates()
                df.to_parquet(cache_file)
            logger.info(f"캐시 로드: {len(df):,} 캔들 ({df.index[0]} ~ {df.index[-1]})")
            return df

        df = self._fetch_range(symbol, timeframe, years=years)
        if df is not None and not df.empty:
            df.to_parquet(cache_file)
            logger.info(f"수집 완료: {len(df):,} 캔들 ({df.index[0]} ~ {df.index[-1]})")
        return df if df is not None else pd.DataFrame()

    def _fetch_range(
        self,
        symbol: str,
        timeframe: str,
        years: int = 2,
        from_dt: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """바이낸스 API를 반복 호출해 전체 기간 데이터를 수집합니다."""
        tf_min    = self.TF_MINUTES.get(timeframe, 1)
        total_min = years * 365 * 24 * 60
        n_candles = total_min // tf_min
        target    = n_candles if from_dt is None else 1000

        if from_dt:
            start_ts = int(from_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        else:
            start_ts = int((datetime.utcnow() - timedelta(minutes=total_min)).timestamp() * 1000)

        logger.info(f"{symbol} {timeframe} 데이터 수집 중... (목표: {target:,} 캔들)")

        all_dfs = []
        fetched = 0
        current_ts = start_ts

        while fetched < target:
            try:
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=timeframe,
                    startTime=current_ts,
                    limit=1000,
                ) if self.client else self._public_klines(symbol, timeframe, current_ts)
            except Exception as e:
                logger.error(f"API 오류: {e}")
                time.sleep(3)
                continue

            if not klines:
                break

            df = self._klines_to_df(klines)
            all_dfs.append(df)
            fetched += len(df)
            last_ts = int(df.index[-1].timestamp() * 1000)
            current_ts = last_ts + tf_min * 60 * 1000

            # 마지막 캔들이 현재 시각을 넘으면 종료
            if current_ts > int(datetime.utcnow().timestamp() * 1000):
                break

            logger.info(f"  수집 중... {df.index[-1].date()} ({fetched:,}/{target:,})")
            time.sleep(0.1)

        if not all_dfs:
            return None

        result = pd.concat(all_dfs).sort_index()
        result = result[~result.index.duplicated(keep="last")]
        return result

    def _public_klines(self, symbol: str, interval: str, start_ts: int) -> list:
        """API 키 없이 공개 엔드포인트로 수집."""
        import requests
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "startTime": start_ts, "limit": 1000}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _klines_to_df(self, klines: list) -> pd.DataFrame:
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time")
        df.index = df.index.tz_localize(None)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_latest(self, symbol: str, timeframe: str, count: int = 300) -> pd.DataFrame:
        """실거래 봇용: 최신 N캔들 수집."""
        try:
            if self.client:
                klines = self.client.futures_klines(symbol=symbol, interval=timeframe, limit=count)
            else:
                klines = self._public_klines(symbol, timeframe,
                    int((datetime.utcnow() - timedelta(minutes=count)).timestamp() * 1000))
            return self._klines_to_df(klines)
        except Exception as e:
            logger.error(f"최신 캔들 수집 실패: {e}")
            return pd.DataFrame()

    # ── 계좌 / 포지션 ─────────────────────────────────────────────────────

    def get_usdt_balance(self) -> float:
        """선물 지갑 USDT 잔고."""
        if not self.client:
            return 0.0
        try:
            for asset in self.client.futures_account_balance():
                if asset["asset"] == "USDT":
                    return float(asset["availableBalance"])
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
        return 0.0

    def get_total_usdt(self) -> float:
        """선물 지갑 총 USDT (미실현 손익 포함)."""
        if not self.client:
            return 0.0
        try:
            info = self.client.futures_account()
            return float(info["totalWalletBalance"])
        except Exception as e:
            logger.error(f"총 자산 조회 실패: {e}")
        return 0.0

    def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        try:
            if self.client:
                return float(self.client.futures_symbol_ticker(symbol=symbol)["price"])
            import requests
            r = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}", timeout=5)
            return float(r.json()["price"])
        except Exception as e:
            logger.error(f"현재가 조회 실패: {e}")
            return 0.0

    def get_position(self, symbol: str = "BTCUSDT") -> Optional[dict]:
        """현재 선물 포지션 조회 (없으면 None)."""
        if not self.client:
            return None
        try:
            for pos in self.client.futures_position_information(symbol=symbol):
                amt = float(pos["positionAmt"])
                if abs(amt) > 0:
                    return {
                        "side":        "LONG" if amt > 0 else "SHORT",
                        "size":        abs(amt),
                        "entry_price": float(pos.get("entryPrice", 0)),
                        "unrealized_pnl": float(pos.get("unRealizedProfit", 0)),
                        "leverage":    int(pos.get("leverage", 1)),
                    }
        except Exception as e:
            logger.error(f"포지션 조회 실패: {e}")
        return None

    # ── 주문 ──────────────────────────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int):
        if not self.client:
            return
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
        except Exception as e:
            logger.warning(f"레버리지 설정 실패: {e}")

    def open_long(self, symbol: str, usdt_amount: float) -> Optional[dict]:
        """롱 진입 (시장가, USDT 금액 기준)."""
        if not self.client:
            raise RuntimeError("API 키 미설정")
        price    = self.get_current_price(symbol)
        quantity = self._calc_quantity(symbol, usdt_amount, price)
        try:
            order = self.client.futures_create_order(
                symbol=symbol, side="BUY",
                type="MARKET", quantity=quantity,
            )
            logger.info(f"롱 진입: {quantity} {symbol} @ {price:,.2f} ({usdt_amount:.2f} USDT)")
            return order
        except BinanceAPIException as e:
            logger.error(f"롱 주문 실패: {e}")
            raise

    def open_short(self, symbol: str, usdt_amount: float) -> Optional[dict]:
        """숏 진입 (시장가, USDT 금액 기준)."""
        if not self.client:
            raise RuntimeError("API 키 미설정")
        price    = self.get_current_price(symbol)
        quantity = self._calc_quantity(symbol, usdt_amount, price)
        try:
            order = self.client.futures_create_order(
                symbol=symbol, side="SELL",
                type="MARKET", quantity=quantity,
            )
            logger.info(f"숏 진입: {quantity} {symbol} @ {price:,.2f} ({usdt_amount:.2f} USDT)")
            return order
        except BinanceAPIException as e:
            logger.error(f"숏 주문 실패: {e}")
            raise

    def close_position(self, symbol: str) -> Optional[dict]:
        """현재 포지션 전량 청산."""
        if not self.client:
            raise RuntimeError("API 키 미설정")
        pos = self.get_position(symbol)
        if not pos:
            return None
        side = "SELL" if pos["side"] == "LONG" else "BUY"
        try:
            order = self.client.futures_create_order(
                symbol=symbol, side=side,
                type="MARKET", quantity=pos["size"],
                reduceOnly=True,
            )
            logger.info(f"포지션 청산: {pos['side']} {pos['size']} @ {self.get_current_price(symbol):,.2f}")
            return order
        except BinanceAPIException as e:
            logger.error(f"청산 실패: {e}")
            raise

    def _calc_quantity(self, symbol: str, usdt_amount: float, price: float) -> float:
        """USDT 금액 → BTC 수량 변환 (최소 수량 단위 맞춤)."""
        quantity = usdt_amount / price
        # BTC 선물 최소 단위: 0.001
        precision = 3
        return round(quantity, precision)
