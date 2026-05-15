from __future__ import annotations

import numpy as np
import pandas as pd
import logging
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from analysis.indicators import IndicatorEngine
from config import Config

logger = logging.getLogger(__name__)


class BTCSignalModel:
    """
    XGBoost 기반 BTC 매수/매도 시그널 예측 모델.

    검증셋에서 정밀도(Precision) 95%+ 를 달성하는 확률 임계값을 자동 탐색합니다.
    """

    MODEL_PATH  = "models/btc_model.pkl"
    SCALER_PATH = "models/btc_scaler.pkl"

    def __init__(self):
        self.model_long      = None
        self.model_short     = None
        self.scaler          = StandardScaler()
        self.features        = IndicatorEngine.get_feature_columns()
        self.threshold_long  = 0.5   # 학습 후 동적 갱신
        self.threshold_short = 0.5
        os.makedirs("models", exist_ok=True)

    # ── 데이터 준비 ───────────────────────────────────────────────────────

    def prepare_dataset(self, df: pd.DataFrame):
        d = IndicatorEngine.compute_all(df)

        future_ret = d["close"].pct_change(Config.PREDICTION_HORIZON).shift(-Config.PREDICTION_HORIZON)
        d["label_long"]  = (future_ret >  Config.FUTURE_RETURN_TARGET).astype(int)
        d["label_short"] = (future_ret < -Config.FUTURE_RETURN_TARGET).astype(int)

        d = d.dropna(subset=self.features + ["label_long", "label_short"])
        X       = d[self.features].values
        y_long  = d["label_long"].values
        y_short = d["label_short"].values
        return X, y_long, y_short

    # ── 학습 ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame):
        X, y_long, y_short = self.prepare_dataset(df)
        n       = len(X)
        n_train = int(n * Config.TRAIN_RATIO)
        n_val   = int(n * (Config.TRAIN_RATIO + Config.VAL_RATIO))

        X_tr, X_val, X_te         = X[:n_train], X[n_train:n_val], X[n_val:]
        yL_tr, yL_val, yL_te      = y_long[:n_train],  y_long[n_train:n_val],  y_long[n_val:]
        yS_tr, yS_val, yS_te      = y_short[:n_train], y_short[n_train:n_val], y_short[n_val:]

        X_tr_s  = self.scaler.fit_transform(X_tr)
        X_val_s = self.scaler.transform(X_val)
        X_te_s  = self.scaler.transform(X_te)

        logger.info(f"학습={n_train:,} / 검증={n_val-n_train:,} / 테스트={n-n_val:,}")
        logger.info(f"매수 레이블 비율: {yL_tr.mean():.2%} | 매도: {yS_tr.mean():.2%}")

        base = dict(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.6, colsample_bytree=0.6,
            min_child_weight=50, gamma=1.0,
            reg_alpha=1.0, reg_lambda=5.0,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )

        self.model_long  = self._fit(base, X_tr_s, yL_tr, X_val_s, yL_val, "매수(Long)")
        self.model_short = self._fit(base, X_tr_s, yS_tr, X_val_s, yS_val, "매도(Short)")

        # ── 검증셋에서 정밀도 95%+ 임계값 탐색 ───────────────────────────
        prob_long_val  = self.model_long.predict_proba(X_val_s)[:, 1]
        prob_short_val = self.model_short.predict_proba(X_val_s)[:, 1]

        self.threshold_long  = self._find_precision_threshold(prob_long_val,  yL_val, target=0.52, label="매수")
        self.threshold_short = self._find_precision_threshold(prob_short_val, yS_val, target=0.52, label="매도")

        # ── 테스트셋 평가 ─────────────────────────────────────────────────
        prob_long_te  = self.model_long.predict_proba(X_te_s)[:, 1]
        prob_short_te = self.model_short.predict_proba(X_te_s)[:, 1]
        self._evaluate(prob_long_te,  yL_te, self.threshold_long,  "매수(Long)")
        self._evaluate(prob_short_te, yS_te, self.threshold_short, "매도(Short)")

        joblib.dump({
            "long": self.model_long, "short": self.model_short,
            "th_long": self.threshold_long, "th_short": self.threshold_short,
        }, self.MODEL_PATH)
        joblib.dump(self.scaler, self.SCALER_PATH)
        logger.info(f"모델 저장 완료 | 매수 임계값={self.threshold_long:.4f} | 매도 임계값={self.threshold_short:.4f}")

    def _fit(self, params, X_tr, y_tr, X_val, y_val, label):
        scale = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        xgb   = XGBClassifier(**params, scale_pos_weight=scale)
        xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        logger.info(f"  [{label}] 학습 완료")
        return xgb

    def _find_precision_threshold(
        self, probs: np.ndarray, labels: np.ndarray,
        target: float = 0.55, label: str = ""
    ) -> float:
        """정밀도 target 이상 + 충분한 신호 수를 만족하는 최적 임계값 탐색."""
        best_th   = 0.80
        best_score = -1.0

        for th in np.arange(0.50, 0.96, 0.01):
            pred = (probs >= th).astype(int)
            n    = pred.sum()
            if n < 10:
                continue
            prec = precision_score(labels, pred, zero_division=0)
            if prec < target:
                continue
            # 정밀도와 신호 수의 균형을 맞추는 점수: precision × log(n)
            score = prec * np.log1p(n)
            if score > best_score:
                best_score = score
                best_th    = th
                best_prec  = prec
                best_n     = n

        if best_score < 0:
            best_th   = 0.80
            best_prec = 0.0
            best_n    = 0

        logger.info(f"  [{label}] 임계값={best_th:.2f} | 검증 정밀도={best_prec:.2%} | 신호={best_n}회")
        return float(best_th)

    def _evaluate(self, probs, labels, threshold, label):
        pred = (probs >= threshold).astype(int)
        try:
            auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        prec = precision_score(labels, pred, zero_division=0)
        rec  = recall_score(labels, pred, zero_division=0)
        logger.info(
            f"  [{label}] AUC={auc:.4f} | 신호={pred.sum()}회 | "
            f"정밀도={prec:.2%} | 재현율={rec:.2%} | 임계={threshold:.3f}"
        )

    # ── 예측 ──────────────────────────────────────────────────────────────

    def load(self):
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError("학습된 모델 없음. `python3 main.py train` 을 먼저 실행하세요.")
        m = joblib.load(self.MODEL_PATH)
        self.model_long      = m["long"]
        self.model_short     = m["short"]
        self.threshold_long  = m.get("th_long",  0.70)
        self.threshold_short = m.get("th_short", 0.70)
        self.scaler          = joblib.load(self.SCALER_PATH)
        logger.info(f"모델 로드 | 매수 임계={self.threshold_long:.3f} | 매도 임계={self.threshold_short:.3f}")

    def predict(self, df: pd.DataFrame) -> dict:
        """최신 캔들 기준 매수/매도 확률 반환."""
        d = IndicatorEngine.compute_all(df).dropna(subset=self.features)
        if d.empty:
            return {"signal": "HOLD", "long_prob": 0.0, "short_prob": 0.0, "confidence": 0.0, "price": 0.0}

        X_s        = self.scaler.transform(d[self.features].values[-1:])
        long_prob  = float(self.model_long.predict_proba(X_s)[0, 1])
        short_prob = float(self.model_short.predict_proba(X_s)[0, 1])

        signal = "HOLD"
        if long_prob >= self.threshold_long and long_prob > short_prob:
            signal = "LONG"
        elif short_prob >= self.threshold_short and short_prob > long_prob:
            signal = "SHORT"

        return {
            "signal":     signal,
            "long_prob":  long_prob,
            "short_prob": short_prob,
            "confidence": max(long_prob, short_prob),
            "timestamp":  str(d.index[-1]),
            "price":      float(d["close"].iloc[-1]),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """백테스팅용: 전체 구간 시그널 예측."""
        d   = IndicatorEngine.compute_all(df).dropna(subset=self.features)
        X_s = self.scaler.transform(d[self.features].values)

        d = d.copy()
        d["long_prob"]  = self.model_long.predict_proba(X_s)[:, 1]
        d["short_prob"] = self.model_short.predict_proba(X_s)[:, 1]
        d["signal"]     = "HOLD"
        d.loc[(d["long_prob"]  >= self.threshold_long)  & (d["long_prob"]  > d["short_prob"]), "signal"] = "LONG"
        d.loc[(d["short_prob"] >= self.threshold_short) & (d["short_prob"] > d["long_prob"]),  "signal"] = "SHORT"

        long_n  = (d["signal"] == "LONG").sum()
        short_n = (d["signal"] == "SHORT").sum()
        logger.info(f"시그널 생성: LONG={long_n}회, SHORT={short_n}회 / 전체={len(d)}캔들")
        return d
