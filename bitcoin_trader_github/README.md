# 업비트 BTC 자동 단타 트레이딩 봇 - 설정 가이드

## 1. 환경 설정

`.env.example`을 `.env`로 복사 후 API 키를 입력합니다.

```bash
cd ~/bitcoin_trader
cp .env.example .env
```

---

## 2. 텔레그램 봇 설정

### 봇 토큰 발급
1. 텔레그램에서 `@BotFather` 검색
2. `/newbot` 입력 → 이름/username 설정
3. 발급된 **HTTP API 토큰** 복사 → `.env`의 `TELEGRAM_BOT_TOKEN` 입력

### 채팅 ID 확인
1. 텔레그램에서 `@userinfobot` 검색 → `/start`
2. 표시된 **id** 숫자 복사 → `.env`의 `TELEGRAM_CHAT_ID` 입력

---

## 3. 업비트 API 설정

1. [업비트](https://upbit.com) 로그인 → **마이페이지** → **Open API 관리**
2. **API Key 발급** 클릭
3. 권한 설정:
   - ✅ **자산 조회** (필수)
   - ✅ **주문 조회** (필수)
   - ✅ **주문하기** (필수)
4. IP 화이트리스트: 봇을 실행할 서버/PC의 IP 입력
5. 발급된 **Access Key**, **Secret Key** 복사 → `.env` 입력

> ⚠️ 업비트는 선물/레버리지 거래 미지원 — **현물 KRW-BTC** 거래만 가능합니다.

---

## 4. 실행 순서

```bash
cd ~/bitcoin_trader

# ① 과거 7년 데이터 수집 (최초 1회, 약 15~25분 소요)
python3 main.py fetch --years 7

# ② ML 모델 학습 (약 5~10분)
python3 main.py train

# ③ 백테스팅 실행 (결과 + 차트 생성)
python3 main.py backtest --capital 1000000

# → backtest_report.png 파일로 차트 확인

# ④ 백테스팅 결과가 만족스러우면 라이브 봇 실행
python3 main.py live
```

한 번에 실행:
```bash
python3 main.py all
```

---

## 5. 트레이딩 로직

| 항목 | 값 | 설명 |
|------|-----|------|
| 신뢰도 임계값 | **95%** | ML 모델이 95% 이상 확신할 때만 진입 |
| 예측 타임프레임 | 12시간 후 | 12캔들(1H) 뒤 가격 방향 예측 |
| 손절 | **-2%** | 매수가 대비 2% 하락 시 자동 매도 |
| 익절 | **+4%** | 매수가 대비 4% 상승 시 자동 매도 |
| 포지션 크기 | KRW 잔고 30% | 거래당 보유 KRW의 30% 사용 |
| 일일 최대 손실 | **-5%** | 하루 5% 손실 시 추가 진입 금지 |
| 수수료 | 0.05% | 업비트 현물 거래 수수료 |

### 매매 방식
- **매수 시그널 (LONG)**: KRW → BTC 시장가 매수
- **매도 시그널 (SHORT)**: BTC → KRW 시장가 전량 매도
- 업비트는 공매도 불가 — 현금(KRW) 보유가 "숏" 포지션

---

## 6. ML 모델 피처 (28개)

| 분류 | 지표 |
|------|------|
| 이동평균 | MA7, MA20, MA50, MA100, MA200 |
| RSI | RSI14, RSI21, Stochastic RSI(K/D) |
| MACD | MACD, Signal, Histogram, 크로스 |
| 볼린저밴드 | 밴드폭, 현재 위치(0=하단~1=상단) |
| 변동성 | ATR14 (가격 대비 %) |
| 모멘텀 | 1h/4h/12h/24h/48h/7d 수익률 |
| 거래량 | 거래량 비율, OBV 시그널 |
| 복합 | 과매수/과매도 복합 지표 |

---

## 7. 파일 구조

```
bitcoin_trader/
├── main.py                  # 진입점 (fetch/train/backtest/live)
├── config.py                # 전체 설정
├── .env                     # API 키 (직접 작성)
├── data/
│   └── fetcher.py           # 업비트 데이터 수집 + 주문
├── analysis/
│   ├── indicators.py        # 기술 지표 계산 (28개 피처)
│   └── ml_model.py          # XGBoost + 확률 교정 모델
├── backtest/
│   └── engine.py            # 백테스팅 시뮬레이션 + 차트
├── trading/
│   └── bot.py               # 실거래 자동 봇
├── notifications/
│   └── telegram_bot.py      # 텔레그램 알림
├── models/                  # 학습된 모델 저장 (자동 생성)
└── data/cache/              # 데이터 캐시 parquet (자동 생성)
```

---

## 8. 텔레그램 알림 종류

| 알림 | 조건 |
|------|------|
| 🟢 매수 시그널 | ML 신뢰도 95%+ 매수 신호 (현재가, 익절/손절가 포함) |
| 🔴 매도 시그널 | ML 신뢰도 95%+ 매도 신호 |
| ✅ 익절 청산 | Take Profit 도달 |
| ❌ 손절 청산 | Stop Loss 도달 |
| 📊 일일 요약 | 매일 거래 결과 요약 |
| 🚨 에러 알림 | 봇 오류 발생 시 즉시 알림 |

---

## 주의사항

> ⚠️ **이 봇은 교육/연구 목적으로 제작되었습니다.**
> 암호화폐 투자는 원금 손실 위험이 있습니다.
> 반드시 소액으로 시작하고, 백테스팅 결과를 충분히 검토 후 사용하세요.
