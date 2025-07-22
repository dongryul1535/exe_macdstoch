#!/usr/bin/env python3
"""
DART 임원·주요주주 특정증권등소유상황보고서 + MACD/Stoch 교차 신호 알림 봇
────────────────────────────────────────────────────────────────────
1) 오늘(한국시간) 공시된 "임원·주요주주특정증권등소유상황보고서"만 필터링
2) 해당 기업의 종목코드 → FinanceDataReader(FDR)로 가격 조회
3) Composite K / D 계산
   - Composite K = MACD(12,26) + Slow %K(14,3)
   - Composite D = MACD Signal(9) + Slow %D(14,3)
4) 골든/데드 크로스 탐지 → 텔레그램으로 텍스트 + 차트 이미지 전송
5) GitHub Actions에서 주기 실행 (requirements.txt, workflow 포함)

환경 변수
---------
- TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (필수)
- DART_API_KEY (필수)
- SCALE_MACD=true  → MACD 계열 0~100 정규화(선택)
- SAVE_CSV=true    → CSV 저장(선택)
- FONT_PATH=/path/to/NanumGothic.ttf  → 한글 폰트 지정(선택)

파일 구성 예시
--------------
- main.py                 ← 본 스크립트
- requirements.txt        ← 의존성
- .github/workflows/run.yml

"""

import os
import io
import zipfile
import logging
import datetime as dt
from typing import List, Optional, Dict
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import FinanceDataReader as fdr

from ta.trend import MACD
from ta.momentum import StochasticOscillator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.font_manager as fm

# ───────────────────────── 기본 설정 ───────────────────────── #
KST = dt.timezone(dt.timedelta(hours=9))
TODAY = dt.datetime.now(KST).strftime('%Y%m%d')

TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")
DART_KEY = os.getenv("DART_API_KEY")
SCALE_MACD = os.getenv("SCALE_MACD", "false").lower() == "true"
SAVE_CSV   = os.getenv("SAVE_CSV",   "false").lower() == "true"
FONT_PATH  = os.getenv("FONT_PATH", "")
# ▶ 테스트용 날짜 보정: 0=오늘, 1=어제, 2=그제 …
DART_OFFSET_DAYS = int(os.getenv("DART_OFFSET_DAYS", "0"))

if not (TOKEN and CHAT_ID and DART_KEY):
    raise SystemExit("필수 환경변수 누락: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DART_API_KEY")

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Font
if FONT_PATH and os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    font_prop = None

# ─────────────────── DART / KRX 유틸 ─────────────────── #
DART_URL = "https://opendart.fss.or.kr/api"
CORP_CODE_URL = f"{DART_URL}/corpCode.xml"

_cache_corp_map: Optional[Dict[str, Dict[str,str]]] = None

def load_corp_map() -> Dict[str, Dict[str, str]]:
    """{stock_code: {corp_code, corp_name}} 매핑 생성"""
    global _cache_corp_map
    if _cache_corp_map is not None:
        return _cache_corp_map
    params = {'crtfc_key': DART_KEY}
    resp = requests.get(CORP_CODE_URL, params=params, timeout=20)
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    xml_bytes = zf.read(zf.namelist()[0])
    root = ET.fromstring(xml_bytes)
    mapping = {}
    for corp in root.findall('list'):
        stock = corp.findtext('stock_code') or ''
        corp_code = corp.findtext('corp_code') or ''
        corp_name = corp.findtext('corp_name') or ''
        if stock:
            stock = stock.zfill(6)
            mapping[stock] = {"corp_code": corp_code, "corp_name": corp_name}
    _cache_corp_map = mapping
    return mapping

# 이름 매핑 (KRX,KOSDAQ)
_krx = fdr.StockListing('KRX')[['Code','Name']]
_kq  = fdr.StockListing('KOSDAQ')[['Code','Name']]
NAME_MAP = {f"{r.Code}.KS": r.Name for _, r in _krx.iterrows()}
NAME_MAP.update({f"{r.Code}.KQ": r.Name for _, r in _kq.iterrows()})

def get_name(code: str) -> str:
    return NAME_MAP.get(code, code)

# ─────────────────── 공시 수집 & 필터 ─────────────────── #
TARGET_KEYWORDS = ["임원", "주요주주", "특정증권등소유상황보고서"]
EXCLUDE_KEYWORDS = ["정정", "변경", "취소", "신규선임", "해임", "사임", "퇴임", "임원현황", "의결권"]


def ymd(days_offset: int = 0) -> str:
    return (dt.datetime.now(KST) - dt.timedelta(days=days_offset)).strftime('%Y%m%d')


def fetch_list(days_offset: int = 0) -> List[dict]:
    """특정 날짜(오프셋) 공시 목록 (다중 페이지)"""
    bgn_de = ymd(days_offset)
    end_de = bgn_de
    all_rows: List[dict] = []
    for page in range(1, 11):
        params = {
            'crtfc_key': DART_KEY,
            'bgn_de': bgn_de,
            'end_de': end_de,
            'page_no': page,
            'page_count': 100
        }
        r = requests.get(f"{DART_URL}/list.json", params=params, timeout=20)
        if r.status_code != 200:
            logging.warning("DART list HTTP %s", r.status_code)
            break
        data = r.json()
        if data.get('status') != '000':
            logging.warning("DART list status %s", data.get('status'))
            break
        rows = data.get('list', [])
        all_rows.extend(rows)
        if len(rows) < 100:
            break
        time.sleep(0.3)
    logging.info("%s 공시 %d건 수집", bgn_de, len(all_rows))
    return all_rows


def is_target_report(report_nm: str) -> bool:
    name = (report_nm or "").replace('·', 'ㆍ')  # dot normalization
    if not all(k in name for k in TARGET_KEYWORDS):
        return False
    if any(k in name for k in EXCLUDE_KEYWORDS):
        return False
    return True


def filter_target_disclosures(rows: List[dict]) -> List[dict]:
    results = []
    for item in rows:
        if is_target_report(item.get('report_nm', '')):
            results.append(item)
    logging.info("타깃 공시 %d건", len(results))
    return results

# ─────────────────── 시세/지표 계산 ─────────────────── #

def fetch_daily(symbol: str, days: int = 180) -> Optional[pd.DataFrame]:
    end = dt.datetime.now()
    start = end - dt.timedelta(days=days)
    try:
        df = fdr.DataReader(symbol, start, end)
        if df.empty:
            return None
        df = df.reset_index()
        df.rename(columns=str.capitalize, inplace=True)
        return df[['Date','Open','High','Low','Close','Volume']]
    except Exception as e:
        logging.warning("%s 데이터 조회 실패: %s", symbol, e)
        return None


def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    st   = StochasticOscillator(df['Close'], df['High'], df['Low'], window=14, smooth_window=3)
    df['MACD']     = macd.macd()
    df['MACD_SIG'] = macd.macd_signal()
    df['SlowK']    = st.stoch()
    df['SlowD']    = st.stoch_signal()

    m  = df['MACD']
    ms = df['MACD_SIG']
    if SCALE_MACD:
        m  = (m  - m.min())  / (m.max()  - m.min())  * 100
        ms = (ms - ms.min()) / (ms.max() - ms.min()) * 100

    df['CompK'] = m  + df['SlowK']
    df['CompD'] = ms + df['SlowD']
    df['Diff']  = df['CompK'] - df['CompD']
    return df


def detect_cross(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 2:
        return None
    prev, curr = df['Diff'].iloc[-2], df['Diff'].iloc[-1]
    if prev <= 0 < curr:
        return 'BUY'
    if prev >= 0 > curr:
        return 'SELL'
    return None

# ─────────────────── 시각화 ─────────────────── #

def make_chart(df: pd.DataFrame, code: str) -> str:
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,6), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    name = get_name(code)
    ax1.plot(df['Date'], df['Close'], label='종가')
    ax1.plot(df['Date'], df['Close'].rolling(20).mean(), linestyle='--', label='MA20')
    ax1.set_title(f"{code} ({name})", fontproperties=font_prop)
    ax1.legend(prop=font_prop)

    ax2.plot(df['Date'], df['CompK'], label='CompK')
    ax2.plot(df['Date'], df['CompD'], label='CompD')
    ax2.axhline(0, color='gray', linewidth=0.7)
    ax2.set_title('Composite Cross', fontproperties=font_prop)
    ax2.legend(prop=font_prop)
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    fig.autofmt_xdate()
    fig.tight_layout()
    path = f"{code}_chart.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

# ─────────────────── 텔레그램 ─────────────────── #

def tg_text(msg: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for chunk in [msg[i:i+3500] for i in range(0, len(msg), 3500)]:
        requests.post(url, json={'chat_id': CHAT_ID, 'text': chunk})
        time.sleep(0.3)


def tg_photo(path: str, caption: str = ''):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(path, 'rb') as f:
        requests.post(url, data={'chat_id': CHAT_ID, 'caption': caption}, files={'photo': f})
    time.sleep(0.3)

# ─────────────────── 메인 ─────────────────── #

def main():
    logging.info("==== 시작: %s ====" , dt.datetime.now(KST))

    corp_map = load_corp_map()  # stock_code → corp_code/name
    rows = fetch_list(DART_OFFSET_DAYS)
    targets = filter_target_disclosures(rows)

    if not targets:
        logging.info("타깃 공시 없음")
        tg_text(f"{ymd(DART_OFFSET_DAYS)} 임원·주요주주 특정증권등소유상황보고서 공시 없음")
        return

    alerts: List[str] = []

    for item in targets:
        corp_name = item.get('corp_name', '')
        corp_code = item.get('corp_code', '')
        rcept_dt  = item.get('rcept_dt', '')
        rcept_no  = item.get('rcept_no', '')
        report_nm = item.get('report_nm', '')

        # stock_code 찾기
        stock_code = None
        for scode, info in corp_map.items():
            if info['corp_code'] == corp_code:
                stock_code = scode
                break
        if not stock_code:
            logging.warning("%s(%s) stock_code 없음", corp_name, corp_code)
            continue

        # .KS / .KQ 판별
        suffix = '.KS' if stock_code in NAME_MAP and NAME_MAP.get(f"{stock_code}.KS") else '.KQ'
        code = f"{stock_code}{suffix}"

        df = fetch_daily(stock_code)
        if df is None or len(df) < 40:
            logging.warning("%s 데이터 부족", code)
            continue
        df = add_composites(df)
        sig = detect_cross(df)

        caption = (f"{corp_name} ({code})\n"
                   f"📄 {report_nm}\n"
                   f"📅 {rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}\n"
                   f"🔗 DART: https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}")
        if sig:
            caption = f"[{sig}]\n" + caption
            alerts.append(f"{sig} - {corp_name} ({code})")

        img = make_chart(df.tail(120), code)
        tg_photo(img, caption=caption)
        if SAVE_CSV:
            df.to_csv(f"{code}_hist.csv", index=False)

    if alerts:
        tg_text("\n".join(alerts))
    else:
        tg_text("신호 없음 (골든/데드 크로스 미발생)")

    logging.info("==== 종료 ====\n")


if __name__ == '__main__':
    main()
