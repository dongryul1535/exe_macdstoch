#!/usr/bin/env python3
"""
DART 임원·주요주주 특정증권등소유상황보고서 + NH MTS-style MACD+Stochastic 알림 봇
────────────────────────────────────────────────────────────────────
1) 오늘(한국시간) 공시된 "임원·주요주주특정증권등소유상황보고서"만 필터링
2) 해당 기업 종목코드로 가격 조회 (FinanceDataReader → yfinance fallback 可 추가 가능)
3) **NH 나무 MTS 방식 Composite K / D 계산**
   - MACD_raw = EMA(12) - EMA(26)
   - MACD_norm = 14기간 스토캐스틱(0~100)화 후 3기간 스무딩
   - Slow%K   = 가격기반 Stochastic(14,3)
   - Composite K = (MACD_norm + Slow%K) / 2
   - Composite D = SMA(Composite K, 3)
4) 골든/데드 크로스 탐지 → 텔레그램 텍스트 + 차트 이미지 전송
5) GitHub Actions에서 주기 실행

ENV
----
- TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DART_API_KEY (필수)
- DART_OFFSET_DAYS: 0=오늘, 1=어제 ... (기본 0)
- SAVE_CSV=true  → CSV 저장
- FONT_PATH=fonts/NanumGothic.ttf  → 한글 폰트

requirements.txt (예시)
-----------------------
numpy>=1.24.0
pandas>=1.5.3
requests>=2.28.2
finance-datareader>=0.9.59
yfinance>=0.2.40
matplotlib>=3.8.4
"""

import os
import io
import zipfile
import logging
import datetime as dt
from typing import List, Optional, Dict
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import FinanceDataReader as fdr

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
SAVE_CSV = os.getenv("SAVE_CSV",   "false").lower() == "true"
FONT_PATH  = os.getenv("FONT_PATH", "")
DART_OFFSET_DAYS = int(os.getenv("DART_OFFSET_DAYS", "0"))

if not (TOKEN and CHAT_ID and DART_KEY):
    raise SystemExit("필수 환경변수 누락: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DART_API_KEY")

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
    name = (report_nm or "").replace('·', 'ㆍ')
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
try:
    import yfinance as yf
except Exception:
    yf = None

def fetch_daily(symbol: str, days: int = 180) -> Optional[pd.DataFrame]:
    end = dt.datetime.now()
    start = end - dt.timedelta(days=days)
    # 우선 FDR
    try:
        df = fdr.DataReader(symbol, start, end)
        if not df.empty:
            df = df.reset_index()
            df.rename(columns=str.capitalize, inplace=True)
            return df[['Date','Open','High','Low','Close','Volume']]
    except Exception:
        pass
    # yfinance fallback
    if yf is not None:
        try:
            ydf = yf.download(f"{symbol}.KS", start=start.date(), end=end.date(), progress=False)
            if not ydf.empty:
                ydf = ydf.rename(columns=str.title).reset_index()
                return ydf[['Date','Open','High','Low','Close','Volume']]
        except Exception:
            pass
    return None

# NH 스타일 Composite

def add_composites(df: pd.DataFrame,
                   fast: int = 12, slow: int = 26,
                   k_window: int = 14, k_smooth: int = 3,
                   d_smooth: int = 3, use_ema: bool = True,
                   clip: bool = True) -> pd.DataFrame:
    close, high, low = df['Close'], df['High'], df['Low']

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_raw = ema_fast - ema_slow

    macd_min = macd_raw.rolling(k_window, min_periods=1).min()
    macd_max = macd_raw.rolling(k_window, min_periods=1).max()
    macd_norm = (macd_raw - macd_min) / (macd_max - macd_min).replace(0, np.nan) * 100
    macd_norm = macd_norm.fillna(50)
    if k_smooth > 1:
        macd_norm = macd_norm.ewm(span=k_smooth, adjust=False).mean() if use_ema \
            else macd_norm.rolling(k_smooth, min_periods=1).mean()

    ll = low.rolling(k_window, min_periods=1).min()
    hh = high.rolling(k_window, min_periods=1).max()
    k_raw = (close - ll) / (hh - ll).replace(0, np.nan) * 100
    k_raw = k_raw.fillna(50)
    slow_k = (k_raw.ewm(span=k_smooth, adjust=False).mean() if (k_smooth > 1 and use_ema)
              else k_raw.rolling(k_smooth, min_periods=1).mean() if k_smooth > 1 else k_raw)

    comp_k = (macd_norm + slow_k) / 2.0
    comp_d = comp_k.rolling(d_smooth, min_periods=1).mean() if d_smooth > 1 else comp_k

    if clip:
        comp_k = comp_k.clip(0, 100)
        comp_d = comp_d.clip(0, 100)

    df['CompK'] = comp_k
    df['CompD'] = comp_d
    df['Diff']  = comp_k - comp_d
    return df

def detect_cross(df: pd.DataFrame, ob: int = 80, os: int = 20) -> Optional[str]:
    if len(df) < 2:
        return None
    prev, curr = df['Diff'].iloc[-2], df['Diff'].iloc[-1]
    prev_k = df['CompK'].iloc[-2]
    if prev <= 0 < curr:
        return 'BUY' if prev_k < os else 'BUY_W'
    if prev >= 0 > curr:
        return 'SELL' if prev_k > ob else 'SELL_W'
    return None

# ─────────────────── 시각화 ─────────────────── #

def make_chart(df: pd.DataFrame, code: str) -> str:
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,6), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    name = get_name(code)
    ax1.plot(df['Date'], df['Close'], label='종가')
    ax1.plot(df['Date'], df['Close'].rolling(20).mean(), linestyle='--', label='MA20')
    ax1.set_title(f"{code} ({name})", fontproperties=font_prop)
    ax1.legend(prop=font_prop)

    ax2.plot(df['Date'], df['CompK'], color='red', label='MACD+Slow%K')
    ax2.plot(df['Date'], df['CompD'], color='purple', label='MACD+Slow%D')
    ax2.axhline(20, color='gray', linestyle='--', linewidth=0.5)
    ax2.axhline(80, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_title('MACD+Stochastic (NH Style)', fontproperties=font_prop)
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
        try:
            requests.post(url, json={'chat_id': CHAT_ID, 'text': chunk}, timeout=15)
        except Exception as e:
            logging.warning("텍스트 전송 실패: %s", e)
        time.sleep(0.3)

def tg_photo(path: str, caption: str = ''):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    try:
        with open(path, 'rb') as f:
            requests.post(url, data={'chat_id': CHAT_ID, 'caption': caption}, files={'photo': f}, timeout=30)
    except Exception as e:
        logging.warning("사진 전송 실패: %s", e)
    time.sleep(0.3)

# ─────────────────── 메인 ─────────────────── #

def main():
    logging.info("==== 시작: %s ====", dt.datetime.now(KST))

    corp_map = load_corp_map()
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

        suffix = '.KS' if f"{stock_code}.KS" in NAME_MAP else '.KQ'
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

    logging.info("==== 종료 ====")


if __name__ == '__main__':
    main()
