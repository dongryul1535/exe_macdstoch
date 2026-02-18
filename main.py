#!/usr/bin/env python3
"""
DART 임원·주요주주 특정증권등소유상황보고서 + NH MTS-style MACD+Stochastic 알림 봇
────────────────────────────────────────────────────────────────────
■ 차트: 1열×4행 (일봉 80일 캔들 → 일봉 지표 → 주봉 40주 캔들 → 주봉 지표)
■ NH MTS: 단기12/장기26/K1=14/K2=3/D=3/기준선 20·80
■ 양봉 빨강, 음봉 파랑 / MACD+Slow%K 빨강, MACD+Slow%D 보라
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
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

# ───────────────────────── 기본 설정 ───────────────────────── #
KST = dt.timezone(dt.timedelta(hours=9))
TODAY = dt.datetime.now(KST).strftime('%Y%m%d')

TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")
DART_KEY = os.getenv("DART_API_KEY")
SAVE_CSV = os.getenv("SAVE_CSV", "false").lower() == "true"
FONT_PATH = os.getenv("FONT_PATH", "")
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
    plt.rcParams['axes.unicode_minus'] = False
    font_prop = None

# ───── NH MTS 설정값 ─────
FAST_PERIOD = 12
SLOW_PERIOD = 26
K_WINDOW    = 14
K_SMOOTH    = 3
D_SMOOTH    = 3
OB_LINE     = 80
OS_LINE     = 20
DAILY_BARS  = 80
WEEKLY_BARS = 40

COLOR_K = "#FF0000"
COLOR_D = "#9900FF"

# ─────────────────── DART / KRX 유틸 ─────────────────── #
DART_URL = "https://opendart.fss.or.kr/api"
CORP_CODE_URL = f"{DART_URL}/corpCode.xml"

_cache_corp_map: Optional[Dict[str, Dict[str, str]]] = None

def load_corp_map() -> Dict[str, Dict[str, str]]:
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

_krx = fdr.StockListing('KRX')[['Code', 'Name']]
_kq  = fdr.StockListing('KOSDAQ')[['Code', 'Name']]
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

# ─────────────────── 시세 조회 ─────────────────── #
try:
    import yfinance as yf
except Exception:
    yf = None

def fetch_daily(symbol: str, days: int = 500) -> Optional[pd.DataFrame]:
    """주봉 40주 + 지표 워밍업 위해 약 500일치"""
    end = dt.datetime.now()
    start = end - dt.timedelta(days=days)
    try:
        df = fdr.DataReader(symbol, start, end)
        if not df.empty:
            df = df.reset_index()
            df.rename(columns=str.capitalize, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        pass
    if yf is not None:
        try:
            ydf = yf.download(f"{symbol}.KS", start=start.date(), end=end.date(), progress=False)
            if not ydf.empty:
                ydf = ydf.rename(columns=str.title).reset_index()
                ydf['Date'] = pd.to_datetime(ydf['Date'])
                return ydf[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception:
            pass
    return None

# ─────────────────── 주봉 리샘플링 ─────────────────── #
def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp['Date'] = pd.to_datetime(tmp['Date'])
    tmp = tmp.set_index('Date')
    weekly = tmp.resample('W-FRI').agg({
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum',
    }).dropna(subset=['Close'])
    return weekly.reset_index()

# ─────────────────── NH 스타일 Composite ─────────────────── #
def add_composites(df: pd.DataFrame,
                   fast=FAST_PERIOD, slow=SLOW_PERIOD,
                   k_window=K_WINDOW, k_smooth=K_SMOOTH,
                   d_smooth=D_SMOOTH) -> pd.DataFrame:
    close = df['Close'].astype(float)
    high  = df['High'].astype(float)
    low   = df['Low'].astype(float)

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_raw = ema_fast - ema_slow

    macd_min = macd_raw.rolling(k_window, min_periods=1).min()
    macd_max = macd_raw.rolling(k_window, min_periods=1).max()
    denom = (macd_max - macd_min).replace(0, np.nan)
    macd_norm = ((macd_raw - macd_min) / denom * 100).fillna(50)
    if k_smooth > 1:
        macd_norm = macd_norm.ewm(span=k_smooth, adjust=False).mean()

    ll = low.rolling(k_window, min_periods=1).min()
    hh = high.rolling(k_window, min_periods=1).max()
    stoch_denom = (hh - ll).replace(0, np.nan)
    k_raw = ((close - ll) / stoch_denom * 100).fillna(50)
    slow_k = k_raw.ewm(span=k_smooth, adjust=False).mean() if k_smooth > 1 else k_raw

    comp_k = ((macd_norm + slow_k) / 2.0).clip(0, 100)
    comp_d = comp_k.rolling(d_smooth, min_periods=1).mean().clip(0, 100)

    df = df.copy()
    df['CompK'] = comp_k
    df['CompD'] = comp_d
    df['Diff']  = comp_k - comp_d
    return df

def detect_cross(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 2:
        return None
    prev, curr = df['Diff'].iloc[-2], df['Diff'].iloc[-1]
    prev_k = df['CompK'].iloc[-2]
    if prev <= 0 < curr:
        return 'BUY' if prev_k < OS_LINE else 'BUY_W'
    if prev >= 0 > curr:
        return 'SELL' if prev_k > OB_LINE else 'SELL_W'
    return None

# ─────────────────── 캔들스틱 ─────────────────── #
def draw_candlestick(ax, df, width_ratio=0.6):
    dates = mdates.date2num(pd.to_datetime(df['Date']))
    if len(dates) >= 2:
        avg_gap = np.median(np.diff(dates))
    else:
        avg_gap = 1.0
    bar_width = avg_gap * width_ratio

    opens  = df['Open'].values.astype(float)
    highs  = df['High'].values.astype(float)
    lows   = df['Low'].values.astype(float)
    closes = df['Close'].values.astype(float)

    for i in range(len(dates)):
        d = dates[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        if c >= o:
            color = '#FF3232'
            body_bottom = o
            body_height = c - o
        else:
            color = '#3232FF'
            body_bottom = c
            body_height = o - c

        ax.plot([d, d], [l, h], color=color, linewidth=0.7, solid_capstyle='round')

        if body_height == 0:
            ax.plot([d - bar_width / 2, d + bar_width / 2], [o, o],
                    color=color, linewidth=1.0)
        else:
            rect = Rectangle(
                (d - bar_width / 2, body_bottom),
                bar_width, body_height,
                facecolor=color, edgecolor=color, linewidth=0.5
            )
            ax.add_patch(rect)

    ax.xaxis_date()

# ─────────────────── 패널 그리기 ─────────────────── #
def _plot_panel(ax_candle, ax_ind, df, title, date_fmt):
    dates_dt  = pd.to_datetime(df['Date'])
    dates_num = mdates.date2num(dates_dt)
    close     = df['Close'].astype(float)

    # 캔들 + 이평선
    draw_candlestick(ax_candle, df)
    ma5  = close.rolling(5,  min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ax_candle.plot(dates_num, ma5,  color='#FF8C00', linewidth=0.8, label='MA5')
    ax_candle.plot(dates_num, ma20, color='#1E90FF', linewidth=0.8,
                   linestyle='--', label='MA20')

    ax_candle.set_title(title, fontproperties=font_prop, fontsize=10, fontweight='bold')
    ax_candle.legend(prop=font_prop, fontsize=7, loc='upper left')
    ax_candle.tick_params(axis='both', labelsize=7)
    ax_candle.grid(True, alpha=0.25)
    ax_candle.set_xlim(dates_num[0] - 1, dates_num[-1] + 1)

    # MACD+Stochastic 지표
    ax_ind.plot(dates_num, df['CompK'].values, color=COLOR_K, linewidth=1.0,
                label='MACD+Slow%K')
    ax_ind.plot(dates_num, df['CompD'].values, color=COLOR_D, linewidth=1.0,
                label='MACD+Slow%D')
    ax_ind.axhline(OS_LINE, color='gray', linestyle='--', linewidth=0.5)
    ax_ind.axhline(OB_LINE, color='gray', linestyle='--', linewidth=0.5)
    ax_ind.fill_between(dates_num, 0,       OS_LINE, alpha=0.06, color='blue')
    ax_ind.fill_between(dates_num, OB_LINE, 100,     alpha=0.06, color='red')
    ax_ind.set_ylim(0, 100)
    ax_ind.set_ylabel('MACD+Stoch', fontsize=7)
    ax_ind.legend(prop=font_prop, fontsize=6, loc='upper left')
    ax_ind.tick_params(axis='both', labelsize=7)
    ax_ind.grid(True, alpha=0.25)
    ax_ind.set_xlim(dates_num[0] - 1, dates_num[-1] + 1)
    ax_ind.xaxis.set_major_formatter(DateFormatter(date_fmt))

    last_k = df['CompK'].iloc[-1]
    last_d = df['CompD'].iloc[-1]
    ax_ind.annotate(f'{last_k:.1f}', xy=(dates_num[-1], last_k),
                    fontsize=7, color=COLOR_K, fontweight='bold',
                    xytext=(5, 3), textcoords='offset points')
    ax_ind.annotate(f'{last_d:.1f}', xy=(dates_num[-1], last_d),
                    fontsize=7, color=COLOR_D, fontweight='bold',
                    xytext=(5, -10), textcoords='offset points')

# ─────────────────── Chart (1×4 세로) ─────────────────── #
def make_chart(daily_full: pd.DataFrame, code: str):
    name = get_name(code)

    # 일봉: 전체로 지표 계산 → 최근 80일만 표시
    df_daily = add_composites(daily_full.copy())
    df_daily_show = df_daily.tail(DAILY_BARS).reset_index(drop=True)
    sig_daily = detect_cross(df_daily)

    # 주봉: 리샘플 → 지표 계산 → 최근 40주만 표시
    df_weekly_full = resample_weekly(daily_full.copy())
    df_weekly = add_composites(df_weekly_full.copy())
    df_weekly_show = df_weekly.tail(WEEKLY_BARS).reset_index(drop=True)
    sig_weekly = detect_cross(df_weekly)

    # 1열 × 4행
    fig, (ax_dc, ax_di, ax_wc, ax_wi) = plt.subplots(
        nrows=4, ncols=1, figsize=(10, 14),
        gridspec_kw={'height_ratios': [3, 1, 3, 1], 'hspace': 0.35}
    )

    d_sig = f"  [{sig_daily}]" if sig_daily else ""
    w_sig = f"  [{sig_weekly}]" if sig_weekly else ""

    _plot_panel(ax_dc, ax_di, df_daily_show,
                title=f"일봉 {DAILY_BARS}일 — {code} ({name}){d_sig}",
                date_fmt='%m/%d')

    _plot_panel(ax_wc, ax_wi, df_weekly_show,
                title=f"주봉 {WEEKLY_BARS}주 — {code} ({name}){w_sig}",
                date_fmt='%y/%m')

    fig.suptitle(
        f"MACD+Stochastic  단기{FAST_PERIOD}/장기{SLOW_PERIOD}/"
        f"K1={K_WINDOW}/K2={K_SMOOTH}/D={D_SMOOTH}  "
        f"기준선 {OS_LINE}/{OB_LINE}",
        fontproperties=font_prop, fontsize=9, y=1.0, color='gray'
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    path = f"{code}_chart.png"
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    logging.info("차트 저장: %s", path)
    return path, sig_daily, sig_weekly

# ─────────────────── 텔레그램 ─────────────────── #
def tg_text(msg: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for chunk in [msg[i:i + 3500] for i in range(0, len(msg), 3500)]:
        try:
            requests.post(url, json={'chat_id': CHAT_ID, 'text': chunk}, timeout=15)
        except Exception as e:
            logging.warning("텍스트 전송 실패: %s", e)
        time.sleep(0.3)

def tg_photo(path: str, caption: str = ''):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    try:
        with open(path, 'rb') as f:
            requests.post(url, data={'chat_id': CHAT_ID, 'caption': caption},
                          files={'photo': f}, timeout=30)
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

        chart_path, sig_daily, sig_weekly = make_chart(df, code)

        caption = (
            f"{corp_name} ({code})\n"
            f"📄 {report_nm}\n"
            f"📅 {rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}\n"
            f"🔗 https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}\n"
            f"일봉: {sig_daily if sig_daily else '없음'} | "
            f"주봉: {sig_weekly if sig_weekly else '없음'}"
        )

        if sig_daily or sig_weekly:
            sig_label = f"[일봉:{sig_daily or '-'}/주봉:{sig_weekly or '-'}]"
            caption = f"{sig_label}\n{caption}"
            alerts.append(f"• {corp_name} ({code}) — 일봉: {sig_daily or '-'} / 주봉: {sig_weekly or '-'}")

        tg_photo(chart_path, caption=caption)
        if SAVE_CSV:
            df.to_csv(f"{code}_hist.csv", index=False)

    if alerts:
        summary = f"📈 신호 종목 ({len(alerts)}개)\n\n" + "\n".join(alerts)
        tg_text(summary)
    else:
        tg_text("📭 신호 없음 (골든/데드 크로스 미발생)")

    logging.info("==== 종료 ====")


if __name__ == '__main__':
    main()
