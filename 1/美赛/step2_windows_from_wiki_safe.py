import re
import sys
import time
import json
import random
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
MAXWEEK_IN = BASE_DIR / "season_max_week.csv"
CACHE_DIR = BASE_DIR / "_wiki_cache"
CACHE_DIR.mkdir(exist_ok=True)

OUT = BASE_DIR / "season_week_windows_clean.csv"


def wiki_urls(season: int):
    # 两种常见 URL，优先你现在 CSV 里那种
    return [
        f"https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_season_{season})",
        f"https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_TV_series)_season_{season}",
    ]


MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

DATE_RE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})")


def parse_first_date(text: str):
    m = DATE_RE.search(text)
    if not m:
        return None
    month = MONTHS[m.group(1)]
    day = int(m.group(2))
    year = int(m.group(3))
    return datetime(year, month, day)


def fetch_html(url: str, timeout=15, retries=3, backoff=(2.0, 6.0)):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MCM/ICM research script"
    }
    last_err = None
    for k in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r.text
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_err = e
        # 指数退避 + 抖动
        sleep_s = min(backoff[1], backoff[0] * (2 ** k)) + random.uniform(0, 1.0)
        time.sleep(sleep_s)
    raise last_err


def extract_premiere_from_infobox(html: str):
    soup = BeautifulSoup(html, "html.parser")
    infobox = soup.select_one("table.infobox")
    if not infobox:
        return None

    # 1) 强制优先抓 "Original release" 这一行
    for row in infobox.select("tr"):
        th = row.find("th")
        td = row.find("td")
        if not th or not td:
            continue
        key = th.get_text(" ", strip=True).lower()
        if "original release" in key:
            val = td.get_text(" ", strip=True)
            # 取该行出现的第一个日期（区间左端）
            m = DATE_RE.search(val)
            if m:
                month = MONTHS[m.group(1)]
                day = int(m.group(2))
                year = int(m.group(3))
                return datetime(year, month, day)

    # 2) 次优：抓含 aired/released/premiere 的行
    for row in infobox.select("tr"):
        th = row.find("th")
        td = row.find("td")
        if not th or not td:
            continue
        key = th.get_text(" ", strip=True).lower()
        if any(k in key for k in ["aired", "released", "premiere", "first aired"]):
            val = td.get_text(" ", strip=True)
            m = DATE_RE.search(val)
            if m:
                month = MONTHS[m.group(1)]
                day = int(m.group(2))
                year = int(m.group(3))
                return datetime(year, month, day)

    # 3) 兜底：infobox全文最早日期（不是第一个日期！）
    txt = infobox.get_text(" ", strip=True)
    all_matches = list(DATE_RE.finditer(txt))
    if not all_matches:
        return None

    dates = []
    for m in all_matches:
        month = MONTHS[m.group(1)]
        day = int(m.group(2))
        year = int(m.group(3))
        dates.append(datetime(year, month, day))

    return min(dates)



def get_premiere_date(season: int):
    cache_path = CACHE_DIR / f"season_{season}.json"
    if cache_path.exists():
        obj = json.loads(cache_path.read_text(encoding="utf-8"))
        if obj.get("premiere"):
            return datetime.fromisoformat(obj["premiere"]), obj.get("url")

    # 未缓存：尝试两个 URL
    for url in wiki_urls(season):
        try:
            html = fetch_html(url)
            premiere = extract_premiere_from_infobox(html)
            if premiere:
                cache_path.write_text(json.dumps({"premiere": premiere.date().isoformat(), "url": url}, ensure_ascii=False), encoding="utf-8")
                return premiere, url
            else:
                # 缓存失败也存一下，避免反复打 URL
                cache_path.write_text(json.dumps({"premiere": None, "url": url, "note": "no date parsed"}, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            # 写失败缓存，避免一直卡同一季
            cache_path.write_text(json.dumps({"premiere": None, "url": url, "note": f"fetch failed: {type(e).__name__}: {e}"}, ensure_ascii=False), encoding="utf-8")
            continue

        # 每季请求间隔，避免 429/403
        time.sleep(random.uniform(1.2, 2.2))

    return None, None


def build_windows(premiere: datetime, max_week: int):
    rows = []
    for w in range(1, max_week + 1):
        start = premiere + timedelta(days=7*(w-1))
        end = start + timedelta(days=6)  # 固定 7 天窗口
        rows.append((w, start.date().isoformat(), end.date().isoformat()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=None, help="Only run these seasons, e.g. --seasons 11 12 19")
    args = ap.parse_args()

    mw = pd.read_csv(MAXWEEK_IN)
    if args.seasons:
        mw = mw[mw["season"].isin(args.seasons)].copy()

    out_rows = []
    missing = []

    for _, r in mw.iterrows():
        season = int(r["season"])
        max_week = int(r["max_week_in_official_csv"])
        print(f"[Season {season}] max_week={max_week}")

        premiere, url = get_premiere_date(season)
        if not premiere:
            print(f"  WARN: premiere not found (see cache: {CACHE_DIR / f'season_{season}.json'})")
            missing.append(season)
            continue

        print(f"  premiere={premiere.date().isoformat()} from {url}")
        for w, ws, we in build_windows(premiere, max_week):
            out_rows.append({
                "season": season,
                "week": w,
                "window_start": ws,
                "window_end": we,
                "source_url": url
            })

    out = pd.DataFrame(out_rows).sort_values(["season", "week"])
    out.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")
    print("Missing seasons:", missing)


if __name__ == "__main__":
    main()
#cd '1\美赛' & 'C:\Users\LENOVO\anaconda3\envs\mcmtrends\python.exe' step2_windows_from_wiki.py