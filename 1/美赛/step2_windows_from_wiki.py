import pandas as pd
import numpy as np
import re
import requests
from io import StringIO
from dateutil import parser
from datetime import timedelta

# ---------------------------------------------------------
# 配置区域
# ---------------------------------------------------------
MAX_WEEK_PATH = "season_max_week.csv"
OUTPUT_PATH = "season_week_windows.csv"

# 伪装头：这一步是解决 403 Forbidden 的关键
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_wiki_tables(url):
    """
    使用 requests 带上伪装头去下载网页，然后传给 pandas 解析
    """
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()  # 如果是 403/404 这里会报错
        # 用 lxml 解析器（一定要装 pip install lxml）
        return pd.read_html(StringIO(r.text), flavor='lxml')
    except Exception as e:
        print(f"  Request Failed: {e}")
        return []

def season_url(season: int) -> str:
    return f"https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_season_{season})"

def clean_date(x):
    if pd.isna(x):
        return None
    s = str(x)
    # 去掉引用角标 [1]
    s = re.sub(r"\[[^\]]+\]", "", s).strip()
    try:
        # 简单过滤过短字符串
        if len(s) < 5: return None
        dt = parser.parse(s, fuzzy=True).date()
        return dt
    except Exception:
        return None

def extract_air_dates_from_tables(tables):
    best = None
    best_score = -1
    
    # 智能寻找含有 Air date 的表格
    for t in tables:
        # 处理多层表头
        if isinstance(t.columns, pd.MultiIndex):
            cols = [str(c).lower() for c in t.columns.get_level_values(-1)]
        else:
            cols = [str(c).lower() for c in t.columns]
            
        score = 0
        for c in cols:
            if "air" in c and "date" in c: score += 3
            elif "original" in c and "date" in c: score += 2
            elif "date" in c: score += 1
        
        # 行数太少的通常不是赛程表
        if len(t) > 5: score += 1
            
        if score > best_score:
            best = t
            best_score = score
            
    if best is None: return []

    # 定位具体列
    date_col = None
    if isinstance(best.columns, pd.MultiIndex):
        flat_cols = best.columns.to_flat_index()
        for idx, col_tuple in enumerate(flat_cols):
            c_str = " ".join([str(x).lower() for x in col_tuple])
            if ("air" in c_str and "date" in c_str) or ("original" in c_str and "date" in c_str):
                date_col = best.columns[idx]
                break
    else:
        for c in best.columns:
            lc = str(c).lower()
            if ("air" in lc and "date" in lc) or ("original" in lc and "date" in lc):
                date_col = c
                break
                
    if date_col is None: return []

    # 提取并清洗日期
    dates = [clean_date(x) for x in best[date_col].tolist()]
    dates = [d for d in dates if d is not None]
    # 去重并排序
    dates = sorted(list(dict.fromkeys(dates)))
    return dates

def build_windows_from_airdates(air_dates):
    windows = []
    for i, d in enumerate(air_dates):
        start = d
        if i < len(air_dates) - 1:
            end = air_dates[i+1] - timedelta(days=1)
        else:
            end = d + timedelta(days=6)
        windows.append((i+1, start, end))
    return windows

def main():
    try:
        max_week_df = pd.read_csv(MAX_WEEK_PATH)
    except:
        print(f"Error: Could not find {MAX_WEEK_PATH}. Please run Step 1 first.")
        return

    rows = []

    for _, r in max_week_df.iterrows():
        season = int(r["season"])
        max_week = int(r["max_week_in_official_csv"])
        if max_week <= 0: continue

        url = season_url(season)
        print(f"[Season {season}] reading {url}")
        
        # 使用带伪装的函数
        tables = get_wiki_tables(url)
        
        if not tables:
            print(f"  WARNING: No tables found for Season {season}")
            continue

        air_dates = extract_air_dates_from_tables(tables)
        
        if len(air_dates) == 0:
            print(f"  WARNING: No valid air dates extracted for Season {season}")
            continue

        if len(air_dates) < max_week:
            print(f"  WARNING: air_dates={len(air_dates)} < max_week={max_week}. Need manual fix.")
        
        windows = build_windows_from_airdates(air_dates)
        
        # 只取官方数据需要的周数
        valid_len = min(len(windows), max_week)
        windows = windows[:valid_len]

        for w, start, end in windows:
            rows.append({
                "season": season,
                "week": w,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "source_url": url
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(out.head(10))

if __name__ == "__main__":
    main()