# step3_google_trends_weekly_batch5.py
# -*- coding: utf-8 -*-

import argparse
import random
import time
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from pytrends.request import TrendReq

# -----------------------------
# Anti-429: rotate User-Agent
# -----------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]

def parse_seasons(tokens):
    """
    支持：
      --seasons 27
      --seasons 1 2 3
      --seasons 1-33
      --seasons 1-10 12 15-20
    """
    seasons = []
    for t in tokens:
        t = str(t).strip()
        if not t:
            continue
        if re.match(r"^\d+\-\d+$", t):
            a, b = t.split("-")
            a, b = int(a), int(b)
            if a <= b:
                seasons.extend(list(range(a, b + 1)))
            else:
                seasons.extend(list(range(b, a + 1)))
        else:
            seasons.append(int(t))
    seasons = sorted(set(seasons))
    return seasons

def detect_column(df, candidates):
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols:
            return cols[key]
    return None

def safe_read_csv(path: Path):
    # 自动处理 UTF-8 BOM / 普通 UTF-8
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")

def new_pytrends(hl, tz, timeout):
    return TrendReq(
        hl=hl,
        tz=tz,
        timeout=(timeout, timeout),
        requests_args={"headers": {"User-Agent": random.choice(USER_AGENTS)}}
    )

def is_429_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("429" in msg) or ("too many requests" in msg) or ("response with code 429" in msg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", required=True, help="e.g. 27 OR 1-33 OR 1 2 3 10-12")
    ap.add_argument("--windows", default="season_week_windows_weekly7.csv", help="season-week windows csv")
    ap.add_argument("--official", default="2026_MCM_Problem_C_Data.csv", help="official DWTS csv")
    ap.add_argument("--out", default="dwts_trends_weekly_panel.csv", help="output panel file")
    ap.add_argument("--cache", default="_dwts_trends_cache.csv", help="cache file for resume")
    ap.add_argument("--hl", default="en-US")
    ap.add_argument("--tz", type=int, default=360)  # US Central-ish; you can keep 360
    ap.add_argument("--geo", default="", help="e.g. US (empty means worldwide)")
    ap.add_argument("--sleep_min", type=float, default=45.0)
    ap.add_argument("--sleep_max", type=float, default=120.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=5, help="max keywords per request (<=5 recommended)")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    windows_path = (base_dir / args.windows).resolve()
    official_path = (base_dir / args.official).resolve()
    out_path = (base_dir / args.out).resolve()
    cache_path = (base_dir / args.cache).resolve()

    seasons = parse_seasons(args.seasons)
    print(f"[INFO] seasons={seasons}")
    print(f"[INFO] windows={windows_path}")
    print(f"[INFO] official={official_path}")
    print(f"[INFO] out={out_path}")
    print(f"[INFO] cache={cache_path}")

    # -----------------------------
    # Load windows
    # -----------------------------
    wdf = safe_read_csv(windows_path)
    wdf.columns = [c.strip() for c in wdf.columns]

    c_season = detect_column(wdf, ["season"])
    c_week = detect_column(wdf, ["week"])
    c_start = detect_column(wdf, ["window_start", "start", "start_date"])
    c_end = detect_column(wdf, ["window_end", "end", "end_date"])

    if not all([c_season, c_week, c_start, c_end]):
        raise ValueError(f"windows表缺列：需要 season, week, window_start, window_end。当前列={list(wdf.columns)}")

    wdf[c_season] = pd.to_numeric(wdf[c_season], errors="coerce").astype("Int64")
    wdf[c_week] = pd.to_numeric(wdf[c_week], errors="coerce").astype("Int64")

    # 标准化日期字符串为 YYYY-MM-DD
    wdf[c_start] = pd.to_datetime(wdf[c_start], errors="coerce").dt.strftime("%Y-%m-%d")
    wdf[c_end] = pd.to_datetime(wdf[c_end], errors="coerce").dt.strftime("%Y-%m-%d")

    wdf = wdf.dropna(subset=[c_season, c_week, c_start, c_end]).copy()
    wdf = wdf[wdf[c_season].isin(seasons)].copy()
    wdf = wdf.sort_values([c_season, c_week]).reset_index(drop=True)

    print(f"[INFO] windows rows={len(wdf)} (filtered)")

    # -----------------------------
    # Load official csv -> celebs by season
    # -----------------------------
    odf = safe_read_csv(official_path)
    odf.columns = [c.strip() for c in odf.columns]
    oc_season = detect_column(odf, ["season"])
    oc_celeb = detect_column(odf, ["celebrity", "celebrity_name", "celeb", "celebrityname"])

    if not oc_season or not oc_celeb:
        raise ValueError(f"official表缺列：需要 season & celebrity_name/celebrity。当前列={list(odf.columns)}")

    odf[oc_season] = pd.to_numeric(odf[oc_season], errors="coerce").astype("Int64")
    odf = odf.dropna(subset=[oc_season, oc_celeb]).copy()
    odf[oc_celeb] = odf[oc_celeb].astype(str).str.strip()
    odf = odf[odf[oc_season].isin(seasons)].copy()

    celebs_by_season = {}
    for s, sdf in odf.groupby(oc_season):
        names = sorted(set([x for x in sdf[oc_celeb].tolist() if x and x.lower() != "nan"]))
        celebs_by_season[int(s)] = names

    total_celebs = sum(len(v) for v in celebs_by_season.values())
    print(f"[INFO] celebs seasons={len(celebs_by_season)} total unique-by-season={total_celebs}")

    # -----------------------------
    # Load cache if exists
    # -----------------------------
    done = set()
    cache_df = None
    if cache_path.exists():
        try:
            cache_df = safe_read_csv(cache_path)
            cache_df.columns = [c.strip() for c in cache_df.columns]
            if set(["season", "week", "celebrity"]).issubset(set(cache_df.columns)):
                for r in cache_df[["season", "week", "celebrity"]].itertuples(index=False):
                    done.add((int(r.season), int(r.week), str(r.celebrity)))
        except Exception as e:
            print(f"[WARN] cannot read cache: {e}")

    print(f"[INFO] already done triples={len(done)} (from cache)")

    # -----------------------------
    # Build tasks: (season, week, start, end, chunk_of_celebs)
    # -----------------------------
    tasks = []
    for row in wdf.itertuples(index=False):
        s = int(getattr(row, c_season))
        w = int(getattr(row, c_week))
        start = getattr(row, c_start)
        end = getattr(row, c_end)
        celebs = celebs_by_season.get(s, [])
        if not celebs:
            continue

        # chunk them
        for i in range(0, len(celebs), args.chunk):
            chunk = celebs[i:i + args.chunk]
            # 如果这个 chunk 的所有 triple 都在 done 里，就跳过
            if all((s, w, c) in done for c in chunk):
                continue
            tasks.append((s, w, start, end, chunk))

    print(f"[INFO] tasks (groups of <= {args.chunk} kws)={len(tasks)}")

    pytrends = new_pytrends(args.hl, args.tz, args.timeout)

    # Prepare output append mode
    out_cols = ["season", "week", "celebrity", "trend", "timeframe", "kw", "geo", "trend_status"]
    if not out_path.exists():
        pd.DataFrame(columns=out_cols).to_csv(out_path, index=False, encoding="utf-8-sig")

    # helper: append rows to cache + out
    def append_rows(rows):
        df = pd.DataFrame(rows, columns=out_cols)
        # append to out
        df.to_csv(out_path, mode="a", index=False, header=False, encoding="utf-8-sig")
        # append to cache
        df[["season", "week", "celebrity", "trend", "timeframe", "kw", "geo", "trend_status"]].to_csv(
            cache_path, mode="a", index=False, header=not cache_path.exists(), encoding="utf-8-sig"
        )

    # -----------------------------
    # Run
    # -----------------------------
    pbar = tqdm(tasks, desc="Season-Week-(<=5 celebs)")
    for (s, w, start, end, chunk) in pbar:
        timeframe = f"{start} {end}"
        kw_list = [f"{c} Dancing with the Stars" for c in chunk]

        # retry loop
        ok = False
        last_err = None
        cooldown = 0.0

        for attempt in range(1, args.retries + 1):
            try:
                # polite random sleep BEFORE request (reduces burst pattern)
                base_sleep = random.uniform(args.sleep_min, args.sleep_max)
                time.sleep(base_sleep)

                # (re)build payload
                pytrends.build_payload(kw_list, timeframe=timeframe, geo=args.geo)
                iot = pytrends.interest_over_time()

                rows = []
                if iot is None or iot.empty:
                    # no data -> NaN, not 0
                    for celeb, kw in zip(chunk, kw_list):
                        if (s, w, celeb) in done:
                            continue
                        rows.append([s, w, celeb, np.nan, timeframe, kw, args.geo, "empty"])
                        done.add((s, w, celeb))
                else:
                    # drop isPartial if exists
                    if "isPartial" in iot.columns:
                        iot = iot.drop(columns=["isPartial"])

                    # average across returned points
                    for celeb, kw in zip(chunk, kw_list):
                        if (s, w, celeb) in done:
                            continue
                        if kw in iot.columns:
                            val = pd.to_numeric(iot[kw], errors="coerce").mean()
                            rows.append([s, w, celeb, float(val) if pd.notna(val) else np.nan, timeframe, kw, args.geo, "ok"])
                        else:
                            rows.append([s, w, celeb, np.nan, timeframe, kw, args.geo, "no_col"])
                        done.add((s, w, celeb))

                if rows:
                    append_rows(rows)

                ok = True
                break

            except Exception as e:
                last_err = e
                if is_429_error(e):
                    # exponential backoff + jitter, also rebuild TrendReq (new UA)
                    cool = min(600, (30 * (2 ** (attempt - 1))) + random.uniform(0, 30))
                    pbar.set_postfix_str(f"s={s} w={w} 429 attempt={attempt}/{args.retries} cool={int(cool)}s")
                    print(f"[429] s={s} w={w} attempt={attempt}/{args.retries} cooling={cool:.1f}s err={e}")
                    time.sleep(cool)
                    pytrends = new_pytrends(args.hl, args.tz, args.timeout)
                else:
                    # non-429: short backoff
                    cool = min(120, 10 * attempt + random.uniform(0, 10))
                    pbar.set_postfix_str(f"s={s} w={w} err attempt={attempt}/{args.retries} cool={int(cool)}s")
                    print(f"[ERR] s={s} w={w} attempt={attempt}/{args.retries} cool={cool:.1f}s err={e}")
                    time.sleep(cool)
                    pytrends = new_pytrends(args.hl, args.tz, args.timeout)

        if not ok:
            # record failure as NaN to avoid infinite loops; still mark done for this chunk
            rows = []
            for celeb, kw in zip(chunk, kw_list):
                if (s, w, celeb) in done:
                    continue
                rows.append([s, w, celeb, np.nan, timeframe, kw, args.geo, f"fail:{type(last_err).__name__}"])
                done.add((s, w, celeb))
            if rows:
                append_rows(rows)

    print(f"[DONE] saved panel to: {out_path}")
    print(f"[DONE] cache kept at: {cache_path} (you can delete it after确认无误)")

if __name__ == "__main__":
    main()
