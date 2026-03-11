import argparse
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from pytrends.request import TrendReq


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Run all seasons found in windows file")
    ap.add_argument("--seasons", nargs="*", type=int, default=None, help="List of seasons to run, e.g. --seasons 27 28")
    ap.add_argument("--windows", type=str, default="season_week_windows_weekly7.csv")
    ap.add_argument("--official", type=str, default="2026_MCM_Problem_C_Data.csv")
    ap.add_argument("--out", type=str, default="dwts_trends_weekly_panel.csv")
    ap.add_argument("--cache", type=str, default="_dwts_trends_cache.csv")
    ap.add_argument("--hl", type=str, default="en-US")
    ap.add_argument("--tz", type=int, default=360)  # pytrends tz=360 is common for US; not critical
    ap.add_argument("--geo", type=str, default="US", help="Google Trends geo, e.g. US or '' for worldwide")
    ap.add_argument("--sleep_min", type=float, default=2.5)
    ap.add_argument("--sleep_max", type=float, default=6.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=20)
    return ap.parse_args()


def load_windows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    need = {"season", "week", "window_start", "window_end"}
    if not need.issubset(df.columns):
        raise ValueError(f"windows file missing columns: {need - set(df.columns)}")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype(int)
    df["window_start"] = pd.to_datetime(df["window_start"])
    df["window_end"] = pd.to_datetime(df["window_end"])

    # timeframe 格式：YYYY-MM-DD YYYY-MM-DD
    df["timeframe"] = df["window_start"].dt.strftime("%Y-%m-%d") + " " + df["window_end"].dt.strftime("%Y-%m-%d")
    return df


def load_celebs_from_official(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", na_values=["N/A"])
    df.columns = df.columns.str.strip()
    if "season" not in df.columns or "celebrity_name" not in df.columns:
        raise ValueError("Official CSV must contain columns: season, celebrity_name")

    celebs = (
        df[["season", "celebrity_name"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"celebrity_name": "celebrity"})
    )
    celebs["season"] = pd.to_numeric(celebs["season"], errors="coerce").astype(int)
    celebs["celebrity"] = celebs["celebrity"].astype(str).str.strip()
    return celebs


def read_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=["season", "week", "celebrity", "trend", "timeframe", "kw"])
    df = pd.read_csv(cache_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    # 统一字段
    for col in ["season", "week", "celebrity", "trend", "timeframe", "kw"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df[["season", "week", "celebrity", "trend", "timeframe", "kw"]].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["celebrity"] = df["celebrity"].astype(str).str.strip()
    return df


def write_cache(cache_path: Path, df_cache: pd.DataFrame):
    df_cache.to_csv(cache_path, index=False, encoding="utf-8-sig")


def fetch_trend(pytrends: TrendReq, kw: str, timeframe: str, geo: str, timeout: int) -> float:
    """
    返回该 timeframe 内的平均热度（0-100 标度），可能为 NaN
    """
    pytrends.build_payload([kw], cat=0, timeframe=timeframe, geo=geo)
    it = pytrends.interest_over_time()
    if it is None or it.empty or kw not in it.columns:
        return np.nan
    return float(it[kw].mean())


def main():
    args = parse_args()
    base = Path(".").resolve()
    windows_path = (base / args.windows).resolve()
    official_path = (base / args.official).resolve()
    out_path = (base / args.out).resolve()
    cache_path = (base / args.cache).resolve()

    print(f"[INFO] windows={windows_path}")
    print(f"[INFO] official={official_path}")
    print(f"[INFO] out={out_path}")
    print(f"[INFO] cache={cache_path}")

    windows = load_windows(windows_path)
    celebs = load_celebs_from_official(official_path)

    # 决定跑哪些赛季
    seasons_all = sorted(windows["season"].unique().tolist())
    if args.all:
        seasons_to_run = seasons_all
    elif args.seasons:
        seasons_to_run = [s for s in args.seasons if s in seasons_all]
    else:
        raise ValueError("Please provide --all or --seasons")

    print(f"[INFO] seasons_to_run={seasons_to_run}")

    # 取这些赛季的 windows + celebs
    windows_sub = windows[windows["season"].isin(seasons_to_run)].copy()
    celebs_sub = celebs[celebs["season"].isin(seasons_to_run)].copy()

    # 任务：season-week × season-celebrity
    tasks = (
        windows_sub[["season", "week", "timeframe"]]
        .merge(celebs_sub, on="season", how="inner")
        .rename(columns={"celebrity": "celebrity"})
    )
    tasks["kw"] = tasks["celebrity"] + " Dancing with the Stars"
    tasks = tasks[["season", "week", "celebrity", "timeframe", "kw"]].copy()

    # 读取 cache，做断点续跑
    cache = read_cache(cache_path)

    done_keys = set(
        zip(
            cache["season"].astype("Int64").fillna(-1).astype(int),
            cache["week"].astype("Int64").fillna(-1).astype(int),
            cache["celebrity"].astype(str),
        )
    )

    tasks_keys = list(zip(tasks["season"], tasks["week"], tasks["celebrity"]))
    pending_mask = [k not in done_keys for k in tasks_keys]
    pending = tasks.loc[pending_mask].copy()

    print(f"[INFO] windows rows={len(windows_sub)} celebs rows={len(celebs_sub)}")
    print(f"[INFO] already done={len(tasks) - len(pending)} (from CACHE)")
    print(f"[INFO] pending tasks={len(pending)}")

    # pytrends
    pytrends = TrendReq(hl=args.hl, tz=args.tz, timeout=(args.timeout, args.timeout))

    results = []
    pbar = tqdm(total=len(pending), desc="Season-Week-Celebrity", ncols=90)

    for _, row in pending.iterrows():
        season = int(row["season"])
        week = int(row["week"])
        celebrity = str(row["celebrity"])
        timeframe = str(row["timeframe"])
        kw = str(row["kw"])

        val = np.nan
        last_err = None
        for attempt in range(1, args.retries + 1):
            try:
                val = fetch_trend(pytrends, kw=kw, timeframe=timeframe, geo=args.geo, timeout=args.timeout)
                last_err = None
                break
            except Exception as e:
                last_err = str(e)
                # 失败就退避
                time.sleep(random.uniform(args.sleep_min, args.sleep_max) * attempt)

        # 记录结果（无论成功/失败，失败就是 NaN）
        results.append(
            {"season": season, "week": week, "celebrity": celebrity, "trend": val, "timeframe": timeframe, "kw": kw}
        )

        # 每次都写入 cache（防止中途断电/封 IP）
        cache = pd.concat([cache, pd.DataFrame([results[-1]])], ignore_index=True)
        write_cache(cache_path, cache)

        # 适度 sleep，减少被封
        time.sleep(random.uniform(args.sleep_min, args.sleep_max))

        pbar.update(1)

    pbar.close()

    # 输出 panel（就是你要的：每周×选手）
    cache_sorted = cache.copy()
    cache_sorted["season"] = pd.to_numeric(cache_sorted["season"], errors="coerce").astype("Int64")
    cache_sorted["week"] = pd.to_numeric(cache_sorted["week"], errors="coerce").astype("Int64")
    cache_sorted["celebrity"] = cache_sorted["celebrity"].astype(str).str.strip()
    cache_sorted = cache_sorted.drop_duplicates(subset=["season", "week", "celebrity"], keep="last")
    cache_sorted = cache_sorted.sort_values(["season", "week", "celebrity"], kind="mergesort")

    cache_sorted.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] saved panel to: {out_path}")
    print(f"[DONE] cache kept at: {cache_path} (delete after you verify)")

if __name__ == "__main__":
    main()
