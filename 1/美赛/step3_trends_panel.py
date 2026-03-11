import re
import time
import random
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "2026_MCM_Problem_C_Data.csv"
WINDOWS_PATH = BASE_DIR / "season_week_windows_completed.csv"

OUT_PATH = BASE_DIR / "dwts_trends_weekly_panel.csv"
CACHE_PATH = BASE_DIR / "_trends_cache.csv"  # 断点续跑缓存

GEO = "US"
HL = "en-US"
TZ = 360  # US Central-ish; 实际不敏感
BATCH_K = 5  # pytrends 建议每次<=5个关键词
SLEEP_BETWEEN_CALLS = (2.0, 4.5)  # 防止被限流
RETRIES = 4


def normalize_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def load_unique_celebrities(df: pd.DataFrame) -> pd.DataFrame:
    # 兼容列名：你官方 CSV 里一般是 celebrity_name（如果不同你告诉我）
    for col in ["celebrity_name", "celebrity", "Celebrity"]:
        if col in df.columns:
            celeb_col = col
            break
    else:
        raise ValueError("找不到选手列名：请确认是否存在 celebrity_name / celebrity")

    out = df[[celeb_col, "season"]].drop_duplicates().copy()
    out.rename(columns={celeb_col: "celebrity"}, inplace=True)
    out["celebrity"] = out["celebrity"].map(normalize_name)
    out["season"] = out["season"].astype(int)
    return out


def build_queries(celeb_list):
    # 加 DWTS 降低同名污染
    return [f"{c} Dancing with the Stars" for c in celeb_list]


def safe_sleep():
    time.sleep(random.uniform(*SLEEP_BETWEEN_CALLS))


def read_cache():
    if CACHE_PATH.exists():
        c = pd.read_csv(CACHE_PATH)
        # key: season, week, celebrity
        c["season"] = c["season"].astype(int)
        c["week"] = c["week"].astype(int)
        c["celebrity"] = c["celebrity"].map(normalize_name)
        return c
    return pd.DataFrame(columns=["season", "week", "celebrity", "trend", "timeframe", "kw"])


def append_cache(rows):
    if not rows:
        return
    add = pd.DataFrame(rows)
    header = not CACHE_PATH.exists()
    add.to_csv(CACHE_PATH, index=False, mode="a", header=header)


def main():
    raw = pd.read_csv(DATA_PATH, na_values=["N/A"])
    raw.columns = raw.columns.str.strip()
    celebs = load_unique_celebrities(raw)

    windows = pd.read_csv(WINDOWS_PATH)
    windows.columns = windows.columns.str.strip()
    windows["season"] = windows["season"].astype(int)
    windows["week"] = windows["week"].astype(int)

    # 做一个 season-week -> (start,end) 字典
    win_map = {
        (int(r.season), int(r.week)): (str(r.window_start), str(r.window_end))
        for r in windows.itertuples(index=False)
    }

    cache = read_cache()
    done_keys = set(zip(cache["season"], cache["week"], cache["celebrity"]))

    pytrends = TrendReq(hl=HL, tz=TZ)

    results_rows = []

    # 我们按 season-week 来跑：窗口固定，便于批量抓取
    all_sw = sorted(win_map.keys())

    for (season, week) in tqdm(all_sw, desc="Season-Week"):
        start, end = win_map[(season, week)]
        timeframe = f"{start} {end}"

        # 该 season 下有哪些选手（只抓仍在该 season 的选手）
        season_celebs = celebs.loc[celebs["season"] == season, "celebrity"].unique().tolist()
        if not season_celebs:
            continue

        # 去掉已经完成的 (season, week, celeb)
        todo = [c for c in season_celebs if (season, week, c) not in done_keys]
        if not todo:
            continue

        # 分批<=5
        for i in range(0, len(todo), BATCH_K):
            batch = todo[i:i + BATCH_K]
            kw_list = build_queries(batch)

            ok = False
            for attempt in range(RETRIES):
                try:
                    pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo=GEO)
                    df = pytrends.interest_over_time()

                    if df is None or df.empty:
                        # 没数据就填 0（也写入缓存，避免死循环）
                        rows = []
                        for celeb, kw in zip(batch, kw_list):
                            rows.append({
                                "season": season,
                                "week": week,
                                "celebrity": celeb,
                                "trend": 0.0,
                                "timeframe": timeframe,
                                "kw": kw
                            })
                            done_keys.add((season, week, celeb))
                        append_cache(rows)
                        ok = True
                        break

                    # df 的列名就是 kw_list；按时间序列取均值作为周度热度
                    rows = []
                    for celeb, kw in zip(batch, kw_list):
                        if kw in df.columns:
                            val = float(df[kw].mean())
                        else:
                            val = 0.0
                        rows.append({
                            "season": season,
                            "week": week,
                            "celebrity": celeb,
                            "trend": val,
                            "timeframe": timeframe,
                            "kw": kw
                        })
                        done_keys.add((season, week, celeb))
                    append_cache(rows)
                    ok = True
                    break

                except Exception as e:
                    # 失败退避
                    sleep_s = (2 ** attempt) + random.uniform(0, 1.0)
                    print(f"[WARN] S{season} W{week} batch {i//BATCH_K} attempt {attempt+1}/{RETRIES} failed: {type(e).__name__}: {e}")
                    time.sleep(sleep_s)

            safe_sleep()

            if not ok:
                # 这一批最终失败：写 NaN，避免卡死
                rows = []
                for celeb, kw in zip(batch, kw_list):
                    rows.append({
                        "season": season,
                        "week": week,
                        "celebrity": celeb,
                        "trend": float("nan"),
                        "timeframe": timeframe,
                        "kw": kw
                    })
                    done_keys.add((season, week, celeb))
                append_cache(rows)

    # 汇总输出
    final = read_cache()
    final = final[["season", "week", "celebrity", "trend"]].copy()
    final.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print(final.head())


if __name__ == "__main__":
    main()
