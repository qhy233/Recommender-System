import argparse
import random
import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm


# ---------------------------
# Helpers
# ---------------------------

def safe_sleep(base=2.0, jitter=2.0):
    """sleep base + uniform(0, jitter) seconds"""
    time.sleep(base + random.random() * jitter)


def build_kw(name: str) -> str:
    # 你也可以改成 f"{name} DWTS" 做对照实验
    return f"{name} Dancing with the Stars"


def trend_payload(pytrends: TrendReq, kw_list, timeframe: str, geo="US"):
    pytrends.build_payload(
        kw_list=kw_list,
        cat=0,
        timeframe=timeframe,
        geo=geo,
        gprop=""
    )
    return pytrends.interest_over_time()


def normalize_single_series(df_iot: pd.DataFrame, kw: str) -> float:
    """
    interest_over_time 返回的是相对热度 0-100
    若为空或 kw 不存在，返回 NaN
    """
    if df_iot is None or df_iot.empty:
        return float("nan")
    if kw not in df_iot.columns:
        return float("nan")
    # 用该时间窗内的均值作为 week-level trend
    s = df_iot[kw].astype(float)
    if s.empty:
        return float("nan")
    return float(s.mean())


def load_existing(out_path: Path) -> pd.DataFrame:
    if out_path.exists():
        return pd.read_csv(out_path, encoding="utf-8-sig")
    return pd.DataFrame(columns=["season", "week", "celebrity", "trend", "timeframe", "kw"])


def make_done_set(existing: pd.DataFrame):
    if existing.empty:
        return set()
    # 断点续跑 key
    return set(
        zip(
            existing["season"].astype(int),
            existing["week"].astype(int),
            existing["celebrity"].astype(str)
        )
    )


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="*", type=int, default=None,
                        help="只跑指定赛季，例如: --seasons 27 28")
    parser.add_argument("--geo", type=str, default="US", help="地区，默认 US")
    parser.add_argument("--batch", type=int, default=1,
                        help="每次查询的关键词数量。强烈建议 1（最稳），2 会更快但更容易触发限制。")
    parser.add_argument("--sleep_base", type=float, default=2.0,
                        help="每次请求基础 sleep 秒数（建议 2-4）")
    parser.add_argument("--sleep_jitter", type=float, default=2.0,
                        help="每次请求额外随机 sleep 秒数（建议 2-5）")
    parser.add_argument("--retries", type=int, default=4, help="失败重试次数")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    WINDOWS_PATH = BASE_DIR / "season_week_windows_weekly7.csv"
    DATA_PATH = BASE_DIR / "2026_MCM_Problem_C_Data.csv"

    OUT_PATH = BASE_DIR / "dwts_trends_weekly_panel.csv"
    CACHE_PATH = BASE_DIR / "_dwts_trends_cache.csv"

    # 1) 读 windows（修复后的周窗口）
    windows = pd.read_csv(WINDOWS_PATH, encoding="utf-8-sig")
    windows.columns = windows.columns.str.strip()
    windows["season"] = windows["season"].astype(int)
    windows["week"] = windows["week"].astype(int)

    # 2) 读官方 CSV，抽取选手
    df = pd.read_csv(DATA_PATH, na_values=["N/A"])
    df.columns = df.columns.str.strip()
    if "celebrity_name" not in df.columns:
        raise ValueError("官方 CSV 里没有 celebrity_name 列，请检查列名。")
    df["season"] = df["season"].astype(int)

    # 3) 只跑指定 seasons（可选）
    if args.seasons and len(args.seasons) > 0:
        seasons_to_run = set(args.seasons)
        windows = windows[windows["season"].isin(seasons_to_run)].copy()
        df = df[df["season"].isin(seasons_to_run)].copy()
    else:
        seasons_to_run = set(sorted(windows["season"].unique().tolist()))

    # 4) 每季选手清单
    celebs = (
        df.drop_duplicates(subset=["season", "celebrity_name"])[["season", "celebrity_name"]]
          .rename(columns={"celebrity_name": "celebrity"})
          .sort_values(["season", "celebrity"])
    )

    # 5) 断点续跑：读取已有输出 + cache
    existing = load_existing(OUT_PATH)
    cache = load_existing(CACHE_PATH)
    combined = pd.concat([existing, cache], ignore_index=True)
    done = make_done_set(combined)

    print(f"[INFO] seasons_to_run={sorted(list(seasons_to_run))}")
    print(f"[INFO] windows rows={len(windows)} celebs rows={len(celebs)}")
    print(f"[INFO] already done={len(done)} (from OUT/CACHE)")

    # 6) pytrends session
    pytrends = TrendReq(
        hl="en-US",
        tz=360,         # US timezone offset minutes (360=UTC-6). 不用太纠结，主要是 trends API 需要这个字段
        retries=0,      # 我们自己控制 retries
        backoff_factor=0.0
    )

    # 7) 迭代 season-week
    # 预先把该 season 的选手列表做成 dict
    season_to_celebs = {
        int(s): g["celebrity"].tolist()
        for s, g in celebs.groupby("season")
    }

    tasks = []
    for _, r in windows.iterrows():
        s = int(r["season"])
        w = int(r["week"])
        start = str(r["window_start"])
        end = str(r["window_end"])
        timeframe = f"{start} {end}"

        for name in season_to_celebs.get(s, []):
            key = (s, w, str(name))
            if key in done:
                continue
            tasks.append((s, w, str(name), timeframe))

    print(f"[INFO] pending tasks={len(tasks)}")
    if len(tasks) == 0:
        print("[INFO] nothing to do.")
        return

    # 8) 开始抓取
    rows = []
    pbar = tqdm(tasks, desc="Season-Week-Celebrity", ncols=90)

    for (s, w, name, timeframe) in pbar:
        kw = build_kw(name)

        # retries
        val = float("nan")
        ok = False
        for attempt in range(1, args.retries + 1):
            try:
                # 限速
                safe_sleep(args.sleep_base, args.sleep_jitter)

                # 单关键词（最稳）
                df_iot = trend_payload(pytrends, [kw], timeframe=timeframe, geo=args.geo)
                val = normalize_single_series(df_iot, kw)
                ok = True
                break
            except Exception as e:
                # 常见：429, 403, connection reset 等
                wait = min(30, 2 ** attempt) + random.random() * 3
                pbar.set_postfix_str(f"err attempt={attempt}, sleep={wait:.1f}s")
                time.sleep(wait)

        if not ok:
            # 失败也记录，避免死循环；后续你可以再筛掉 NaN 重跑
            val = float("nan")

        rows.append({
            "season": s,
            "week": w,
            "celebrity": name,
            "trend": val,
            "timeframe": timeframe,
            "kw": kw
        })

        # 每攒一小批就落盘，防崩
        if len(rows) >= 200:
            tmp = pd.DataFrame(rows)
            tmp.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")
            rows = []

    # 9) 收尾：合并输出
    if rows:
        tmp = pd.DataFrame(rows)
        tmp.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")

    final_cache = load_existing(CACHE_PATH)
    final_out = pd.concat([existing, final_cache], ignore_index=True)
    final_out.drop_duplicates(subset=["season", "week", "celebrity"], keep="last", inplace=True)
    final_out.sort_values(["season", "week", "celebrity"], inplace=True)
    final_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved panel to: {OUT_PATH}")
    print(f"[DONE] cache kept at: {CACHE_PATH} (you can delete it after确认无误)")


if __name__ == "__main__":
    main()
