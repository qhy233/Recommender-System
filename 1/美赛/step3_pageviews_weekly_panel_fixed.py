# step3_pageviews_weekly_panel_fixed.py
# Build weekly popularity panel using Wikimedia Pageviews API (Wikipedia pageviews).
# Output: season-week-celebrity weekly pageviews + derived within-week features.

import argparse
import re
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

WIKI_API = "https://en.wikipedia.org/w/api.php"
PV_API_TMPL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "{project}/{access}/{agent}/{article}/daily/{start}/{end}"
)

DEFAULT_WINDOWS = "season_week_windows_weekly7.csv"
DEFAULT_OFFICIAL = "2026_MCM_Problem_C_Data.csv"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=DEFAULT_WINDOWS, help="CSV with season,week,window_start,window_end")
    ap.add_argument("--official", default=DEFAULT_OFFICIAL, help="Official MCM csv with celebrity list by season")
    ap.add_argument("--seasons", nargs="*", type=int, default=None, help="Seasons to run, e.g. --seasons 27 28")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between HTTP calls (seconds)")
    ap.add_argument("--outdir", default="OUT_PAGEVIEWS_FIXED", help="Output directory")
    ap.add_argument("--project", default="en.wikipedia", help="Wikimedia project")
    ap.add_argument("--access", default="all-access", help="all-access / desktop / mobile-app / mobile-web")
    ap.add_argument("--agent", default="user", help="user / spider / automated")
    ap.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds")
    return ap.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    # robust encoding: try utf-8-sig then utf-8 then default
    for enc in ("utf-8-sig", "utf-8", None):
        try:
            return pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
        except Exception:
            pass
    # final fallback
    return pd.read_csv(path, engine="python")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def load_windows(windows_path: Path) -> pd.DataFrame:
    w = safe_read_csv(windows_path)
    w = normalize_columns(w)
    required = {"season", "week", "window_start", "window_end"}
    missing = required - set(w.columns)
    if missing:
        raise ValueError(f"windows missing columns {missing}. got={list(w.columns)}")
    w["season"] = w["season"].astype(int)
    w["week"] = w["week"].astype(int)
    w["window_start"] = pd.to_datetime(w["window_start"])
    w["window_end"] = pd.to_datetime(w["window_end"])
    return w.sort_values(["season", "week"]).reset_index(drop=True)


def load_celebrities(official_path: Path) -> pd.DataFrame:
    df = safe_read_csv(official_path)
    df = normalize_columns(df)
    # try common column names
    name_col = None
    for cand in ["celebrity", "celebrity_name", "Celebrity", "name", "Name"]:
        if cand in df.columns:
            name_col = cand
            break
    if name_col is None:
        # last resort: find a column containing 'celebrity'
        for c in df.columns:
            if "celebr" in c.lower():
                name_col = c
                break
    if name_col is None:
        raise ValueError(f"Cannot find celebrity column in official csv. columns={list(df.columns)}")

    if "season" not in df.columns:
        raise ValueError("Official csv must contain 'season' column.")

    out = df[["season", name_col]].rename(columns={name_col: "celebrity"}).dropna()
    out["season"] = out["season"].astype(int)
    out["celebrity"] = out["celebrity"].astype(str).str.strip()
    out = out.drop_duplicates()
    return out.sort_values(["season", "celebrity"]).reset_index(drop=True)


def wiki_search_titles(query: str, timeout: int = 15) -> list[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "srlimit": 10,
    }
    r = requests.get(WIKI_API, params=params, timeout=timeout, headers={"User-Agent": "mcm-pageviews-bot/0.1"})
    r.raise_for_status()
    js = r.json()
    items = js.get("query", {}).get("search", [])
    return [it.get("title", "") for it in items if it.get("title")]


def choose_best_title(celebrity: str, titles: list[str]) -> str | None:
    """
    Avoid mapping to DWTS season pages or other irrelevant pages.
    Preference: person page, matching tokens in name.
    """
    if not titles:
        return None

    name_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", celebrity)]
    bad_patterns = [
        r"\bDancing with the Stars\b",
        r"\bseason\b",
        r"\b\(American TV series\)\s+season\b",
        r"\bList of\b",
        r"\bEpisode\b",
    ]

    def score(title: str) -> float:
        tl = title.lower()
        # hard-penalize bad pages
        for bp in bad_patterns:
            if re.search(bp.lower(), tl):
                return -999.0
        # token overlap
        overlap = sum(1 for t in name_tokens if t and t in tl)
        # small bonus if looks like person page (has space, not list)
        bonus = 0.2 if " " in title and "list" not in tl else 0.0
        # penalize disambiguation
        pen = 0.5 if "disambiguation" in tl else 0.0
        return overlap + bonus - pen

    scored = sorted(((score(t), t) for t in titles), reverse=True)
    best_score, best_title = scored[0]
    if best_score < 0:
        return None
    return best_title


def map_celebs_to_titles(season_celebs: pd.DataFrame, outdir: Path, timeout: int, sleep: float) -> pd.DataFrame:
    """
    Build (season, celebrity) -> wikipedia title mapping.
    Caches to outdir/wiki_title_map.csv and reuses if exists.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    map_path = outdir / "wiki_title_map.csv"

    if map_path.exists():
        m = safe_read_csv(map_path)
        m = normalize_columns(m)
        # ensure columns exist
        if {"season", "celebrity", "wiki_title"}.issubset(set(m.columns)):
            m["season"] = m["season"].astype(int)
            m["celebrity"] = m["celebrity"].astype(str)
            m["wiki_title"] = m["wiki_title"].astype(str)
            # keep only needed pairs (avoid old seasons interference)
            merged = season_celebs.merge(m, on=["season", "celebrity"], how="left")
            if merged["wiki_title"].notna().all():
                return merged
            # else continue to fill missing
        # else: rebuild

    rows = []
    for _, row in tqdm(season_celebs.iterrows(), total=len(season_celebs), desc="Mapping to Wikipedia titles"):
        season = int(row["season"])
        celeb = str(row["celebrity"])
        titles = wiki_search_titles(celeb, timeout=timeout)
        best = choose_best_title(celeb, titles)
        rows.append({"season": season, "celebrity": celeb, "wiki_title": best if best else ""})
        time.sleep(sleep)

    m = pd.DataFrame(rows)
    # Save mapping
    m.to_csv(map_path, index=False, encoding="utf-8-sig")
    return season_celebs.merge(m, on=["season", "celebrity"], how="left")


def ymd00(dt: pd.Timestamp) -> str:
    # Wikimedia pageviews expects yyyymmdd00
    return dt.strftime("%Y%m%d") + "00"


def fetch_pageviews_daily(title: str, start: pd.Timestamp, end: pd.Timestamp,
                          project: str, access: str, agent: str,
                          timeout: int) -> pd.DataFrame:
    """
    Returns DataFrame with columns: date, views
    """
    article = quote(title.replace(" ", "_"), safe="")
    url = PV_API_TMPL.format(
        project=project,
        access=access,
        agent=agent,
        article=article,
        start=ymd00(start),
        end=ymd00(end),
    )
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "mcm-pageviews-bot/0.1"})
    if r.status_code == 404:
        # no pageviews data or wrong title
        raise FileNotFoundError(f"404 for title={title}")
    r.raise_for_status()
    js = r.json()
    items = js.get("items", [])
    if not items:
        return pd.DataFrame(columns=["date", "views"])

    out = pd.DataFrame({
        "date": [pd.to_datetime(it["timestamp"][:8]) for it in items],
        "views": [int(it.get("views", 0)) for it in items],
    })
    return out


def build_daily_cache(tasks: pd.DataFrame, outdir: Path, args) -> pd.DataFrame:
    """
    tasks columns: season, celebrity, wiki_title, start, end
    cache file: outdir/dwts_pageviews_daily_cache.csv
    """
    cache_path = outdir / "dwts_pageviews_daily_cache.csv"
    if cache_path.exists():
        cache = safe_read_csv(cache_path)
        cache = normalize_columns(cache)
        # expected: season,celebrity,date,views,wiki_title
        if {"season", "celebrity", "date", "views", "wiki_title"}.issubset(set(cache.columns)):
            cache["season"] = cache["season"].astype(int)
            cache["celebrity"] = cache["celebrity"].astype(str)
            cache["date"] = pd.to_datetime(cache["date"])
            cache["views"] = pd.to_numeric(cache["views"], errors="coerce")
        else:
            cache = pd.DataFrame(columns=["season", "celebrity", "wiki_title", "date", "views"])
    else:
        cache = pd.DataFrame(columns=["season", "celebrity", "wiki_title", "date", "views"])

    # Determine which (season, celebrity) already cached fully
    cached_pairs = set()
    if len(cache) > 0:
        cached_pairs = set(zip(cache["season"].astype(int), cache["celebrity"].astype(str)))

    new_rows = []
    for _, t in tqdm(tasks.iterrows(), total=len(tasks), desc="Fetching daily pageviews"):
        season = int(t["season"])
        celeb = str(t["celebrity"])
        title = str(t["wiki_title"])
        start = pd.to_datetime(t["start"])
        end = pd.to_datetime(t["end"])

        # If already have any data for this pair, we still might need extend range,
        # but in this workflow we cache per season range once; keep simple:
        if (season, celeb) in cached_pairs:
            continue

        if not title or title.strip() == "":
            # no title mapped -> skip; keep missing to fill later
            continue

        try:
            daily = fetch_pageviews_daily(
                title=title, start=start, end=end,
                project=args.project, access=args.access, agent=args.agent,
                timeout=args.timeout
            )
            if len(daily) == 0:
                # keep empty record as "seen"
                pass
            else:
                daily["season"] = season
                daily["celebrity"] = celeb
                daily["wiki_title"] = title
                new_rows.append(daily[["season", "celebrity", "wiki_title", "date", "views"]])
        except FileNotFoundError:
            # 404: likely wrong title; leave missing and allow manual fix via wiki_title_map
            print(f"[WARN] 404 pageviews: s={season} celeb={celeb} title={title}")
        except Exception as e:
            print(f"[WARN] failed s={season} celeb={celeb} title={title} err={e}")

        time.sleep(args.sleep)

    if new_rows:
        add = pd.concat(new_rows, ignore_index=True)
        cache = pd.concat([cache, add], ignore_index=True)

    # de-duplicate
    if len(cache) > 0:
        cache = cache.drop_duplicates(subset=["season", "celebrity", "date"], keep="last").reset_index(drop=True)

    cache.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return cache


def expand_windows_to_daily(windows: pd.DataFrame) -> pd.DataFrame:
    # build mapping (season, week, date)
    rows = []
    for _, r in windows.iterrows():
        s = int(r["season"])
        w = int(r["week"])
        start = pd.to_datetime(r["window_start"])
        end = pd.to_datetime(r["window_end"])
        # inclusive range
        days = pd.date_range(start=start, end=end, freq="D")
        rows.append(pd.DataFrame({"season": s, "week": w, "date": days}))
    return pd.concat(rows, ignore_index=True)


def within_week_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    g: one group of (season, week)
    Adds within-week relative features.
    Requires: pageviews_raw_final (can contain NaN)
    """
    x = g["pageviews_raw_final"].astype(float)

    # total share (ignore NaN)
    total = np.nansum(x.values)
    g["pageviews_share"] = np.where(np.isfinite(x), x / total if total > 0 else np.nan, np.nan)

    # log1p
    g["pageviews_log1p"] = np.where(np.isfinite(x), np.log1p(x), np.nan)

    # z-score within week
    mu = np.nanmean(x.values)
    sd = np.nanstd(x.values, ddof=0)
    if sd > 0:
        g["pageviews_z_week"] = (x - mu) / sd
    else:
        g["pageviews_z_week"] = np.nan

    # rank within week (1 = highest)
    # use dense rank; NaN stays NaN
    g["pageviews_rank_week"] = x.rank(ascending=False, method="dense")

    return g


def fill_missing_by_season_celebrity(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing weekly pageviews by:
    1) linear interpolation over weeks for each (season,celebrity)
    2) fallback: within-season median for that week
    """
    weekly = weekly.sort_values(["season", "celebrity", "week"]).reset_index(drop=True)

    # 1) interp per (season,celebrity)
    def interp_one(g):
        g = g.sort_values("week").copy()
        g["pageviews_raw_interp"] = g["pageviews_raw"].astype(float).interpolate(limit_direction="both")
        return g

    weekly = weekly.groupby(["season", "celebrity"], group_keys=False).apply(interp_one).reset_index(drop=True)

    # 2) fallback: within-season-week median across celebs
    med = (
        weekly.groupby(["season", "week"])["pageviews_raw_interp"]
        .median()
        .rename("week_median")
        .reset_index()
    )
    weekly = weekly.merge(med, on=["season", "week"], how="left")

    # final
    weekly["pageviews_raw_final"] = weekly["pageviews_raw_interp"]
    miss = weekly["pageviews_raw_final"].isna()
    weekly.loc[miss, "pageviews_raw_final"] = weekly.loc[miss, "week_median"]

    # flag fill source
    weekly["fill_source"] = np.where(
        weekly["pageviews_raw"].notna(),
        "observed",
        np.where(weekly["pageviews_raw_interp"].notna(), "interp", np.where(weekly["week_median"].notna(), "week_median", "missing"))
    )

    return weekly


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent
    windows_path = (base / args.windows).resolve()
    official_path = (base / args.official).resolve()
    outdir = (base / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    windows = load_windows(windows_path)
    celebs = load_celebrities(official_path)

    if args.seasons:
        seasons = sorted(set(args.seasons))
        windows = windows[windows["season"].isin(seasons)].reset_index(drop=True)
        celebs = celebs[celebs["season"].isin(seasons)].reset_index(drop=True)

    print(f"[INFO] seasons={sorted(windows['season'].unique().tolist())}")
    print(f"[INFO] windows rows={len(windows)} unique seasons={windows['season'].nunique()}")
    print(f"[INFO] unique celebs in selected seasons={len(celebs)}")

    # Map celebs -> wiki titles
    mapped = map_celebs_to_titles(celebs, outdir, timeout=args.timeout, sleep=args.sleep)
    # keep mapped records
    mapped["wiki_title"] = mapped["wiki_title"].fillna("").astype(str)

    # Build (season,celebrity) season-wide ranges
    season_ranges = (
        windows.groupby("season")
        .agg(start=("window_start", "min"), end=("window_end", "max"))
        .reset_index()
    )
    tasks = mapped.merge(season_ranges, on="season", how="left")
    tasks = tasks[["season", "celebrity", "wiki_title", "start", "end"]].drop_duplicates()

    # Fetch daily cache
    daily_cache = build_daily_cache(tasks, outdir, args)

    if len(daily_cache) == 0:
        raise RuntimeError("Daily cache is empty. Likely all wiki_title mappings failed. Check OUT_PAGEVIEWS_FIXED/wiki_title_map.csv")

    # Expand windows to daily
    daily_windows = expand_windows_to_daily(windows)

    # Merge daily views into windows, aggregate to week
    merged = daily_windows.merge(
        daily_cache[["season", "celebrity", "date", "views"]],
        on=["season", "date"],
        how="left",
    )
    # Note: this merge duplicates across celebs? We need celeb dimension in daily_windows:
    # build full grid (season,week,date) x celebs of that season
    # Fix: create (season,week,date) then join celebs, then join views by season,celebrity,date
    grid = daily_windows.merge(celebs, on="season", how="left")  # adds celebrity for each season
    merged = grid.merge(
        daily_cache[["season", "celebrity", "date", "views"]],
        on=["season", "celebrity", "date"],
        how="left",
    )

    weekly = (
        merged.groupby(["season", "week", "celebrity"], as_index=False)["views"]
        .sum(min_count=1)
        .rename(columns={"views": "pageviews_raw"})
        .sort_values(["season", "week", "celebrity"])
        .reset_index(drop=True)
    )

    # Fill missing
    weekly = fill_missing_by_season_celebrity(weekly)

    # Within-week features
    # ensure columns are columns, not index
    weekly = weekly.reset_index(drop=True)
    weekly = weekly.groupby(["season", "week"], group_keys=False).apply(within_week_features).reset_index(drop=True)

    # Save outputs
    panel_path = outdir / "dwts_pageviews_weekly_panel.csv"
    feats_path = outdir / "dwts_pageviews_features.csv"

    weekly.to_csv(panel_path, index=False, encoding="utf-8-sig")

    # feature subset
    feats = weekly[[
        "season", "week", "celebrity",
        "pageviews_raw_final", "pageviews_log1p", "pageviews_share",
        "pageviews_z_week", "pageviews_rank_week", "fill_source"
    ]].copy()
    feats.to_csv(feats_path, index=False, encoding="utf-8-sig")

    # Print stats
    n_total = len(weekly)
    n_missing_raw = int(weekly["pageviews_raw"].isna().sum())
    print(f"[DONE] weekly panel saved: {panel_path}")
    print(f"[DONE] features saved: {feats_path}")
    print(f"[STATS] rows={n_total}, raw missing={n_missing_raw}, fill_source counts:")
    print(weekly["fill_source"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
