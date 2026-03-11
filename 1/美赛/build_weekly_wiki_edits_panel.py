#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weekly Wikipedia EDITS panel (full-history, stable, reproducible).

Inputs (same directory):
  - 2026_MCM_Problem_C_Data.csv
  - season_week_windows_weekly7.csv

Outputs:
  - weekly_wiki_edits_panel.csv
      season, week, celebrity_name, wiki_title,
      wiki_edits_week, wiki_editors_week, wiki_bytes_net_week,
      window_start, window_end, status
  - wiki_title_map.csv  (audit & manual disambiguation)

Install:
  pip install pandas requests tqdm python-dateutil

Run:
  python -u build_weekly_wiki_edits_panel.py
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import pandas as pd
import requests
from dateutil.parser import parse as dtparse
from tqdm import tqdm

WIKI_API = "https://en.wikipedia.org/w/api.php"


def sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def iso_z(d: date, end_of_day: bool = False) -> str:
    # MediaWiki wants ISO8601 with Z (UTC). Day-level is fine.
    return d.strftime("%Y-%m-%dT23:59:59Z" if end_of_day else "%Y-%m-%dT00:00:00Z")


@dataclass
class FetchConfig:
    cache_dir: Path
    sleep_sec: float = 0.6
    timeout_sec: int = 25
    connect_timeout_sec: int = 10
    max_retries: int = 8
    backoff_base: float = 1.8


def cache_path(cfg: FetchConfig, key: str) -> Path:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg.cache_dir / f"{sha1(key)}.json"


def get_json_with_backoff(session: requests.Session, params: dict, cfg: FetchConfig) -> dict:
    last_err = None
    for attempt in range(cfg.max_retries):
        try:
            r = session.get(WIKI_API, params=params, timeout=(cfg.connect_timeout_sec, cfg.timeout_sec))
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(120.0, cfg.backoff_base ** attempt)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            wait = min(120.0, cfg.backoff_base ** attempt)
            time.sleep(wait)
    raise RuntimeError(f"Failed after retries: {last_err}")


# -----------------------------
# Title resolution (name -> wiki page title)
# -----------------------------
def resolve_wiki_title(session: requests.Session, name: str, cfg: FetchConfig) -> Optional[str]:
    # Use built-in search to find best page title
    params = {
        "action": "query",
        "list": "search",
        "srsearch":f'{name} "Dancing with the Stars"',
        "srlimit": 6,
        "format": "json",
    }
    data = get_json_with_backoff(session, params, cfg)
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None

    # Prefer first non-disambiguation
    for h in hits:
        title = h.get("title")
        if not title:
            continue
        # Check disambiguation via pageprops
        pp_params = {
            "action": "query",
            "titles": title,
            "prop": "pageprops",
            "format": "json",
        }
        pp = get_json_with_backoff(session, pp_params, cfg)
        pages = pp.get("query", {}).get("pages", {})
        is_disambig = False
        for _, pg in pages.items():
            if "pageprops" in pg and "disambiguation" in pg["pageprops"]:
                is_disambig = True
        if not is_disambig:
            return title

    # If all disambig, just return top result (manual fix later)
    return hits[0].get("title")


# -----------------------------
# Fetch revisions in [start, end] and compute daily series
# -----------------------------
def fetch_revisions(session: requests.Session, title: str, start: date, end: date, cfg: FetchConfig) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      date, edits, editors, bytes_net
    by scanning all revisions timestamps + size + user in the date range.
    """
    key = f"revs|{title}|{start.isoformat()}|{end.isoformat()}"
    cp = cache_path(cfg, key)
    if cp.exists():
        data_all = json.loads(cp.read_text(encoding="utf-8"))
    else:
        data_all = []

        cont = None
        while True:
            params = {
                "action": "query",
                "prop": "revisions",
                "titles": title,
                "rvprop": "timestamp|size|user",
                "rvlimit": "max",
                "rvdir": "newer",                 # chronological
                "rvstart": iso_z(start, False),
                "rvend": iso_z(end, True),
                "format": "json",
            }
            if cont:
                params.update(cont)

            data = get_json_with_backoff(session, params, cfg)
            data_all.append(data)

            cont = data.get("continue")
            time.sleep(cfg.sleep_sec)
            if not cont:
                break

        cp.write_text(json.dumps(data_all), encoding="utf-8")

    # Parse revision stream
        # Parse revision stream
    rows = []
    for data in data_all:
        pages = data.get("query", {}).get("pages", {})
        for _, pg in pages.items():
            revs = pg.get("revisions", [])
            for rv in revs:
                ts = rv.get("timestamp")  # e.g. 2005-06-12T03:44:00Z
                sz = rv.get("size")
                usr = rv.get("user")
                if not ts:
                    continue
                d = datetime.strptime(ts[:10], "%Y-%m-%d").date()
                rows.append((d, int(sz) if sz is not None else None, str(usr) if usr else None))

    if not rows:
        return pd.DataFrame(columns=["date", "edits", "editors", "bytes_net", "editors_set"])

    df = pd.DataFrame(rows, columns=["date", "size", "user"]).sort_values("date")

    # edits per day
    edits = df.groupby("date").size().rename("edits")

    # unique editors SET per day  (关键：保留集合，周内可去重)
    editors_set = df.groupby("date")["user"].apply(lambda s: set(x for x in s.dropna().tolist())).rename("editors_set")

    # net bytes per day: last_size - first_size (approx proxy)
    def net_bytes(g):
        g2 = g.dropna()
        if g2.empty:
            return 0
        return int(g2.iloc[-1] - g2.iloc[0])

    bytes_net = df.groupby("date")["size"].apply(net_bytes).rename("bytes_net")

    out = pd.concat([edits, editors_set, bytes_net], axis=1).reset_index()

    # 仍然保留一个“按天 unique editors 数”（可选，方便你检查）
    out["editors"] = out["editors_set"].apply(lambda x: len(x) if isinstance(x, set) else 0)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official_csv", default="2026_MCM_Problem_C_Data.csv")
    ap.add_argument("--windows_csv", default="season_week_windows_weekly7.csv")
    ap.add_argument("--out_csv", default="weekly_wiki_edits_panel.csv")
    ap.add_argument("--cache_dir", default=".wiki_revs_cache")
    ap.add_argument("--sleep", type=float, default=0.6)
    ap.add_argument("--max_seasons", type=int, default=0)
    args = ap.parse_args()

    official = pd.read_csv(args.official_csv)
    windows = pd.read_csv(args.windows_csv)

    for c in ["window_start", "window_end"]:
        windows[c] = windows[c].apply(lambda x: dtparse(str(x)).date())

    if not {"season", "celebrity_name"}.issubset(official.columns):
        raise ValueError("official_csv must contain columns: season, celebrity_name")
    if not {"season", "week", "window_start", "window_end"}.issubset(windows.columns):
        raise ValueError("windows_csv must contain columns: season, week, window_start, window_end")

    session = requests.Session()
    session.headers.update({
        "User-Agent": os.environ.get("WIKI_UA", "MCM-ICM-DWTS-Research/1.0 (contact: youremail@example.com)"),
        "Accept": "application/json",
    })

    cfg = FetchConfig(cache_dir=Path(args.cache_dir), sleep_sec=args.sleep)

    # title map for audit/manual fix
    title_map_path = Path("wiki_title_map.csv")
    title_map: Dict[str, str] = {}
    if title_map_path.exists():
        tm = pd.read_csv(title_map_path)
        for _, r in tm.iterrows():
            if pd.notna(r.get("celebrity_name")) and pd.notna(r.get("wiki_title")):
                title_map[str(r["celebrity_name"])] = str(r["wiki_title"])

    all_celebs = sorted(official["celebrity_name"].dropna().unique().tolist())
    missing = [c for c in all_celebs if c not in title_map]

    if missing:
        for c in tqdm(missing, desc="Resolving wiki titles"):
            t = resolve_wiki_title(session, c, cfg)
            if t:
                title_map[c] = t
            time.sleep(cfg.sleep_sec)
        pd.DataFrame([{"celebrity_name": k, "wiki_title": v} for k, v in sorted(title_map.items())]).to_csv(title_map_path, index=False)

    seasons = sorted(windows["season"].dropna().unique().tolist())
    if args.max_seasons and args.max_seasons > 0:
        seasons = seasons[: args.max_seasons]

    out_rows = []

    for season in tqdm(seasons, desc="Seasons"):
        win_s = windows[windows["season"] == season].copy()
        if win_s.empty:
            continue

        s_start = win_s["window_start"].min()
        s_end = win_s["window_end"].max()

        celebs = sorted(official.loc[official["season"] == season, "celebrity_name"].dropna().unique().tolist())
        for celeb in tqdm(celebs, desc=f"Season {season} celebs", leave=False):
            title = title_map.get(str(celeb))
            if not title:
                for _, w in win_s.iterrows():
                    out_rows.append({
                        "season": int(season),
                        "week": int(w["week"]),
                        "celebrity_name": str(celeb),
                        "wiki_title": None,
                        "wiki_edits_week": None,
                        "wiki_editors_week": None,
                        "wiki_bytes_net_week": None,
                        "window_start": w["window_start"].isoformat(),
                        "window_end": w["window_end"].isoformat(),
                        "status": "NO_TITLE",
                    })
                continue

            try:
                daily = fetch_revisions(session, title, s_start, s_end, cfg)
            except Exception:
                daily = pd.DataFrame(columns=["date", "edits", "editors", "bytes_net"])

            if daily.empty:
                for _, w in win_s.iterrows():
                    out_rows.append({
                        "season": int(season),
                        "week": int(w["week"]),
                        "celebrity_name": str(celeb),
                        "wiki_title": title,
                        "wiki_edits_week": 0,
                        "wiki_editors_week": 0,
                        "wiki_bytes_net_week": 0,
                        "window_start": w["window_start"].isoformat(),
                        "window_end": w["window_end"].isoformat(),
                        "status": "OK",
                    })
                continue

            daily = daily.set_index("date")
            for _, w in win_s.iterrows():
                ws, we = w["window_start"], w["window_end"]
                mask = (daily.index >= ws) & (daily.index <= we)
                edits = int(daily.loc[mask, "edits"].sum()) if "edits" in daily else 0
                # 周内去重：把该周每天的 editors_set 做 union
                editors = len(set().union(*daily.loc[mask, "editors_set"].tolist())) if "editors_set" in daily else 0
                bytes_net = int(daily.loc[mask, "bytes_net"].sum()) if "bytes_net" in daily else 0

                out_rows.append({
                    "season": int(season),
                    "week": int(w["week"]),
                    "celebrity_name": str(celeb),
                    "wiki_title": title,
                    "wiki_edits_week": edits,
                    "wiki_editors_week": editors,
                    "wiki_bytes_net_week": bytes_net,
                    "window_start": ws.isoformat(),
                    "window_end": we.isoformat(),
                    "status": "OK",
                })

    out = pd.DataFrame(out_rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")
    print(f"Saved: {title_map_path} (manual disambiguation if needed)")


if __name__ == "__main__":
    main()
