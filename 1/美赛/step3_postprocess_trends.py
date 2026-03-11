import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def safe_float(x):
    if pd.isna(x):
        return np.nan
    try:
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def add_features(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """
    输入 df: columns 至少包含 season, week, celebrity, trend_filled
    输出：增加 TrendShare / TrendGrowth / TrendMomentum 等
    """
    df = df.copy()
    df = df.sort_values(["season", "celebrity", "week"], kind="mergesort")

    # TrendLevel
    df["TrendLevel"] = df["trend_filled"].astype(float)

    # TrendShare: 按 (season, week) 归一化份额
    denom = df.groupby(["season", "week"])["TrendLevel"].transform(lambda s: float(np.nansum(s.values)) + eps)
    df["TrendShare"] = df["TrendLevel"] / denom

    # Growth: 本周 - 上周（同一 season + celebrity）
    df["TrendGrowth"] = df.groupby(["season", "celebrity"])["TrendLevel"].diff()

    # Momentum: 3 周滑动均值（同一 season + celebrity）
    df["TrendMomentum3"] = (
        df.groupby(["season", "celebrity"])["TrendLevel"]
          .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )

    # 可选：log1p，避免极端值影响
    df["TrendLog1p"] = np.log1p(df["TrendLevel"])

    return df

def fill_missing_trend(panel: pd.DataFrame) -> pd.DataFrame:
    """
    缺失处理策略（不直接置 0）：
    1) (season, celebrity) 内按 week 插值（线性），只填“中间缺失”
    2) 若仍缺失，用 (season, week) 的中位数回填（同周其它选手）
    3) 若仍缺失，用 (season, celebrity) 的中位数回填（该选手该季）
    4) 若仍缺失，用全局最小正值的一半（表示“极低但非零”）
    """
    df = panel.copy()

    # 确保类型
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["trend"] = df["trend"].apply(safe_float)

    # 记录原始缺失
    df["trend_missing_raw"] = df["trend"].isna()

    # 1) 组内插值（线性）
    df = df.sort_values(["season", "celebrity", "week"], kind="mergesort")
    df["trend_interp"] = (
        df.groupby(["season", "celebrity"])["trend"]
          .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
    )
    # 说明：limit_direction="both" 会填首尾，这里我们允许，因为后面还有同周中位数等更稳的回填；
    # 如果你希望首尾不插值，可改成 limit_area="inside"

    # 2) 同周中位数回填
    week_median = df.groupby(["season", "week"])["trend_interp"].transform(lambda s: np.nanmedian(s.values))
    df["trend_fill_wkmed"] = df["trend_interp"].where(~df["trend_interp"].isna(), week_median)

    # 3) 选手季内中位数回填
    celeb_med = df.groupby(["season", "celebrity"])["trend_fill_wkmed"].transform(lambda s: np.nanmedian(s.values))
    df["trend_fill_cbmed"] = df["trend_fill_wkmed"].where(~df["trend_fill_wkmed"].isna(), celeb_med)

    # 4) 全局兜底：最小正值的一半
    positive_vals = df["trend_fill_cbmed"][df["trend_fill_cbmed"] > 0].dropna().values
    if len(positive_vals) == 0:
        fallback = 1.0  # 极端情况：全为空/全0，给个常数兜底
    else:
        fallback = float(np.min(positive_vals) / 2.0)

    df["trend_filled"] = df["trend_fill_cbmed"].fillna(fallback)

    # 生成缺失标记（用于论文透明度）
    df["trend_filled_from"] = np.where(
        ~df["trend_missing_raw"], "original",
        np.where(~df["trend_interp"].isna(), "interp",
                 np.where(~week_median.isna(), "week_median",
                          np.where(~celeb_med.isna(), "season_celeb_median", "global_fallback")))
    )

    return df

def load_cache_as_panel(cache_path: Path) -> pd.DataFrame:
    """
    兼容两种常见输出：
    - _dwts_trends_cache.csv（一般就是 season, week, celebrity, trend, timeframe, kw）
    - dwts_trends_weekly_panel.csv（同样字段）
    """
    df = pd.read_csv(cache_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    needed = {"season", "week", "celebrity", "trend"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Cache file missing required columns: {needed - set(df.columns)}")

    # 只保留核心列 + 你原始的信息列（若存在）
    keep = ["season", "week", "celebrity", "trend"]
    for extra in ["timeframe", "kw", "source", "err", "attempt"]:
        if extra in df.columns:
            keep.append(extra)

    df = df[keep].copy()
    df["celebrity"] = df["celebrity"].astype(str).str.strip()

    # 去重（同一 season-week-celebrity 若重复，保留最后一次）
    df = df.sort_values(list(df.columns), kind="mergesort")
    df = df.drop_duplicates(subset=["season", "week", "celebrity"], keep="last")

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, required=True, help="Path to _dwts_trends_cache.csv (or weekly panel csv)")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory")
    args = ap.parse_args()

    cache_path = Path(args.cache).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] reading: {cache_path}")
    panel = load_cache_as_panel(cache_path)
    print(f"[INFO] rows={len(panel):,} unique (season,week,celebrity)={panel[['season','week','celebrity']].drop_duplicates().shape[0]:,}")

    # 缺失概览
    raw_missing = panel["trend"].isna().sum()
    print(f"[INFO] raw missing trend count = {raw_missing:,}")

    # 缺失处理
    cleaned = fill_missing_trend(panel)

    # 特征工程
    featured = add_features(cleaned)

    # 输出
    out_clean = outdir / "dwts_trends_weekly_panel_clean.csv"
    out_feat = outdir / "dwts_trends_features.csv"

    cleaned_cols = [
        "season","week","celebrity","trend","trend_filled","trend_missing_raw","trend_filled_from"
    ]
    for extra in ["timeframe", "kw"]:
        if extra in cleaned.columns:
            cleaned_cols.append(extra)

    cleaned[cleaned_cols].to_csv(out_clean, index=False, encoding="utf-8-sig")
    featured_cols = [
        "season","week","celebrity",
        "TrendLevel","TrendShare","TrendGrowth","TrendMomentum3","TrendLog1p",
        "trend_missing_raw","trend_filled_from"
    ]
    featured[featured_cols].to_csv(out_feat, index=False, encoding="utf-8-sig")

    # 关键统计（写论文用）
    filled_total = int((cleaned["trend_missing_raw"]).sum())
    filled_by = cleaned.loc[cleaned["trend_missing_raw"], "trend_filled_from"].value_counts(dropna=False)
    print(f"[DONE] saved:\n  {out_clean}\n  {out_feat}")
    print(f"[STATS] filled rows (originally missing) = {filled_total:,}")
    print("[STATS] filled sources:")
    print(filled_by.to_string())

if __name__ == "__main__":
    main()
