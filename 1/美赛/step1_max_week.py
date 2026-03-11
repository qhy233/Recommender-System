import pandas as pd
import numpy as np
import re

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "2026_MCM_Problem_C_Data.csv"
df = pd.read_csv(CSV_PATH, na_values=["N/A"])
df.columns = df.columns.str.strip()

# 找所有 week*_judge*_score 列
judge_cols = [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]
weeks = sorted({int(re.findall(r"week(\d+)_", c)[0]) for c in judge_cols})

# 对每个 season 统计“出现过有效评分”的最大周
max_week_by_season = {}

for s, sdf in df.groupby("season"):
    max_w = 0
    for w in weeks:
        wcols = [c for c in judge_cols if c.startswith(f"week{w}_")]
        block = sdf[wcols].apply(pd.to_numeric, errors="coerce")
        # 只要这一周有任意选手任意评委分数>0，就视为该周存在
        if (block.fillna(0).to_numpy() > 0).any():
            max_w = w
    max_week_by_season[int(s)] = max_w

out = pd.DataFrame(
    [{"season": s, "max_week_in_official_csv": mw} for s, mw in sorted(max_week_by_season.items())]
)
out.to_csv(BASE_DIR / "season_max_week.csv", index=False)
print("Saved:", BASE_DIR / "season_max_week.csv")
print(out.head(10))

