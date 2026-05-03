
import sys
from pathlib import Path
from typing import Final

_SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib, numpy as np, pandas as pd
from pathlib import Path
from src.preprocessing.features import FEATURE_COLUMNS

bundle = joblib.load("models1/isolation_forest_tuned.pkl")
val_df = pd.read_parquet("data/processeded1/val_features.parquet")

X = val_df[FEATURE_COLUMNS].values.astype("float32")
scores = bundle["model"].decision_function(X)  # higher = more normal
anomaly = -scores

attack = val_df["is_attack_window"].values
normal = ~attack

print("=== Score distribution (anomaly score, higher = more anomalous) ===")
print(f"Attack  windows: mean={anomaly[attack].mean():.4f}  "
      f"std={anomaly[attack].std():.4f}  "
      f"p10={np.percentile(anomaly[attack],10):.4f}  "
      f"p50={np.percentile(anomaly[attack],50):.4f}  "
      f"p90={np.percentile(anomaly[attack],90):.4f}")
print(f"Normal  windows: mean={anomaly[normal].mean():.4f}  "
      f"std={anomaly[normal].std():.4f}  "
      f"p10={np.percentile(anomaly[normal],10):.4f}  "
      f"p50={np.percentile(anomaly[normal],50):.4f}  "
      f"p90={np.percentile(anomaly[normal],90):.4f}")

print()
print("=== Alert rate at various score thresholds ===")
for pct in [94, 95, 96, 97, 98, 99]:
    thresh = np.percentile(anomaly, pct)
    flagged = (anomaly >= thresh).mean()
    tp = ((anomaly >= thresh) & attack).sum()
    fp = ((anomaly >= thresh) & normal).sum()
    recall = tp / attack.sum() if attack.sum() > 0 else 0
    fpr = fp / normal.sum() if normal.sum() > 0 else 0
    print(f"  p{pct} threshold={thresh:.4f}  "
          f"alert_rate={flagged:.4f}  recall={recall:.4f}  fpr_normal={fpr:.4f}  "
          f"TP={tp}  FP={fp}")
