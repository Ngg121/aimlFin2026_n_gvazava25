import re
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ----------------------------
# CONFIG
# ----------------------------
LOG_PATH = "/Users/ninogvazava/Desktop/task_3/n_gvazava25_24957_server.log"

# Use "min" or "1min" instead of "T"
AGG_WINDOW = "1min"   # change to "1s" if you want per-second detection
Z_THRESHOLD = 3.0     # try 2.5 or 2.0 if it detects nothing

# ----------------------------
# 1) Parse timestamps from log
# ----------------------------
# Example: [2024-03-22 18:00:09+04:00]
TS_REGEX = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\]")

timestamps = []
bad_lines = 0

with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = TS_REGEX.search(line)
        if not m:
            bad_lines += 1
            continue
        ts_str = m.group(1)
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S%z")
            timestamps.append(dt)
        except ValueError:
            bad_lines += 1

if not timestamps:
    raise RuntimeError("No timestamps parsed. Check regex / log format.")

df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps)})
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Parsed {len(df)} log lines. Skipped {bad_lines} lines.")

# ----------------------------
# 2) Aggregate requests by time bucket
# ----------------------------
df["bucket"] = df["timestamp"].dt.floor(AGG_WINDOW)

traffic = df.groupby("bucket").size().reset_index(name="request_count")
traffic["time_index"] = np.arange(len(traffic), dtype=float)

# ----------------------------
# 3) Regression baseline
# ----------------------------
X = traffic[["time_index"]].values
y = traffic["request_count"].values

model = LinearRegression()
model.fit(X, y)

traffic["predicted"] = model.predict(X)
traffic["residual"] = traffic["request_count"] - traffic["predicted"]

# Standardize residuals -> z-score
scaler = StandardScaler()
traffic["residual_z"] = scaler.fit_transform(traffic[["residual"]])

# Flag anomalies (DDoS candidates)
traffic["is_ddos"] = traffic["residual_z"] > Z_THRESHOLD

# ----------------------------
# 4) Merge flagged buckets into intervals
# ----------------------------
def buckets_to_intervals(buckets: pd.Series, agg_window: str):
    if buckets.empty:
        return []

    buckets = pd.to_datetime(buckets).sort_values().reset_index(drop=True)

    # Convert aggregation window into a timedelta
    step = pd.to_timedelta(agg_window)

    intervals = []
    start = buckets.iloc[0]
    prev = buckets.iloc[0]

    for t in buckets.iloc[1:]:
        if t - prev == step:
            prev = t
        else:
            # end is exclusive: prev + step
            intervals.append((start, prev + step))
            start = t
            prev = t

    intervals.append((start, prev + step))
    return intervals

ddos_buckets = traffic.loc[traffic["is_ddos"], "bucket"]
intervals = buckets_to_intervals(ddos_buckets, AGG_WINDOW)

# ----------------------------
# 5) Output results
# ----------------------------
print("\n=== DDoS suspected intervals (regression residual anomalies) ===")
if not intervals:
    print("No DDoS intervals detected at this threshold.")
else:
    for i, (start, end) in enumerate(intervals, 1):
        print(f"{i}. {start}  -->  {end}")

# Save results
out_csv = "/Users/ninogvazava/Desktop/task_3/ddos_regression_results.csv"
traffic.to_csv(out_csv, index=False)
print(f"\nSaved detailed per-bucket results to: {out_csv}")

# ----------------------------
# 6) Plot
# ----------------------------
plt.figure()
plt.plot(traffic["bucket"], traffic["request_count"], label="Observed")
plt.plot(traffic["bucket"], traffic["predicted"], label="Regression baseline")

plt.scatter(
    traffic.loc[traffic["is_ddos"], "bucket"],
    traffic.loc[traffic["is_ddos"], "request_count"],
    label="Flagged buckets",
)

plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel(f"Requests per {AGG_WINDOW}")
plt.title("DDoS detection using regression residuals")
plt.legend()
plt.tight_layout()
plt.show()


print("\n=== Interval summary ===")
for i, (start, end) in enumerate(intervals, 1):
    window = traffic[(traffic["bucket"] >= start) & (traffic["bucket"] < end)].copy()
    peak_req = window["request_count"].max()
    avg_req = window["request_count"].mean()
    peak_z = window["residual_z"].max()
    print(f"{i}. {start} --> {end} | peak req/min={peak_req:.0f}, avg req/min={avg_req:.1f}, max z={peak_z:.2f}")
