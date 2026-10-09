"""Replay an IMS run-to-failure test as if it were streaming from the machine.

    streamlit run app/dashboard.py

Reads only the committed artifacts (data/features, reports/scores.csv.gz); run
``make experiment`` first if reports/ is missing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bearing_efw.dataset import TESTS, fault_frequencies  # noqa: E402
from bearing_efw.preprocess import load_features, prepare  # noqa: E402

COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
STATUS = {
    "ok": ("#0f8a3c", "●", "Healthy"),
    "watch": ("#c98500", "▲", "Above threshold"),
    "alarm": ("#e34948", "■", "ALARM"),
}
DEFECTS = {"env_bpfo": "outer race", "env_bpfi": "inner race", "env_bsf": "rolling element"}

st.set_page_config(page_title="Bearing early warning", page_icon="⚙️", layout="wide")


@st.cache_data
def load_scores() -> pd.DataFrame:
    path = ROOT / "reports" / "scores.csv.gz"
    if not path.exists():
        st.error("reports/scores.csv.gz not found. Run `make experiment` first.")
        st.stop()
    return pd.read_csv(path, parse_dates=["timestamp"])


@st.cache_data
def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())


@st.cache_data
def defect_zscores(test: str) -> dict[int, pd.DataFrame]:
    """Per bearing: z-score (vs. its healthy baseline) of envelope energy at each defect frequency."""
    cfg = load_cfg()
    data = prepare(load_features(ROOT / "data" / "features", test), cfg["split"]["train_end"], cfg["split"]["calib_end"])
    out = {}
    for b, d in data.items():
        cols = list(d.raw.columns)
        z = {k: d.X[:, [i for i, c in enumerate(cols) if c.startswith(k + "_")]].mean(axis=1) for k in DEFECTS}
        z["rms"] = d.X[:, [i for i, c in enumerate(cols) if c.startswith("rms_")]].mean(axis=1)
        out[b] = pd.DataFrame(z, index=d.timestamps)
    return out


def diagnose(z: pd.DataFrame) -> tuple[str, float]:
    recent = z[list(DEFECTS)].tail(6).mean()
    best = recent.idxmax()
    return DEFECTS[best], float(recent[best])


scores = load_scores()
cfg = load_cfg()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Bearing early warning")
test = st.sidebar.selectbox("Run-to-failure test", list(TESTS), index=1,
                            format_func=lambda t: f"{t} ({TESTS[t].folder.split('/')[0]})")
detector = st.sidebar.selectbox("Detector", list(dict.fromkeys(scores["detector"])), index=1)
df = scores[(scores.test == test) & (scores.detector == detector)]
times = np.sort(df["timestamp"].unique())
n = len(times)

key = f"slider_{test}"
first = int(n * cfg["split"]["calib_end"])  # playback starts where monitoring starts
if key not in st.session_state:
    st.session_state[key] = first
if "advance_to" in st.session_state:  # set by the playback loop on the previous run
    st.session_state[key] = st.session_state.pop("advance_to")
st.session_state.setdefault("playing", False)

c1, c2 = st.sidebar.columns(2)
if c1.button("▶ Play" if not st.session_state.playing else "⏸ Pause", use_container_width=True):
    st.session_state.playing = not st.session_state.playing
if c2.button("⏮ Restart", use_container_width=True):
    st.session_state[key] = first
speed = st.sidebar.select_slider("Speed (snapshots per frame)", [1, 2, 5, 10, 20, 50], value=10)
idx = st.sidebar.slider("Snapshot", 1, n, key=key)
reveal = st.sidebar.checkbox("Reveal ground truth", value=idx >= n)

st.sidebar.caption(
    "Detectors were fitted on the first 25 % of snapshots and calibrated on the next 10 %. "
    "An alarm needs 4 of the last 6 snapshots above threshold."
)

now = pd.Timestamp(times[idx - 1])
seen = df[df.timestamp <= now]
start = pd.Timestamp(times[0])
hours_now = (now - start).total_seconds() / 3600
zs = defect_zscores(test)

# ---------------------------------------------------------------- header
st.title("Bearing early-warning monitor")
st.markdown(
    f"**{test}** · detector **{detector}** · snapshot {idx}/{n} · "
    f"**{hours_now:,.1f} h** since start ({now:%Y-%m-%d %H:%M})"
)

# ---------------------------------------------------------------- status tiles
# The first bearing to raise an alarm is the likely source: once it degrades, its
# vibration travels through the shaft and housing and lifts its neighbours too.
onsets = {b: g.loc[g["alarm"], "hours"].min() for b, g in seen.groupby("bearing") if g["alarm"].any()}
source = min(onsets, key=onsets.get) if onsets else None

cols = st.columns(4)
events = []
for col, (b, g) in zip(cols, seen.groupby("bearing")):
    last = g.iloc[-1]
    ratio = last.score / last.threshold if last.threshold > 0 else np.nan
    state = "alarm" if last.alarm else ("watch" if last.score > last.threshold else "ok")
    color, icon, label = STATUS[state]
    # current alarm episode start
    since = ""
    if last.alarm:
        a = g["alarm"].to_numpy()
        start_i = len(a) - np.argmax(~a[::-1]) if (~a).any() else 0
        since_h = g["hours"].iloc[start_i]
        since = f"since {since_h:,.1f} h"
    z = zs[b].loc[:now]
    defect, zval = diagnose(z)
    diag = f"likely <b>{defect}</b> (z={zval:.1f})" if last.alarm and zval > 3 else "no clear defect signature"
    truth = f"<br><span style='color:#52514e'>ground truth: {TESTS[test].failed.get(b, 'survived')}</span>" if reveal else ""
    col.markdown(
        f"""<div style="border:1px solid #e4e3df;border-left:6px solid {color};border-radius:8px;padding:10px 12px">
        <div style="font-size:0.85rem;color:#52514e">Bearing {b}{" · <b>first to alarm</b>" if b == source else ""}</div>
        <div style="font-size:1.35rem;font-weight:700;color:{color}">{icon} {label}</div>
        <div style="font-size:0.85rem">score / threshold <b>{ratio:,.2f}</b> {since}</div>
        <div style="font-size:0.85rem">{diag if last.alarm else ''}{truth}</div></div>""",
        unsafe_allow_html=True,
    )
    a = g["alarm"].to_numpy()
    for s in np.flatnonzero(np.diff(np.concatenate([[0], a.astype(int)])) == 1):
        events.append({"bearing": b, "alarm raised at [h]": round(g["hours"].iloc[s], 1),
                       "time": g["timestamp"].iloc[s]})

# ---------------------------------------------------------------- score chart
st.subheader("Anomaly score")
st.caption("Each bearing's score divided by its own alarm threshold: the dashed line at 1 is the threshold.")
fig = go.Figure()
log_axis = bool((df["score"] > 0).all() and (df["threshold"] > 0).all())
for b, g in seen.groupby("bearing"):
    fig.add_trace(go.Scattergl(x=g["hours"], y=g["score"] / g["threshold"], mode="lines", name=f"Bearing {b}",
                               line=dict(color=COLORS[b], width=1.4)))
fig.add_hline(y=1, line_dash="dash", line_color="#52514e")
t_hours = df[df.bearing == 1]["hours"].to_numpy()
fig.add_vrect(x0=0, x1=t_hours[int(n * cfg["split"]["train_end"]) - 1], fillcolor="#8a8984", opacity=0.08, line_width=0,
              annotation_text="train", annotation_position="top left")
fig.add_vrect(x0=t_hours[int(n * cfg["split"]["train_end"]) - 1], x1=t_hours[int(n * cfg["split"]["calib_end"]) - 1],
              fillcolor="#8a8984", opacity=0.16, line_width=0, annotation_text="calibration", annotation_position="top left")
fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="hours since start",
                  yaxis_title="score / threshold", yaxis_type="log" if log_axis else "linear",
                  xaxis_range=[0, t_hours[-1]], legend=dict(orientation="h", y=1.12), template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- diagnosis chart
left, right = st.columns([3, 2])
with left:
    b_sel = st.radio("Envelope diagnosis for bearing", sorted(zs), horizontal=True,
                     index=sorted(zs).index(min(TESTS[test].failed)) if reveal else 0)
    z = zs[b_sel].loc[:now]
    h = (z.index - start).total_seconds() / 3600
    f2 = go.Figure()
    ff = fault_frequencies()
    for k, c in zip(DEFECTS, ["#e34948", "#4a3aa7", "#1baf7a"]):
        f2.add_trace(go.Scattergl(x=h, y=z[k].rolling(6, min_periods=1).mean(), mode="lines",
                                  name=f"{DEFECTS[k]} ({ff[k[4:]]:.0f} Hz)", line=dict(color=c, width=1.4)))
    f2.add_hline(y=3, line_dash="dot", line_color="#52514e")
    f2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                     xaxis_title="hours since start", yaxis_title="z-score vs. healthy",
                     xaxis_range=[0, t_hours[-1]], legend=dict(orientation="h", y=1.15))
    st.plotly_chart(f2, use_container_width=True)
with right:
    st.markdown("**Alarm log**")
    if events:
        st.dataframe(pd.DataFrame(events).sort_values("alarm raised at [h]"), hide_index=True, use_container_width=True)
    else:
        st.caption("No alarms yet.")
    if reveal:
        end_h = t_hours[-1]
        st.markdown(f"**Ground truth:** test ends at **{end_h:,.0f} h** with "
                    + ", ".join(f"bearing {b} {m}" for b, m in TESTS[test].failed.items()) + ".")

# ---------------------------------------------------------------- playback
if st.session_state.playing:
    if idx < n:
        st.session_state["advance_to"] = min(n, idx + speed)
        time.sleep(0.15)
        st.rerun()
    else:
        st.session_state.playing = False
