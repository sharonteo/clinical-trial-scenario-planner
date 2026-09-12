import math
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="TrialPath", page_icon="◌", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700&display=swap');
:root { --ink:#142536; --muted:#65717d; --teal:#1f847f; --mint:#dcefeb; --cream:#f7f4ee; --coral:#e78665; }
.stApp { background: var(--cream); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background:#eef0e9; border-right:1px solid #dfe2da; }
h1,h2,h3 { font-family:Manrope,sans-serif !important; color:var(--ink) !important; letter-spacing:-.03em; }
p,div,label { font-family:DM Sans,sans-serif; }
.hero {padding:2.2rem 2.4rem; border-radius:24px; color:white; margin-bottom:1.4rem;
 background:radial-gradient(circle at 85% 10%,#4aa49a 0,transparent 27%),linear-gradient(120deg,#132c3c,#1e5d64);}
.eyebrow {font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:.75rem;color:#bfe4df;margin-bottom:.8rem}
.hero h1 {font-size:3rem;line-height:1.05;color:white !important;max-width:800px;margin:.2rem 0 1rem}
.hero p {font-size:1.08rem;max-width:740px;color:#dbe8e6;margin:0}
.metric {background:rgba(255,255,255,.72);border:1px solid #dedfd8;border-radius:18px;padding:1.1rem 1.2rem;min-height:112px}
.metric .label {color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700}
.metric .value {font:700 1.65rem Manrope;color:var(--ink);margin:.35rem 0}.metric .sub {font-size:.82rem;color:var(--muted)}
.status {display:inline-block;padding:.35rem .75rem;border-radius:999px;font-weight:700;font-size:.78rem;background:#dcefeb;color:#17655f}
.warning {background:#fbe7df;color:#9c4b32}.complete {background:#dcefeb;color:#17655f}
.note {border-left:3px solid var(--teal);padding:.55rem 1rem;color:var(--muted);font-size:.9rem}
.site {background:#fff;border:1px solid #e2e3dd;border-radius:14px;padding:.8rem 1rem;margin:.35rem 0}
.site strong {color:var(--ink)}
.footer {color:#7d868c;font-size:.78rem;padding:2rem 0 1rem;border-top:1px solid #dedfd8;margin-top:2rem}
div[data-testid="stMetric"] {background:white;border:1px solid #e2e3dd;padding:1rem;border-radius:16px}
.stButton button {border-radius:999px;border:0;background:#1f847f;color:white;font-weight:700;padding:.55rem 1.3rem}
.stButton button:hover {background:#176d68;color:white;border:0}
</style>
""", unsafe_allow_html=True)


@dataclass
class Inputs:
    target: int
    sites: int
    activation_months: int
    screened_per_site: float
    screen_fail: float
    dropout: float
    startup_cost: float
    patient_cost: float
    fixed_monthly: float
    starting_cash: float
    target_months: int


def simulate(x: Inputs, horizon: int = 60) -> pd.DataFrame:
    activation = np.linspace(1, max(1, x.activation_months), x.sites).round().astype(int)
    rows, screened, enrolled, completed, spend = [], 0.0, 0.0, 0.0, 0.0
    for month in range(1, horizon + 1):
        active = int((activation <= month).sum())
        new_sites = int((activation == month).sum())
        new_screened = active * x.screened_per_site if completed < x.target else 0
        new_enrolled = new_screened * (1 - x.screen_fail)
        new_completed = new_enrolled * (1 - x.dropout)
        if completed + new_completed > x.target:
            scale = (x.target - completed) / max(new_completed, 1e-9)
            new_screened *= max(scale, 0)
            new_enrolled *= max(scale, 0)
            new_completed = max(x.target - completed, 0)
        screened += new_screened
        enrolled += new_enrolled
        completed += new_completed
        monthly_spend = new_sites * x.startup_cost + new_enrolled * x.patient_cost + x.fixed_monthly
        spend += monthly_spend
        rows.append(dict(month=month, active_sites=active, screened=screened, enrolled=enrolled,
                         completed=completed, monthly_spend=monthly_spend, cumulative_spend=spend,
                         cash_remaining=x.starting_cash-spend))
    return pd.DataFrame(rows)


def money(v):
    return f"${v/1_000_000:.1f}M" if abs(v) >= 1_000_000 else f"${v/1_000:.0f}K"


with st.sidebar:
    st.markdown("### Trial assumptions")
    st.caption("Adjust the plan. All figures are illustrative.")
    target = st.slider("Completed participant target", 100, 800, 400, 25)
    sites = st.slider("Planned sites", 4, 50, 20)
    activation_months = st.slider("Months to activate all sites", 1, 18, 8)
    screened_rate = st.slider("Screened per active site / month", 0.5, 8.0, 3.0, 0.25)
    screen_fail = st.slider("Screen failure", 0, 70, 30) / 100
    dropout = st.slider("Participant dropout", 0, 40, 12) / 100
    st.markdown("### Funding assumptions")
    startup_cost = st.number_input("Startup cost per site", 10_000, 500_000, 125_000, 5_000)
    patient_cost = st.number_input("Cost per enrolled participant", 5_000, 150_000, 35_000, 1_000)
    fixed_monthly = st.number_input("Fixed monthly program cost", 50_000, 2_000_000, 350_000, 25_000)
    starting_cash = st.number_input("Funding available", 1_000_000, 100_000_000, 25_000_000, 500_000)
    target_months = st.slider("Desired enrollment deadline (months)", 6, 48, 24)

x = Inputs(target, sites, activation_months, screened_rate, screen_fail, dropout,
           startup_cost, patient_cost, fixed_monthly, starting_cash, target_months)
df = simulate(x)
complete_rows = df[df.completed >= x.target - 0.01]
completion_month = int(complete_rows.month.iloc[0]) if not complete_rows.empty else None
completion_idx = completion_month - 1 if completion_month else len(df) - 1
total_cost = float(df.iloc[completion_idx].cumulative_spend)
funding_gap = max(0, total_cost - x.starting_cash)
runout = df[df.cash_remaining < 0]
runout_month = int(runout.month.iloc[0]) if not runout.empty else None
net_completion_rate = (1-x.screen_fail)*(1-x.dropout)
effective_site_months = max(x.target_months - x.activation_months / 2, 1)
needed_rate = x.target / max(x.sites * effective_site_months * net_completion_rate, .01)

st.markdown("""
<div class="hero"><div class="eyebrow">TrialPath · Scenario Intelligence</div>
<h1>Can your clinical trial finish on time—and within budget?</h1>
<p>Explore how site activation, participant recruitment, screening failures and dropouts shape enrollment timelines and funding requirements.</p></div>
""", unsafe_allow_html=True)

cols = st.columns(4)
cards = [
    ("Enrollment complete", f"Month {completion_month}" if completion_month else "60+ months", f"Target: month {x.target_months}"),
    ("Estimated cost", money(total_cost), "Through enrollment completion"),
    ("Funding gap", money(funding_gap), "Additional capital indicated" if funding_gap else "Plan remains within funding"),
    ("Cash runway", f"Month {runout_month}" if runout_month else "Beyond plan", "First projected negative balance" if runout_month else "Funding covers enrollment"),
]
for col, (label,value,sub) in zip(cols,cards):
    col.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

st.markdown("## Explore the forecast")
tab1, tab2, tab3 = st.tabs(["Forecast", "Run simulation", "Meet the target"])

with tab1:
    left, right = st.columns([1.55,1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.month, y=df.completed, name="Completed", fill="tozeroy", line=dict(color="#1f847f",width=3), fillcolor="rgba(31,132,127,.14)"))
        fig.add_trace(go.Scatter(x=df.month, y=df.enrolled, name="Enrolled", line=dict(color="#e78665",width=2,dash="dot")))
        fig.add_hline(y=x.target, line_dash="dash", line_color="#142536", annotation_text="Enrollment target")
        fig.add_vline(x=x.target_months, line_dash="dot", line_color="#9aa1a6", annotation_text="Desired deadline")
        fig.update_layout(title="Participant forecast", xaxis_title="Month", yaxis_title="Participants", height=410,
                          margin=dict(l=20,r=20,t=60,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)",legend_orientation="h")
        st.plotly_chart(fig, width="stretch")
    with right:
        cash = go.Figure()
        cash.add_trace(go.Scatter(x=df.month, y=df.cash_remaining/1e6, fill="tozeroy", line=dict(color="#244e65",width=3), fillcolor="rgba(36,78,101,.14)"))
        cash.add_hline(y=0,line_color="#e78665",line_dash="dash")
        cash.update_layout(title="Funding runway", xaxis_title="Month", yaxis_title="$ millions remaining", height=410,
                           margin=dict(l=20,r=20,t=60,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)",showlegend=False)
        st.plotly_chart(cash, width="stretch")

with tab2:
    st.caption("Watch the operating plan unfold month by month.")
    speed = st.select_slider("Playback speed", options=["Slow","Normal","Fast"], value="Normal")
    delay = {"Slow":.45,"Normal":.22,"Fast":.08}[speed]
    run = st.button("▶ Run simulation", type="primary")
    stage = st.empty()
    months_to_show = min(completion_month or 36, 36)
    def render_month(m):
        r=df.iloc[m-1]
        health = "complete" if r.completed >= x.target else ("warning" if r.cash_remaining < 0 else "")
        status = "Target reached" if r.completed >= x.target else ("Funding exhausted" if r.cash_remaining < 0 else "Recruiting")
        stage.markdown(f"""
        <div class="metric" style="padding:1.4rem">
          <span class="status {health}">{status}</span>
          <h2 style="margin:.7rem 0">Month {m}</h2>
          <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1rem">
           <div><small>ACTIVE SITES</small><h3>{int(r.active_sites)} / {x.sites}</h3></div>
           <div><small>SCREENED</small><h3>{int(r.screened)}</h3></div>
           <div><small>ENROLLED</small><h3>{int(r.enrolled)}</h3></div>
           <div><small>COMPLETED</small><h3>{int(r.completed)} / {x.target}</h3></div>
           <div><small>CASH REMAINING</small><h3>{money(r.cash_remaining)}</h3></div>
          </div>
          <div style="height:10px;background:#e3e6df;border-radius:999px;overflow:hidden"><div style="height:100%;width:{min(100,r.completed/x.target*100):.1f}%;background:#1f847f"></div></div>
        </div>""", unsafe_allow_html=True)
    if run:
        for m in range(1,months_to_show+1):
            render_month(m); time.sleep(delay)
    else:
        render_month(1)

with tab3:
    st.markdown("### What must be true to hit the deadline?")
    st.markdown(f"To complete **{x.target:,} participants by month {x.target_months}** with the current site plan:")
    a,b,c=st.columns(3)
    a.metric("Minimum screening rate", f"{needed_rate:.1f}", "per active site / month")
    needed_sites = math.ceil(x.target / max(x.screened_per_site*effective_site_months*net_completion_rate,.01))
    b.metric("Sites required", f"{needed_sites}", f"{needed_sites-x.sites:+d} vs current plan")
    budget_needed = float(df[df.month<=x.target_months].cumulative_spend.iloc[-1])
    c.metric("Funding through deadline", money(budget_needed), "Based on current cost assumptions")
    if completion_month and completion_month <= x.target_months:
        st.success(f"Current assumptions meet the deadline with approximately {x.target_months-completion_month} month(s) of schedule margin.")
    else:
        late=(completion_month-x.target_months) if completion_month else f"more than {60-x.target_months}"
        st.warning(f"Current assumptions miss the desired deadline by {late} month(s). Increase recruitment capacity, activate sites faster, or revisit the target.")
    st.markdown('<div class="note"><b>Decision insight:</b> This is a planning model, not a clinical or financial forecast. Replace illustrative assumptions with validated inputs from clinical operations, finance, regulatory and manufacturing teams.</div>',unsafe_allow_html=True)

st.markdown("""<div class="footer"><b>TrialPath</b> · Generic demonstration using fictional assumptions. Not affiliated with any sponsor, medicine or clinical study. Not medical, regulatory or investment advice.</div>""",unsafe_allow_html=True)
