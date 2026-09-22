import json
import math
import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


st.set_page_config(
    page_title="StudyRunway",
    page_icon="◌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
.metric {background:rgba(255,255,255,.72);border:1px solid #dedfd8;border-radius:18px;
 padding:1.1rem 1.2rem;height:170px;box-sizing:border-box}
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
.agent-status {background:#e9f3f0;border-left:4px solid var(--teal);border-radius:10px;
 padding:.8rem 1rem;margin:.35rem 0 1.15rem;color:#29434b;font-size:.95rem}
.agent-status strong {color:var(--ink)}
div[data-testid="stChatMessage"] {background:rgba(255,255,255,.72);border:1px solid #e5e4de;
 border-radius:16px;padding:.35rem .55rem;margin-bottom:.65rem}
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
    followup_months: int
    startup_cost: float
    patient_cost: float
    fixed_monthly: float
    starting_cash: float
    target_months: int


def activation_schedule(sites: int, activation_months: int) -> np.ndarray:
    """Planned activation month for each site."""
    return np.linspace(1, max(1, activation_months), sites).round().astype(int)


def simulate(x: Inputs, horizon: int = 60) -> pd.DataFrame:
    """Create a deterministic expected-value enrollment and cost forecast.

    Participants reach the completion endpoint only after the follow-up period.
    Recruitment stops once the enrolled pipeline is sufficient to produce the
    requested number of expected completers after dropout.
    """
    activation = activation_schedule(x.sites, x.activation_months)
    activated = np.zeros(x.sites, dtype=bool)
    enrolled_cohorts = {}
    required_enrolled = x.target / max(1 - x.dropout, 1e-9)
    rows, screened, enrolled, completed, spend = [], 0.0, 0.0, 0.0, 0.0

    for month in range(1, horizon + 1):
        was_complete = completed >= x.target - 1e-9
        recruiting = enrolled < required_enrolled - 1e-9 and not was_complete

        newly_activated = (activation == month) & (~activated) if recruiting else np.zeros(x.sites, dtype=bool)
        activated |= newly_activated
        new_sites = int(newly_activated.sum())
        active = int(activated.sum())

        new_screened = active * x.screened_per_site if recruiting else 0.0
        potential_enrolled = new_screened * (1 - x.screen_fail)
        new_enrolled = min(potential_enrolled, max(required_enrolled - enrolled, 0.0))
        if potential_enrolled > 0:
            new_screened *= new_enrolled / potential_enrolled
        enrolled_cohorts[month] = new_enrolled

        completing_cohort_month = month - x.followup_months
        new_completed = enrolled_cohorts.get(completing_cohort_month, 0.0) * (1 - x.dropout)
        new_completed = min(new_completed, max(x.target - completed, 0.0))

        screened += new_screened
        enrolled += new_enrolled
        completed += new_completed
        program_active = not was_complete
        monthly_spend = (
            new_sites * x.startup_cost
            + new_enrolled * x.patient_cost
            + (x.fixed_monthly if program_active else 0.0)
        )
        spend += monthly_spend
        rows.append(dict(month=month, active_sites=active, screened=screened, enrolled=enrolled,
                         completed=completed, new_screened=new_screened, new_enrolled=new_enrolled,
                         new_completed=new_completed, monthly_spend=monthly_spend, cumulative_spend=spend,
                         cash_remaining=x.starting_cash-spend))
    return pd.DataFrame(rows)


def active_site_months(sites: int, activation_months: int, through_month: int) -> int:
    """Exact recruitment capacity contributed by the planned site ramp."""
    if through_month < 1:
        return 0
    activation = activation_schedule(sites, activation_months)
    return int(np.maximum(through_month - activation + 1, 0).sum())


def minimum_screening_rate(x: Inputs) -> float:
    recruitment_deadline = x.target_months - x.followup_months
    capacity = active_site_months(x.sites, x.activation_months, recruitment_deadline)
    net_yield = (1 - x.screen_fail) * (1 - x.dropout)
    return x.target / (capacity * net_yield) if capacity > 0 and net_yield > 0 else math.inf


def minimum_sites(x: Inputs, maximum: int = 500) -> int | None:
    recruitment_deadline = x.target_months - x.followup_months
    net_yield = (1 - x.screen_fail) * (1 - x.dropout)
    for candidate in range(1, maximum + 1):
        capacity = active_site_months(candidate, x.activation_months, recruitment_deadline)
        expected_completers = capacity * x.screened_per_site * net_yield
        if expected_completers >= x.target - 1e-9:
            return candidate
    return None


def money(v):
    return f"${v/1_000_000:.1f}M" if abs(v) >= 1_000_000 else f"${v/1_000:.0f}K"


def scenario_summary(x: Inputs) -> dict:
    """Return auditable scenario results for the agent and the UI."""
    horizon = max(60, x.target_months + x.followup_months + 24)
    result = simulate(x, horizon=horizon)
    complete = result[result.completed >= x.target - 0.01]
    completion_month = int(complete.month.iloc[0]) if not complete.empty else None
    completion_index = completion_month - 1 if completion_month else len(result) - 1
    total_cost = float(result.iloc[completion_index].cumulative_spend)
    funding_gap = max(0.0, total_cost - x.starting_cash)
    negative_cash = result[df_column_negative(result, "cash_remaining")]
    runout_month = int(negative_cash.month.iloc[0]) if not negative_cash.empty else None
    return {
        "target_completed_participants": x.target,
        "desired_completion_month": x.target_months,
        "projected_completion_month": completion_month,
        "deadline_met": bool(completion_month and completion_month <= x.target_months),
        "schedule_variance_months": (
            completion_month - x.target_months if completion_month else None
        ),
        "planned_sites": x.sites,
        "site_activation_months": x.activation_months,
        "screened_per_active_site_per_month": round(x.screened_per_site, 3),
        "screen_failure_rate": round(x.screen_fail, 4),
        "dropout_rate": round(x.dropout, 4),
        "followup_months": x.followup_months,
        "estimated_cost": round(total_cost, 2),
        "funding_available": round(x.starting_cash, 2),
        "funding_gap": round(funding_gap, 2),
        "cash_runout_month": runout_month,
    }


def df_column_negative(frame: pd.DataFrame, column: str) -> pd.Series:
    """Small helper kept separate so scenario calculations are easy to test."""
    return frame[column] < 0


def evaluate_agent_scenario(base: Inputs, arguments: dict) -> dict:
    """Evaluate one bounded alternative using the trusted simulation engine."""
    candidate = replace(
        base,
        sites=max(1, min(int(arguments["sites"]), 100)),
        activation_months=max(1, min(int(arguments["activation_months"]), 36)),
        screened_per_site=max(0.1, min(float(arguments["screened_per_site"]), 20.0)),
        target_months=max(
            base.followup_months + 1,
            min(int(arguments["desired_completion_month"]), 60),
        ),
    )
    return scenario_summary(candidate)


def search_recovery_options(base: Inputs, arguments: dict) -> dict:
    """Search a bounded grid and return the lowest modeled-cost feasible plans."""
    deadline = max(
        base.followup_months + 1,
        min(int(arguments["desired_completion_month"]), 60),
    )
    max_extra_sites = max(0, min(int(arguments["max_extra_sites"]), 20))
    max_screening_multiplier = max(
        1.0, min(float(arguments["max_screening_multiplier"]), 3.0)
    )
    site_values = range(base.sites, base.sites + max_extra_sites + 1)
    rate_values = np.linspace(
        base.screened_per_site,
        base.screened_per_site * max_screening_multiplier,
        13,
    )
    feasible = []
    for candidate_sites in site_values:
        for candidate_rate in rate_values:
            candidate = replace(
                base,
                sites=int(candidate_sites),
                screened_per_site=float(candidate_rate),
                target_months=deadline,
            )
            outcome = scenario_summary(candidate)
            if outcome["deadline_met"] and outcome["funding_gap"] <= 0:
                outcome["additional_sites"] = candidate_sites - base.sites
                outcome["screening_rate_change"] = round(
                    candidate_rate - base.screened_per_site, 3
                )
                feasible.append(outcome)

    feasible.sort(
        key=lambda item: (
            item["estimated_cost"],
            item["additional_sites"],
            item["screening_rate_change"],
        )
    )
    return {
        "desired_completion_month": deadline,
        "scenarios_evaluated": len(site_values) * len(rate_values),
        "feasible_scenarios_found": len(feasible),
        "lowest_modeled_cost_options": feasible[:5],
        "important_limitation": (
            "Screening-rate improvement has no separate implementation cost in this "
            "prototype. 'Lowest modeled cost' is not necessarily the most operationally "
            "feasible or least expensive real-world option."
        ),
    }


def diagnose_trial_risk(base: Inputs) -> dict:
    """Compare approved one-factor interventions against the current plan."""
    baseline = scenario_summary(base)
    baseline_month = baseline["projected_completion_month"]
    interventions = [
        ("Activate all planned sites 25% faster", replace(base, activation_months=max(1, math.ceil(base.activation_months * 0.75))), "site activation"),
        ("Add two sites", replace(base, sites=min(100, base.sites + 2)), "site capacity"),
        ("Increase screening per site by 25%", replace(base, screened_per_site=min(20.0, base.screened_per_site * 1.25)), "screening productivity"),
        ("Reduce screen failure by 5 percentage points", replace(base, screen_fail=max(0.0, base.screen_fail - 0.05)), "screening conversion"),
        ("Reduce dropout by 5 percentage points", replace(base, dropout=max(0.0, base.dropout - 0.05)), "participant retention"),
    ]
    comparisons = []
    for label, candidate, lever in interventions:
        outcome = scenario_summary(candidate)
        candidate_month = outcome["projected_completion_month"]
        months_recovered = (
            baseline_month - candidate_month
            if baseline_month is not None and candidate_month is not None
            else None
        )
        comparisons.append({
            "intervention": label,
            "lever": lever,
            "projected_completion_month": candidate_month,
            "months_recovered": months_recovered,
            "estimated_cost": outcome["estimated_cost"],
            "incremental_modeled_cost": round(outcome["estimated_cost"] - baseline["estimated_cost"], 2),
            "deadline_met": outcome["deadline_met"],
        })
    comparisons.sort(key=lambda item: (
        -(item["months_recovered"] if item["months_recovered"] is not None else -999),
        item["incremental_modeled_cost"],
    ))
    return {
        "baseline": baseline,
        "status": "on_track" if baseline["deadline_met"] else "schedule_at_risk",
        "one_factor_intervention_comparison": comparisons,
        "interpretation_rule": (
            "A larger modeled improvement indicates sensitivity to that lever; it does "
            "not prove the lever is the real-world root cause or operationally feasible."
        ),
        "data_scope": "Fictional planning assumptions; no patient-level data.",
    }


AGENT_TOOLS = [
    {
        "type": "function",
        "name": "get_current_plan",
        "description": "Read the current StudyRunway assumptions and calculated outcome.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "diagnose_trial_risk",
        "description": (
            "Diagnose schedule risk by comparing the current plan with approved "
            "one-factor interventions. Use this when asked why the plan is at risk, "
            "which levers matter most, or what should be investigated first."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "evaluate_scenario",
        "description": (
            "Run one approved StudyRunway scenario. Use this for a specific combination "
            "of sites, activation timing, screening rate, and deadline."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sites": {"type": "integer", "minimum": 1, "maximum": 100},
                "activation_months": {"type": "integer", "minimum": 1, "maximum": 36},
                "screened_per_site": {"type": "number", "minimum": 0.1, "maximum": 20},
                "desired_completion_month": {"type": "integer", "minimum": 2, "maximum": 60},
            },
            "required": [
                "sites",
                "activation_months",
                "screened_per_site",
                "desired_completion_month",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_recovery_options",
        "description": (
            "Search bounded combinations of added sites and higher screening rates, then "
            "return feasible options ranked by modeled cost."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "desired_completion_month": {"type": "integer", "minimum": 2, "maximum": 60},
                "max_extra_sites": {"type": "integer", "minimum": 0, "maximum": 20},
                "max_screening_multiplier": {"type": "number", "minimum": 1, "maximum": 3},
            },
            "required": [
                "desired_completion_month",
                "max_extra_sites",
                "max_screening_multiplier",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


AGENT_INSTRUCTIONS = """
You are StudyRunway's Trial Risk Agent. You support scenario exploration for a
fictional clinical-trial planning prototype.

Rules:
- Use a tool before stating any number about cost, funding, sites, or timing.
- Never perform arithmetic projections yourself and never invent a result.
- When asked why a plan is at risk or which lever matters most, call
  diagnose_trial_risk and describe the result as sensitivity analysis, not
  proof of root cause.
- When asked to recommend a recovery plan, call search_recovery_options.
- Say "lowest modeled cost" rather than "cheapest" and repeat the returned
  limitation about unpriced operational effort.
- Clearly distinguish assumptions, simulation results, and recommendations for
  human consideration.
- Do not make clinical, regulatory, investment, patient-selection, or patient-
  recruitment decisions.
- Do not request or accept protected health information or identifiable patient data.
- Write for a general business audience, not a statistician.
- Keep the answer under 120 words and use no more than three short bullets.
- Lead with the direct answer. Then explain the most important change and what it
  could improve. End with one short sentence saying the results use fictional data.
- Avoid jargon such as "sensitivity analysis," "modeled lever," "one-factor
  intervention," "root cause," and "operational feasibility." Use ordinary phrases
  such as "the model tested," "the biggest improvement," and "the study team must
  decide what is realistic."
"""


def run_trial_risk_agent(
    client, model: str, base: Inputs, messages: list[dict]
) -> tuple[str, list[str]]:
    """Run a short, bounded function-calling loop."""
    trace = []
    input_items = [
        {"role": item["role"], "content": item["content"]}
        for item in messages[-8:]
    ]
    for _ in range(4):
        response = client.responses.create(
            model=model,
            instructions=AGENT_INSTRUCTIONS,
            tools=AGENT_TOOLS,
            input=input_items,
        )
        input_items += response.output
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            answer = response.output_text or "I could not produce an answer from the available tools."
            return answer, trace

        for call in calls:
            args = json.loads(call.arguments or "{}")
            if call.name == "get_current_plan":
                result = scenario_summary(base)
                trace.append("Read the current StudyRunway plan and calculated outcome.")
            elif call.name == "diagnose_trial_risk":
                result = diagnose_trial_risk(base)
                trace.append(
                    "Tested five possible changes using the StudyRunway simulation."
                )
            elif call.name == "evaluate_scenario":
                result = evaluate_agent_scenario(base, args)
                trace.append(
                    "Simulated a plan with "
                    f"{args['sites']} sites, {args['screened_per_site']:g} screened per "
                    f"site/month and a month-{args['desired_completion_month']} deadline."
                )
            elif call.name == "search_recovery_options":
                result = search_recovery_options(base, args)
                trace.append(
                    f"Tested {result['scenarios_evaluated']} recovery plans and found "
                    f"{result['feasible_scenarios_found']} that reached the deadline."
                )
            else:
                result = {"error": "Tool not allowed."}
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )
    return "I reached the scenario-analysis limit. Please ask a narrower question.", trace


def app_secret(name: str, default=None):
    """Read a Streamlit secret, with an environment-variable fallback for local use."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


with st.sidebar:
    st.markdown("### Trial assumptions")
    st.caption("Adjust the plan. All figures are illustrative.")
    target = st.slider("Completed participant target", 100, 800, 400, 25)
    sites = st.slider("Planned sites", 4, 50, 20)
    activation_months = st.slider("Months to activate all sites", 1, 18, 8)
    screened_rate = st.slider("Screened per active site / month", 0.5, 8.0, 3.0, 0.25)
    screen_fail = st.slider("Screen failure", 0, 70, 30) / 100
    dropout = st.slider("Participant dropout", 0, 40, 12) / 100
    followup_months = st.slider("Months from enrollment to endpoint", 1, 12, 3)
    st.markdown("### Funding assumptions")
    startup_cost = st.number_input("Startup cost per site", 10_000, 500_000, 125_000, 5_000)
    patient_cost = st.number_input("Cost per enrolled participant", 5_000, 150_000, 35_000, 1_000)
    fixed_monthly = st.number_input("Fixed monthly program cost", 50_000, 2_000_000, 350_000, 25_000)
    starting_cash = st.number_input("Funding available", 1_000_000, 100_000_000, 25_000_000, 500_000)
    target_months = st.slider("Desired completion deadline (months)", 6, 48, 14)

x = Inputs(target, sites, activation_months, screened_rate, screen_fail, dropout, followup_months,
           startup_cost, patient_cost, fixed_monthly, starting_cash, target_months)
df = simulate(x)
complete_rows = df[df.completed >= x.target - 0.01]
completion_month = int(complete_rows.month.iloc[0]) if not complete_rows.empty else None
completion_idx = completion_month - 1 if completion_month else len(df) - 1
total_cost = float(df.iloc[completion_idx].cumulative_spend)
funding_gap = max(0, total_cost - x.starting_cash)
runout = df[df.cash_remaining < 0]
runout_month = int(runout.month.iloc[0]) if not runout.empty else None
st.markdown("""
<div class="hero">
<div class="eyebrow">StudyRunway · Clinical Trial Scenario Planner</div>
<h1>Can your clinical trial reach its participant target—on time and within budget?</h1>
<p>Explore how site activation, recruitment, screening failures, follow-up and dropouts shape participant timelines and funding requirements.</p></div>
""", unsafe_allow_html=True)

cols = st.columns(4)
cards = [
    ("Participant target reached", f"Month {completion_month}" if completion_month else "60+ months", f"Deadline: month {x.target_months}"),
    ("Estimated cost", money(total_cost), "Through participant completion" if completion_month else "Through month 60"),
    ("Funding gap", money(funding_gap), "Additional capital indicated" if funding_gap else "Plan remains within funding"),
    ("Cash runway", f"Month {runout_month}" if runout_month else "Beyond completion", "First projected negative balance" if runout_month else "Funding covers completion"),
]
for col, (label,value,sub) in zip(cols,cards):
    col.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Trial Outlook", "AI Rescue Agent"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.month, y=df.completed, name="Completed", fill="tozeroy", line=dict(color="#1f847f",width=3), fillcolor="rgba(31,132,127,.14)"))
    fig.add_trace(go.Scatter(x=df.month, y=df.enrolled, name="Enrolled", line=dict(color="#e78665",width=2,dash="dot")))
    fig.add_hline(y=x.target, line_dash="dash", line_color="#142536", annotation_text="Participant target")
    fig.add_vline(x=x.target_months, line_dash="dot", line_color="#9aa1a6", annotation_text="Desired deadline")
    fig.update_layout(title="Participant forecast", xaxis_title="Month", yaxis_title="Participants", height=430,
                      margin=dict(l=20,r=20,t=60,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)",legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    if completion_month and completion_month <= x.target_months:
        st.success(f"Projected completion is month {completion_month}, providing {x.target_months-completion_month} month(s) of schedule margin.")
    else:
        late = completion_month - x.target_months if completion_month else f"more than {60-x.target_months}"
        st.warning(f"Projected completion is month {completion_month or '60+'} — {late} month(s) behind the desired deadline.")
    st.markdown('<div class="note"><b>Planning note:</b> Results use fictional assumptions. Replace them with validated clinical-operations and finance inputs before making decisions.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### AI Trial Rescue Agent")
    st.caption("Ask what is driving the delay or test a recovery plan.")

    variance = completion_month - x.target_months if completion_month else None
    if variance and variance > 0:
        status_text = (
            f"Current forecast: <strong>Month {completion_month}</strong> — "
            f"{variance} months behind the Month {x.target_months} deadline."
        )
    else:
        status_text = (
            f"Current forecast: <strong>Month {completion_month or '60+'}</strong> — "
            f"on schedule for the Month {x.target_months} deadline."
        )
    st.markdown(
        f'<div class="agent-status">{status_text}</div>',
        unsafe_allow_html=True,
    )

    scenario_signature = repr(x)
    if st.session_state.get("agent_scenario_signature") != scenario_signature:
        st.session_state.agent_scenario_signature = scenario_signature
        st.session_state.agent_messages = []
        st.session_state.agent_requests = 0

    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "agent_requests" not in st.session_state:
        st.session_state.agent_requests = 0

    if st.session_state.agent_messages:
        _, top_right = st.columns([5, 1])
        with top_right:
            if st.button("Clear", key="clear_agent_chat", use_container_width=True):
                st.session_state.agent_messages = []
                st.session_state.agent_requests = 0
                st.rerun()

    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("trace"):
                with st.expander("What the agent tested"):
                    for action in message["trace"]:
                        st.markdown(f"- {action}")

    selected_prompt = None
    if not st.session_state.agent_messages:
        st.caption("START WITH A QUESTION")
        suggested = [
            "Why is this trial projected to finish late?",
            f"Find a recovery plan to finish by month {x.target_months}.",
        ]
        suggestion_columns = st.columns(2)
        for index, suggestion in enumerate(suggested):
            if suggestion_columns[index].button(
                suggestion, key=f"agent_suggestion_{index}", use_container_width=True
            ):
                selected_prompt = suggestion

    typed_prompt = st.chat_input(
        "Ask about the delay or test a recovery plan…",
        key="trial_risk_agent_input",
        max_chars=600,
    )
    prompt = selected_prompt or typed_prompt

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.agent_messages.append({"role": "user", "content": prompt})

        api_key = app_secret("OPENAI_API_KEY")
        model = app_secret("OPENAI_MODEL", "gpt-5.6-luna")
        if OpenAI is None:
            answer = (
                "The OpenAI Python package is not installed. Add `openai>=2.0.0` "
                "to `requirements.txt`, redeploy, and try again."
            )
            trace = []
        elif not api_key:
            answer = (
                "The agent is not configured yet. Add `OPENAI_API_KEY` to Streamlit "
                "Secrets; do not place the key in this file or commit it to GitHub."
            )
            trace = []
        elif st.session_state.agent_requests >= 8:
            answer = (
                "This demo's eight-request session limit has been reached. Clear the "
                "conversation to begin a new demonstration."
            )
            trace = []
        else:
            try:
                with st.spinner("Testing the scenario with StudyRunway…"):
                    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=1)
                    answer, trace = run_trial_risk_agent(
                        client,
                        str(model),
                        x,
                        st.session_state.agent_messages,
                    )
                st.session_state.agent_requests += 1
            except Exception as exc:
                trace = []
                answer = (
                    "I couldn't complete the scenario analysis. Confirm the API key, "
                    "model access and account limits, then try again. "
                    f"Technical detail: `{type(exc).__name__}`."
                )

        with st.chat_message("assistant"):
            st.markdown(answer)
            if trace:
                with st.expander("What the agent tested"):
                    for action in trace:
                        st.markdown(f"- {action}")
        st.session_state.agent_messages.append(
            {"role": "assistant", "content": answer, "trace": trace}
        )
        st.rerun()

    with st.expander("About this demo"):
        st.caption(
            "The agent uses StudyRunway's simulation tools and fictional assumptions. "
            "It supports human review and does not make clinical, regulatory, financial "
            "or patient-level decisions."
        )

st.markdown("""<div class="footer"><b>StudyRunway</b> · Clinical Trial Scenario Planner · Generic demonstration using fictional assumptions. Not affiliated with any sponsor, medicine or clinical study. Not medical, regulatory or investment advice.</div>""",unsafe_allow_html=True)
