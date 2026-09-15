# StudyRunway

**An interactive clinical-trial participant and funding scenario planner.**

[Open the live StudyRunway app](https://clinical-trial-scenario-planner.streamlit.app/)

StudyRunway helps teams explore how changes in site activation, recruitment, screening failures, participant dropout, follow-up time and funding assumptions can affect a clinical trial's projected timeline and budget.

The app is designed for scenario planning: change an assumption and immediately see its potential operational and financial impact.

## What the app shows

- Projected month when the completed-participant target is reached
- Estimated cost through participant completion
- Potential funding gap and cash-runway risk
- Participant enrollment and completion trajectories
- Minimum screening rate needed to meet the desired deadline
- Estimated number of sites and funding needed to recover a delayed plan
- Month-by-month progression of recruitment and spending

## How the model works

StudyRunway uses a deterministic expected-value model:

1. Sites activate gradually over the selected activation period.
2. Active sites screen participants at the selected monthly rate.
3. Screen failures are removed before enrollment.
4. Enrolled participants reach the completion endpoint after the selected follow-up period.
5. Expected dropout is applied before participants count toward the completed-participant target.
6. Recruitment stops when the enrolled pipeline is sufficient to reach the target.
7. Site startup, participant and fixed monthly program costs are accumulated through completion.

The **Meet the target** view uses the actual site-activation schedule to estimate the recruitment capacity required to reach the selected deadline.

## Example use case

The accompanying demonstration uses a fictional autism clinical-trial scenario. It shows how reducing recruitment from 3.0 to 1.5 screened participants per active site per month can move projected completion from month 18 to month 29 and create a potential funding gap.

The example is illustrative and is not based on any specific sponsor, medicine or clinical trial.

## Run locally

```bash
git clone https://github.com/sharonteo/clinical-trial-scenario-planner.git
cd clinical-trial-scenario-planner
pip install -r requirements.txt
streamlit run app.py
```

## Technology

- Python
- Streamlit
- pandas and NumPy
- Plotly

## Important limitations

StudyRunway is an early planning prototype, not a validated clinical-operations, financial or statistical forecasting system. It uses expected values rather than patient-level stochastic simulation and does not currently model site-specific performance, recruitment uncertainty, protocol amendments, geographic differences, adverse events, regulatory delays, manufacturing constraints or study closeout costs.

All default values are fictional. Replace them with validated inputs from clinical operations, finance, regulatory and manufacturing teams before using the model for internal planning.

## Disclaimer

This project is not medical, regulatory, financial or investment advice. It is not affiliated with any sponsor, medicine or clinical study.
