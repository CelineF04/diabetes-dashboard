# Diabetes: Risk, Reach, and Inequality — Dashboard

MSBA382 Healthcare Analytics consultant dashboard. Three tabs tell one story:

1. **How serious is diabetes?** — complication rates + age/sex-adjusted odds ratios
2. **Who is affected most?** — demographics, lifestyle, and a U.S. state map
3. **Why inequality matters?** — income, education, and healthcare-cost access barriers

## Files

- `app.py` — the Streamlit app
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — theme colors matching the chart palette
- `diabetes_2015_cleaned.csv`, `diabetes_2022_cleaned.csv` — **add these yourself**, exported by the
  companion Colab notebook (`Diabetes_ABC_Analysis.ipynb`, last cell). Place both files in the same
  folder as `app.py` before running or deploying.

If the CSVs aren't found next to `app.py`, the sidebar will show file-upload widgets instead, so the
app still works — but for a published dashboard, bundle the CSVs in the repo so reviewers don't have
to upload anything.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free, gives you the "published dashboard" link)

1. Create a new GitHub repo and push these files (including the two CSVs).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, pick the repo/branch, set the main file path to `app.py`, and deploy.
4. Copy the resulting `https://<your-app>.streamlit.app` URL for your Moodle submission.

## Notes for the report

- The two surveys define "diabetes" slightly differently (BRFSS2015 groups prediabetes in; BRFSS2022
  here is strict Yes/No) — mention this as a minor limitation.
- The Income/Education/SES coefficient model in Tab 3 is trained once on the **full, unfiltered**
  2015 dataset so it always reflects the overall structural relationship; the sidebar filters affect
  every other chart in Tabs 1–3.
- Plotly's `USA-states` choropleth scope can't render Puerto Rico/Guam/Virgin Islands geographically —
  they're surfaced instead in the "Top 10 by prevalence" table next to the map.
