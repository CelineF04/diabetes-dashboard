"""
Diabetes: Risk, Reach, and Inequality — Consultant Dashboard
MSBA382 Healthcare Analytics

Three tabs tell one story:
  A. How serious is diabetes?     -> complication rates + adjusted odds ratios
  B. Who is affected most?        -> demographics, lifestyle, geography
  C. Why does inequality matter?  -> income, education, access to care

Data:
  diabetes_2015_cleaned.csv  (BRFSS 2015 — SES/access fields: Income, Education, NoDocbcCost)
  diabetes_2022_cleaned.csv (BRFSS 2022 — complications, demographics, lifestyle, State)

Both files are produced by the companion Colab notebook (Diabetes_ABC_Analysis.ipynb)
and are expected to sit next to this app.py when deployed. If they aren't found,
the sidebar offers a manual upload so the app still runs.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Diabetes: Risk, Reach & Inequality",
    page_icon="🩺",
    layout="wide",
)

# ============================================================================
# THEME — same red/grey palette as the analysis notebook
# ============================================================================
PRIMARY = "#8B1E2D"     # deep red   -> diabetes / higher risk / higher burden
SECONDARY = "#4D4D4D"   # charcoal grey -> no diabetes / lower risk / neutral
LIGHT_GREY = "#BFBFBF"
BG = "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": SECONDARY, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlepad": 12,
    "axes.labelsize": 10, "axes.labelcolor": "#222222",
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "text.color": "#222222", "figure.dpi": 120,
})


def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def red_sequential(n):
    return sns.light_palette(PRIMARY, n_colors=n + 2)[2:]


def label_hbars(ax, fmt="{:.1f}%"):
    for c in ax.containers:
        ax.bar_label(c, labels=[fmt.format(v.get_width()) if v.get_width() else "" for v in c],
                      padding=3, fontsize=8, color="#222222")


def label_vbars(ax, fmt="{:.1f}%"):
    for c in ax.containers:
        ax.bar_label(c, labels=[fmt.format(v.get_height()) if v.get_height() else "" for v in c],
                      padding=3, fontsize=8, color="#222222")


def show(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ============================================================================
# DATA LOADING
# ============================================================================
INCOME_ORDER = ["<$10k", "$10-15k", "$15-20k", "$20-25k", "$25-35k", "$35-50k", "$50-75k", "$75k+"]
EDUC_ORDER = ["No school/Elem.", "Elementary", "Some HS", "HS Grad", "Some College", "College Grad"]

COMPLICATION_COLS = {
    "HadHeartAttack": "Heart Attack",
    "HadStroke": "Stroke",
    "HadKidneyDisease": "Kidney Disease",
    "DifficultyWalking": "Difficulty Walking",
}

US_STATE_TO_ABBREV = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO',
    'Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY',
    'Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN',
    'Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
    'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI',
    'South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT',
    'Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY',
    'Puerto Rico':'PR','Guam':'GU','Virgin Islands':'VI'
}


def age_category_to_midpoint(cat):
    nums = [int(n) for n in re.findall(r"\d+", str(cat))]
    if len(nums) == 2:
        return (nums[0] + nums[1]) / 2
    if len(nums) == 1:
        return nums[0] + 2.5
    return np.nan


@st.cache_data(show_spinner="Loading data...")
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)


@st.cache_data(show_spinner=False)
def prep_2022(df):
    df = df.copy()
    if "diab_bin" not in df.columns:
        df["diab_bin"] = (df["HadDiabetes"] == "Yes").astype(int)
    if "Sex_bin" not in df.columns:
        df["Sex_bin"] = (df["Sex"] == "Male").astype(int)
    if "Age_mid" not in df.columns:
        df["Age_mid"] = df["AgeCategory"].apply(age_category_to_midpoint)
    if "SleepGroup" not in df.columns and "SleepHours" in df.columns:
        df["SleepGroup"] = pd.cut(df["SleepHours"], bins=[0, 5, 7, 9, 24],
                                   labels=["<=5", "6-7", "8-9", "10+"], include_lowest=True, right=True)
    if "SmokerGrouped" not in df.columns and "SmokerStatus" in df.columns:
        df["SmokerGrouped"] = df["SmokerStatus"].isin([
            "Current smoker - now smokes every day",
            "Current smoker - now smokes some days",
            "Former smoker",
        ]).map({True: "Current/Former smoker", False: "Never smoker"})
    for col in COMPLICATION_COLS:
        bin_col = col + "_bin"
        if bin_col not in df.columns and col in df.columns:
            df[bin_col] = (df[col] == "Yes").astype(int)
    return df


@st.cache_data(show_spinner=False)
def prep_2015(df):
    df = df.copy()
    income_map = {1:'<$10k',2:'$10-15k',3:'$15-20k',4:'$20-25k',5:'$25-35k',6:'$35-50k',7:'$50-75k',8:'$75k+'}
    educ_map = {1:'No school/Elem.',2:'Elementary',3:'Some HS',4:'HS Grad',5:'Some College',6:'College Grad'}
    if "Income_lbl" not in df.columns:
        df["Income_lbl"] = df["Income"].map(income_map)
    if "Education_lbl" not in df.columns:
        df["Education_lbl"] = df["Education"].map(educ_map)
    if "Target_lbl" not in df.columns:
        df["Target_lbl"] = df["Diabetes_binary"].map({0: "No Diabetes", 1: "Diabetes/Prediabetes"})
    return df


def find_default(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, filename)
    return candidate if os.path.exists(candidate) else None


# ============================================================================
# SIDEBAR — data loading + filters
# ============================================================================
with st.sidebar:
    st.markdown("## 🩺 Diabetes Dashboard")
    st.caption("Healthcare Analytics — Consultant Dashboard")

    # Primary: GitHub Release asset URLs (auto-load for shared app link)
    url_2015 = "https://github.com/CelineF04/diabetes-dashboard/releases/download/v1.0/diabetes_2015_cleaned.csv"
    url_2022 = "https://github.com/CelineF04/diabetes-dashboard/releases/download/v1.0/diabetes_2022_cleaned.csv"

    df22_raw, df15_raw = None, None

    # Try remote URLs first
    try:
        df22_raw = load_csv(url_2022)
    except Exception:
        df22_raw = None

    try:
        df15_raw = load_csv(url_2015)
    except Exception:
        df15_raw = None

    # Validate required columns (prevents KeyError crash)
    required_22 = {"HadDiabetes", "Sex", "AgeCategory", "State"}
    required_15 = {"Diabetes_binary", "Income", "Education", "NoDocbcCost"}

    if df22_raw is not None:
        df22_raw.columns = df22_raw.columns.str.strip()
        if not required_22.issubset(set(df22_raw.columns)):
            df22_raw = None

    if df15_raw is not None:
        df15_raw.columns = df15_raw.columns.str.strip()
        if not required_15.issubset(set(df15_raw.columns)):
            df15_raw = None

    # Fallback: local files next to app.py
    if df22_raw is None:
        df22_path = find_default("diabetes_2022_cleaned.csv")
        if df22_path:
            try:
                df22_raw = load_csv(df22_path)
            except Exception:
                df22_raw = None

    if df15_raw is None:
        df15_path = find_default("diabetes_2015_cleaned.csv")
        if df15_path:
            try:
                df15_raw = load_csv(df15_path)
            except Exception:
                df15_raw = None

    # Final fallback: manual upload
    if df22_raw is None:
        up = st.file_uploader("Upload diabetes_2022_cleaned.csv", type=["csv"], key="up22")
        if up is not None:
            df22_raw = load_csv(up)

    if df15_raw is None:
        up = st.file_uploader("Upload diabetes_2015_cleaned.csv", type=["csv"], key="up15")
        if up is not None:
            df15_raw = load_csv(up)

    # Stop if still missing
    if df22_raw is None or df15_raw is None:
        st.warning("Upload both cleaned CSVs (from the Colab notebook) to load the dashboard.")
        st.stop()

    # Prep
    df22 = prep_2022(df22_raw)
    df15 = prep_2015(df15_raw)
  
    st.divider()
    st.markdown("### Filters — Tabs 1 & 2 (2022 data)")
    sex_opts = sorted(df22["Sex"].dropna().unique().tolist())
    sex_sel = st.multiselect("Sex", sex_opts, default=sex_opts)

    age_opts = sorted(
        df22["AgeCategory"].dropna().unique().tolist(),
        key=lambda c: df22.loc[df22["AgeCategory"] == c, "Age_mid"].iloc[0]
    )
    age_sel = st.multiselect("Age category", age_opts, default=age_opts)

    st.divider()
    st.markdown("### Filters — Tab 3 (2015 SES data)")
    inc_sel = st.multiselect("Income level", [x for x in INCOME_ORDER if x in df15["Income_lbl"].unique()],
                              default=[x for x in INCOME_ORDER if x in df15["Income_lbl"].unique()])
    edu_sel = st.multiselect("Education level", [x for x in EDUC_ORDER if x in df15["Education_lbl"].unique()],
                              default=[x for x in EDUC_ORDER if x in df15["Education_lbl"].unique()])

    st.divider()
    with st.expander("About this dashboard"):
        st.markdown(
            "Built on two CDC BRFSS surveys (2015 and 2022). The 2022 file carries "
            "complication outcomes, demographics, lifestyle and state; the 2015 file "
            "carries income, education, and healthcare-cost-barrier fields. "
            "Diabetes definitions differ slightly between the two surveys (see report)."
        )

# Apply filters
df22_f = df22[df22["Sex"].isin(sex_sel) & df22["AgeCategory"].isin(age_sel)]
df15_f = df15[df15["Income_lbl"].isin(inc_sel) & df15["Education_lbl"].isin(edu_sel)]

if df22_f.empty:
    st.warning("No 2022 rows match the current filters — showing the unfiltered dataset instead.")
    df22_f = df22
if df15_f.empty:
    st.warning("No 2015 rows match the current filters — showing the unfiltered dataset instead.")
    df15_f = df15

# ============================================================================
# HEADER + KPI ROW
# ============================================================================
st.title("Diabetes: Risk, Reach, and Inequality")
st.caption("How serious is diabetes, who does it affect most, and why? — a three-part consultant brief")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Respondents (2022 ds, filtered)", f"{len(df22_f):,}")
k2.metric("Diabetes rate (2022 ds, filtered)", f"{df22_f['diab_bin'].mean()*100:.1f}%")
k3.metric("Respondents (2015 ds, filtered)", f"{len(df15_f):,}")
k4.metric("Diabetes rate (2015 ds, filtered)", f"{df15_f['Diabetes_binary'].mean()*100:.1f}%")

tab1, tab2, tab3 = st.tabs([
    "How serious is diabetes?",
    "Who is affected most?",
    "Why inequality matters",
])

# ============================================================================
# TAB 1 — PART A: EFFECT & RISK
# ============================================================================
with tab1:
    st.subheader("Diabetes is not trivial — it's strongly associated with serious outcomes")
    st.caption("BRFSS 2022 · complications compared by diabetes status, adjusted for age and sex")

    c1, c2 = st.columns(2)

    # --- Chart A1: unadjusted complication rates ---
    with c1:
        rows = []
        for col, label in COMPLICATION_COLS.items():
            bin_col = col + "_bin"
            if bin_col not in df22_f.columns:
                continue
            g = df22_f.groupby("diab_bin")[bin_col].mean() * 100
            rows.append({"Complication": label, "No Diabetes": g.get(0, 0), "Diabetes": g.get(1, 0)})
        comp_df = pd.DataFrame(rows).sort_values("Diabetes", ascending=False)

        x = np.arange(len(comp_df))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(x - width/2, comp_df["No Diabetes"], width, color=SECONDARY, label="No Diabetes")
        ax.bar(x + width/2, comp_df["Diabetes"], width, color=PRIMARY, label="Diabetes")
        label_vbars(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(comp_df["Complication"], rotation=15, ha="right")
        style_ax(ax, "Complication rates: diabetics vs. non-diabetics", "", "Rate (%)")
        ax.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        show(fig)

    # --- Chart A2: adjusted odds ratios ---
    with c2:
        @st.cache_data(show_spinner=False)
        def fit_adjusted_model(df, outcome_bin_col, random_state=42):
            feats = ["diab_bin", "Sex_bin", "Age_mid"]
            sub = df[feats + [outcome_bin_col]].dropna()
            if sub[outcome_bin_col].nunique() < 2 or len(sub) < 50:
                return None
            X, y = sub[feats], sub[outcome_bin_col]
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=random_state, stratify=y
                )
            except ValueError:
                return None
            scaler = StandardScaler()
            X_train_s, X_test_s = X_train.copy(), X_test.copy()
            X_train_s[["Age_mid"]] = scaler.fit_transform(X_train[["Age_mid"]])
            X_test_s[["Age_mid"]] = scaler.transform(X_test[["Age_mid"]])
            model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
            model.fit(X_train_s, y_train)
            auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1]) if y_test.nunique() > 1 else np.nan
            diab_coef = model.coef_[0][feats.index("diab_bin")]
            return float(np.exp(diab_coef)), float(auc), len(sub)

        results = []
        for col, label in COMPLICATION_COLS.items():
            bin_col = col + "_bin"
            if bin_col not in df22_f.columns:
                continue
            out = fit_adjusted_model(df22_f, bin_col)
            if out is not None:
                odds_ratio, auc, n = out
                results.append({"Complication": label, "Adjusted Odds Ratio": odds_ratio, "Test AUC": auc, "n": n})

        if results:
            results_df = pd.DataFrame(results).sort_values("Adjusted Odds Ratio", ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            colors = [PRIMARY if v > 1 else SECONDARY for v in results_df["Adjusted Odds Ratio"]]
            ax.barh(results_df["Complication"], results_df["Adjusted Odds Ratio"], color=colors)
            ax.axvline(1, color=LIGHT_GREY, linewidth=1.5, linestyle="--")
            for i, v in enumerate(results_df["Adjusted Odds Ratio"]):
                ax.text(v + 0.05, i, f"{v:.2f}x", va="center", fontsize=9, fontweight="bold")
            style_ax(ax, "Adjusted odds ratio (age & sex controlled)", "Odds ratio vs. no diabetes", "")
            plt.tight_layout()
            show(fig)
        else:
            st.info("Not enough filtered data to fit the adjusted models — widen the filters.")

    st.info(
        "💡 **Insight:** diabetics report major complications at multiples of the rate of non-diabetics, "
        "and the gap survives adjustment for age and sex — diabetes carries real downstream risk, which "
        "is the case for why prevalence and inequality (next two tabs) matter."
    )

# ============================================================================
# TAB 2 — PART B: WHO IS AFFECTED MOST?
# ============================================================================
with tab2:
    st.subheader("Diabetes burden is uneven across people, behaviors, and places")
    st.caption("BRFSS 2022 · demographic and lifestyle prevalence, plus geography")

    st.markdown("##### Demographic prevalence")
    d1, d2, d3 = st.columns(3)

    with d1:
        sex_tbl = df22_f.groupby("Sex")["diab_bin"].mean().mul(100).sort_values(ascending=False)
        colors = [PRIMARY if v == sex_tbl.max() else SECONDARY for v in sex_tbl.values]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.barh(sex_tbl.index, sex_tbl.values, color=colors)
        label_hbars(ax)
        style_ax(ax, "By Sex", "Prevalence (%)", "")
        plt.tight_layout()
        show(fig)

    with d2:
        race_tbl = df22_f.groupby("RaceEthnicityCategory")["diab_bin"].mean().mul(100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.barh(race_tbl.index[::-1], race_tbl.values[::-1], color=red_sequential(len(race_tbl))[::-1])
        label_hbars(ax)
        ax.tick_params(axis="y", labelsize=7)
        style_ax(ax, "By Race / Ethnicity", "Prevalence (%)", "")
        plt.tight_layout()
        show(fig)

    with d3:
        age_tbl = df22_f.groupby("AgeCategory")["diab_bin"].mean().mul(100)
        order = sorted(age_tbl.index, key=lambda c: df22_f.loc[df22_f["AgeCategory"] == c, "Age_mid"].iloc[0])
        age_tbl = age_tbl.reindex(order)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.barh(age_tbl.index, age_tbl.values, color=red_sequential(len(age_tbl)))
        label_hbars(ax)
        ax.tick_params(axis="y", labelsize=7)
        style_ax(ax, "By Age Category", "Prevalence (%)", "")
        plt.tight_layout()
        show(fig)

    st.markdown("##### Lifestyle prevalence")
    l1, l2, l3 = st.columns(3)

    with l1:
        pa_tbl = df22_f.groupby("PhysicalActivities")["diab_bin"].mean().mul(100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.bar(pa_tbl.index, pa_tbl.values, color=[PRIMARY, SECONDARY])
        label_vbars(ax)
        style_ax(ax, "By Physical Activity", "", "Prevalence (%)")
        plt.tight_layout()
        show(fig)

    with l2:
        sleep_order = ["<=5", "6-7", "8-9", "10+"]
        sleep_tbl = df22_f.groupby("SleepGroup")["diab_bin"].mean().reindex(sleep_order).mul(100)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.bar(sleep_tbl.index, sleep_tbl.values, color=red_sequential(len(sleep_tbl)))
        label_vbars(ax)
        style_ax(ax, "By Sleep Duration", "Hours/night", "Prevalence (%)")
        plt.tight_layout()
        show(fig)

    with l3:
        smk_tbl = df22_f.groupby("SmokerGrouped")["diab_bin"].mean().mul(100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.bar(smk_tbl.index, smk_tbl.values, color=[PRIMARY, SECONDARY])
        label_vbars(ax)
        ax.tick_params(axis="x", labelsize=7)
        style_ax(ax, "By Smoking Status", "", "Prevalence (%)")
        plt.tight_layout()
        show(fig)

    st.markdown("##### Geography")
    g1, g2 = st.columns([2, 1])

    state_prev = df22_f.groupby("State", as_index=False)["diab_bin"].mean()
    state_prev["Diabetes_pct"] = state_prev["diab_bin"] * 100
    state_prev["code"] = state_prev["State"].map(US_STATE_TO_ABBREV)
    mapable = state_prev.dropna(subset=["code"])
    mapable = mapable[~mapable["code"].isin(["PR", "GU", "VI"])]

    with g1:
        fig = px.choropleth(
            mapable, locations="code", locationmode="USA-states", scope="usa",
            color="Diabetes_pct", color_continuous_scale=["#F5E6E6", PRIMARY],
            hover_name="State", labels={"Diabetes_pct": "Diabetes prevalence (%)"},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420)
        st.plotly_chart(fig, width="stretch")

    with g2:
        st.markdown("**Top 10 by prevalence**")
        top10 = state_prev.sort_values("Diabetes_pct", ascending=False)[["State", "Diabetes_pct"]].head(10)
        top10["Diabetes_pct"] = top10["Diabetes_pct"].round(1)
        st.dataframe(top10.rename(columns={"Diabetes_pct": "Diabetes %"}), hide_index=True, width="stretch")
        st.caption(
            "Plotly's USA-states map can't render territories, but Puerto Rico "
            "typically tops this ranking — which motivates the next tab."
        )

    st.info(
        "💡 **Insight:** prevalence climbs with age, varies by sex/race, worsens with inactivity, short sleep, "
        "and smoking, and is highest in several lower-income states/territories — pointing toward structural, "
        "not just individual, drivers."
    )

# ============================================================================
# TAB 3 — PART C: SES & ACCESS
# ============================================================================
with tab3:
    st.subheader("SES and access barriers are key structural correlates of diabetes burden")
    st.caption("BRFSS 2015 · income, education, and cost-related access to care")

    c1, c2 = st.columns(2)

    with c1:
        inc_tbl = df15_f.groupby("Income_lbl")["Diabetes_binary"].mean().reindex(INCOME_ORDER).mul(100).dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(inc_tbl.index, inc_tbl.values, color=red_sequential(len(inc_tbl))[::-1])
        label_vbars(ax)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        style_ax(ax, "Prevalence by income level", "", "Prevalence (%)")
        plt.tight_layout()
        show(fig)

    with c2:
        edu_tbl = df15_f.groupby("Education_lbl")["Diabetes_binary"].mean().reindex(EDUC_ORDER).mul(100).dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(edu_tbl.index, edu_tbl.values, color=red_sequential(len(edu_tbl))[::-1])
        label_vbars(ax)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        style_ax(ax, "Prevalence by education level", "", "Prevalence (%)")
        plt.tight_layout()
        show(fig)

    c3, c4 = st.columns(2)

    with c3:
        pivot_ie = df15_f.pivot_table(values="Diabetes_binary", index="Education_lbl",
                                       columns="Income_lbl", aggfunc="mean").reindex(
            index=EDUC_ORDER, columns=INCOME_ORDER) * 100
        red_cmap = sns.light_palette(PRIMARY, as_cmap=True)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(pivot_ie, annot=True, fmt=".1f", cmap=red_cmap, cbar_kws={"label": "Diabetes rate (%)"},
                    linewidths=0.5, linecolor="white", ax=ax, annot_kws={"color": "#222222", "size": 7})
        style_ax(ax, "Income × Education heatmap", "Income", "Education")
        plt.xticks(rotation=35, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        show(fig)

    with c4:
        barrier = df15_f.groupby(["Income_lbl", "Target_lbl"])["NoDocbcCost"].mean().mul(100).unstack().reindex(INCOME_ORDER).dropna(how="all")
        x = np.arange(len(barrier))
        width = 0.38
        fig, ax = plt.subplots(figsize=(6, 4.5))
        if "No Diabetes" in barrier.columns:
            ax.bar(x - width/2, barrier["No Diabetes"], width, color=SECONDARY, label="No Diabetes")
        if "Diabetes/Prediabetes" in barrier.columns:
            ax.bar(x + width/2, barrier["Diabetes/Prediabetes"], width, color=PRIMARY, label="Diabetes/Prediabetes")
        ax.set_xticks(x)
        ax.set_xticklabels(barrier.index, rotation=30, ha="right", fontsize=8)
        style_ax(ax, "Cost barrier by income & diabetes status", "", "Skipped doctor (cost), %")
        ax.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        show(fig)

    st.markdown("##### Structural drivers in the full risk model")
    st.caption("Trained once on the full 2015 dataset (not affected by sidebar filters) — shows overall structural effects")

    @st.cache_data(show_spinner="Fitting model...")
    def fit_full_model(df):
        X = df.drop(columns=["Diabetes_binary", "Income_lbl", "Education_lbl", "Target_lbl"], errors="ignore")
        y = df["Diabetes_binary"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        cont_features = [c for c in ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income"] if c in X.columns]
        X_train_s, X_test_s = X_train.copy(), X_test.copy()
        X_train_s[cont_features] = scaler.fit_transform(X_train[cont_features])
        X_test_s[cont_features] = scaler.transform(X_test[cont_features])
        logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        logreg.fit(X_train_s, y_train)
        coef = pd.Series(logreg.coef_[0], index=X.columns)
        return coef

    coef = fit_full_model(df15)
    ses_vars = [v for v in ["Income", "Education", "NoDocbcCost", "AnyHealthcare"] if v in coef.index]
    ses_coef = coef[ses_vars].sort_values()

    fig, ax = plt.subplots(figsize=(7, 3))
    colors = [PRIMARY if v > 0 else SECONDARY for v in ses_coef.values]
    ax.barh(ses_coef.index, ses_coef.values, color=colors)
    ax.axvline(0, color=LIGHT_GREY, linewidth=1)
    style_ax(ax, "Socioeconomic & access factors: effect on diabetes risk", "Standardized coefficient", "")
    plt.tight_layout()
    show(fig)

    st.info(
        "💡 **Insight:** diabetes prevalence falls steadily as income and education rise, the two compound "
        "at the bottom of the grid, and low-income diabetics face the steepest cost barriers to care — "
        "the people most at risk are often the least able to afford treatment."
    )

st.divider()
st.caption(
    "Data: CDC BRFSS 2015 & 2022 (Kaggle). Built with Streamlit, scikit-learn, Plotly. "
    "MSBA382 Healthcare Analytics — Consultant Dashboard."
)
