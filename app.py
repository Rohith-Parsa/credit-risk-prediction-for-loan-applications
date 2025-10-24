# app.py
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple

# ===================== Page Config & Theme =====================
st.set_page_config(page_title="RiskScore | Credit Risk Analytics", page_icon="💳", layout="wide")
px.defaults.template = "plotly_white"

st.markdown("""
<style>
.block-container {padding-top: 0.8rem; padding-bottom: 0.5rem; max-width: 1400px;}
h1,h2,h3{font-weight:800;margin-bottom:.35rem}
small, .caption{color:#6b7280}
[data-testid="stSidebar"] > div:first-child {background: linear-gradient(180deg,#f0f4ff 0%,#ffffff 100%); border-right:1px solid #e5e7eb;}
.hero {background: radial-gradient(1200px 400px at 15% -60%, #dbeafe 0%, transparent 60%) , linear-gradient(180deg,#ffffff 0%,#f9fafb 100%); border:1px solid #e5e7eb; border-radius:16px; padding:20px 24px; box-shadow:0 4px 14px rgba(0,0,0,.06); margin-bottom:1.5rem;}
.hero .title {font-size:36px; font-weight:900; margin-bottom:8px; background: linear-gradient(135deg, #1e40af 0%, #4f46e5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero .sub {color:#64748b; font-size:15px; line-height:1.5}
.kpi {background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,0.04); transition: transform 0.2s;}
.kpi:hover {transform: translateY(-2px); box-shadow:0 4px 12px rgba(0,0,0,0.08);}
.kpi h3{margin:0 0 8px 0;font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;font-weight:600}
.kpi .val{font-size:30px;font-weight:800;background: linear-gradient(135deg, #1e40af 0%, #4f46e5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.card {background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05); margin-bottom:1rem}
.section-title{font-size:20px;font-weight:800;margin:1.5rem 0 1rem 0; color:#1e293b}
.stButton>button {background: linear-gradient(90deg,#2563eb,#4f46e5); color:white;border:0;border-radius:10px;padding:.6rem 1.2rem;font-weight:700;font-size:15px; box-shadow:0 4px 10px rgba(37,99,235,0.3); transition: all 0.2s;}
.stButton>button:hover{filter:brightness(1.1); transform: translateY(-1px); box-shadow:0 6px 14px rgba(37,99,235,0.4)}
.tile {background:linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);border:1px solid #e5e7eb;border-radius:12px;padding:14px;text-align:center; margin-bottom:10px}
.tile .name{font-size:12px;color:#64748b;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;font-weight:600}
.tile .p{font-size:24px;font-weight:800}
.gauge {background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;padding:12px; box-shadow:0 2px 6px rgba(0,0,0,0.04)}
.stTabs [data-baseweb="tab-list"] {gap: 8px; border-bottom: 2px solid #e5e7eb}
.stTabs [data-baseweb="tab"] {background: #f1f5f9; border-radius: 10px 10px 0px 0px; padding: 10px 16px; font-weight:600}
.stTabs [aria-selected="true"] {background: linear-gradient(180deg, #e0e7ff 0%, #dbeafe 100%); color:#1e40af}
.metric-card {background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); border:1px solid #e5e7eb; border-radius:12px; padding:16px; box-shadow:0 2px 6px rgba(0,0,0,0.04)}
.comparison-box {background:#f8fafc; border-left:4px solid #3b82f6; padding:12px; border-radius:8px; margin:8px 0}
</style>
""", unsafe_allow_html=True)

# ===================== Helpers & Caching =====================
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    try:
        df = pd.read_csv("credit.csv")
        if 'SeriousDlqin2yrs' in df.columns:
            df['SeriousDlqin2yrs'] = df['SeriousDlqin2yrs'].astype(int)
        return df
    except FileNotFoundError:
        st.error("❌ **Error**: credit.csv file not found in the current directory.")
        st.info("Please ensure 'credit.csv' is in the same folder as app.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Error loading dataset**: {str(e)}")
        st.stop()

@st.cache_data(show_spinner=False)
def filter_by_age(df: pd.DataFrame, age_min: int, age_max: int) -> pd.DataFrame:
    return df[(df['age'] >= age_min) & (df['age'] <= age_max)].copy() if 'age' in df.columns else df.copy()

@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        models = {k: joblib.load(f"{v}_model.pkl") for k, v in [('rf', 'RandomForest'), ('xgb', 'XGBoost'), ('lgb', 'LightGBM')]}
        models['months'] = joblib.load("xgb_months_regressor_realistic.pkl")
    except FileNotFoundError as e:
        st.error(f"❌ **Model file not found**: {str(e)}")
        st.info("Please ensure all .pkl model files are in the same directory as app.py")
        st.stop()
    
    params = joblib.load("preprocessing_params.pkl") if joblib else {}
    feat_order = joblib.load("feature_order.pkl") if joblib else None
    
    try:
        with open("model_results_simple.json","r") as f:
            train_summary = json.load(f)
    except:
        train_summary = {}
    
    return models, params, feat_order, train_summary

def apply_preproc(df_in: pd.DataFrame, p: dict) -> pd.DataFrame:
    d = df_in.copy()
    d['Income_missing'] = d['MonthlyIncome'].isna().astype(int) if 'MonthlyIncome' in d.columns else 0
    d['Deps_missing'] = d['NumberOfDependents'].isna().astype(int) if 'NumberOfDependents' in d.columns else 0
    
    if 'MonthlyIncome' in d.columns and 'age' in d.columns:
        age_binned = pd.cut(d['age'].clip(0,120), bins=p.get('age_bins', [0,25,35,45,55,65,120]), include_lowest=True)
        income_map = p.get('income_median_by_age', {})
        for idx in d.index:
            if pd.isna(d.at[idx, 'MonthlyIncome']):
                d.at[idx, 'MonthlyIncome'] = income_map.get(age_binned.iloc[idx], p.get('global_income_median', 5400))
    
    if 'NumberOfDependents' in d.columns:
        d['NumberOfDependents'] = d['NumberOfDependents'].fillna(p.get('dependents_median', 0)).round().clip(0, p.get('dependents_max', 8)).astype(int)
    
    for col, cap_key in [('RevolvingUtilizationOfUnsecuredLines', 'util_cap'), ('DebtRatio', 'debt_cap')]:
        if col in d.columns and p.get(cap_key):
            d[col] = d[col].clip(0, p[cap_key])
    
    if 'MonthlyIncome' in d.columns:
        d['MonthlyIncome'] = np.log1p(d['MonthlyIncome'])
    
    return d

def enforce_order(dfin: pd.DataFrame, order: List[str] | None) -> pd.DataFrame:
    if order is None:
        base = ['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans',
                'NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfDependents','Income_missing','Deps_missing']
        cols = [c for c in base if c in dfin.columns] + [c for c in dfin.columns if c not in base]
        d = dfin.reindex(columns=cols)
        for c in cols:
            if c not in d.columns: d[c] = 0
        return d
    d = dfin.copy()
    for c in order:
        if c not in d.columns: d[c] = 0
    return d[order]

def gauge(prob: float, title: str="Default Probability"):
    thr, clr = 0.5, "#ef4444" if prob >= 0.5 else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob*100, number={'suffix':" %", 'font': {'size': 40, 'weight': 'bold'}},
        title={'text': title, 'font': {'size': 18, 'weight': 'bold'}},
        gauge={'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#94a3b8'}, 'bar': {'color': clr, 'thickness': 0.8},
               'bgcolor': 'white', 'borderwidth': 2, 'bordercolor': '#e2e8f0',
               'steps': [{'range': [0, 50], 'color': '#e8f5e9'},{'range': [50, 100], 'color': '#ffdde0'}],
               'threshold': {'line': {'color': '#3b82f6', 'width': 4}, 'thickness': 0.75, 'value': 50}}
    ))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def risk_band(p: float) -> Tuple[str, str]:
    if p < 0.125: return "Very Low", "#16a34a"
    if p < 0.375: return "Low", "#22c55e"
    if p < 0.5: return "Moderate", "#f59e0b"
    if p < 0.75: return "High", "#ef4444"
    return "Very High", "#b91c1c"

def predict_model(proc: pd.DataFrame, model_key: str, models: Dict) -> float:
    return float(models[model_key].predict_proba(proc)[0,1]) if model_key in models else np.nan

# ===================== Navigation =====================
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio("Go to", ["🏠 Home", "📁 Dataset", "📈 Dashboard", "🧠 Predictor"], label_visibility="collapsed")

# ===================== Data & Models =====================
df = load_dataset()
models, params, feature_order, train_summary = load_artifacts()

# ===================== Pages =====================
if section == "🏠 Home":
    st.markdown('<div class="hero"><div class="title">Credit Risk Prediction</div><div class="sub">ML-powered credit scoring with portfolio health analytics and calibrated probability predictions for informed lending decisions.</div></div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi"><h3>Total Records</h3><div class="val">{len(df):,}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi"><h3>Default Rate</h3><div class="val">{df["SeriousDlqin2yrs"].mean()*100:.2f}%</div></div>', unsafe_allow_html=True)
    with c3:
        inc = df['MonthlyIncome'].median()
        st.markdown(f'<div class="kpi"><h3>Median Income</h3><div class="val">${inc:,.0f}</div></div>' if pd.notna(inc) else '<div class="kpi"><h3>Median Income</h3><div class="val">N/A</div></div>', unsafe_allow_html=True)
    with c4:
        util = df['RevolvingUtilizationOfUnsecuredLines'].median()
        st.markdown(f'<div class="kpi"><h3>Median Utilization</h3><div class="val">{util:.1%}</div></div>' if pd.notna(util) else '<div class="kpi"><h3>Median Utilization</h3><div class="val">N/A</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Performance Summary</div>', unsafe_allow_html=True)
    if train_summary and 'classification' in train_summary:
        cols = st.columns(3)
        for col, mname in zip(cols, ['RandomForest', 'XGBoost', 'LightGBM']):
            if mname in train_summary['classification']:
                m = train_summary['classification'][mname]
                with col:
                    st.markdown(f'<div class="metric-card"><h3 style="margin:0 0 10px 0; font-size:16px; color:#1e40af">{mname}</h3>'
                               f'<div style="font-size:13px; color:#64748b; margin:4px 0">ROC-AUC: <b>{m.get("roc_auc", 0):.3f}</b></div>'
                               f'<div style="font-size:13px; color:#64748b; margin:4px 0">PR-AUC: <b>{m.get("pr_auc", 0):.3f}</b></div>'
                               f'<div style="font-size:13px; color:#64748b; margin:4px 0">Accuracy: <b>{m.get("accuracy", 0):.3f}</b></div></div>', unsafe_allow_html=True)

elif section == "📁 Dataset":
    st.title("📁 Dataset Explorer")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📊 Data Preview", "📈 Statistics"])
    
    with tab1:
        st.markdown("### Sample Records")
        st.dataframe(df.head(50), use_container_width=True, height=400)
        st.caption(f"**Dataset Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    with tab2:
        st.markdown("### Descriptive Statistics")
        st.dataframe(df.describe(include='all').T.style.format("{:.2f}"), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif section == "📈 Dashboard":
    st.title("📈 Executive Dashboard")
    
    if 'age' in df.columns:
        st.markdown("### 🎯 Interactive Filters")
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            a_min, a_max = int(df['age'].min()), int(df['age'].max())
            age_min, age_max = st.slider("Age Range", a_min, a_max, (18, min(a_max,65)), 1, key="dash_age_filter")
        with col_f2:
            st.metric("Filtered Records", f"{len(filter_by_age(df, age_min, age_max)):,}")
        dfd = filter_by_age(df, age_min, age_max)
    else:
        dfd = df.copy()

    st.markdown("---")
    st.markdown("### 📊 Key Performance Indicators")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.markdown(f'<div class="kpi"><h3>Records</h3><div class="val">{len(dfd):,}</div></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="kpi"><h3>Default Rate</h3><div class="val">{dfd["SeriousDlqin2yrs"].mean()*100:.2f}%</div></div>', unsafe_allow_html=True)
    with k3:
        inc = dfd['MonthlyIncome'].median()
        st.markdown(f'<div class="kpi"><h3>Median Income</h3><div class="val">${inc:,.0f}</div></div>' if pd.notna(inc) else '<div class="kpi"><h3>Median Income</h3><div class="val">N/A</div></div>', unsafe_allow_html=True)
    with k4:
        util = dfd['RevolvingUtilizationOfUnsecuredLines'].median()
        st.markdown(f'<div class="kpi"><h3>Avg Utilization</h3><div class="val">{util:.1%}</div></div>' if pd.notna(util) else '<div class="kpi"><h3>Avg Utilization</h3><div class="val">N/A</div></div>', unsafe_allow_html=True)
    with k5:
        debt = dfd['DebtRatio'].median()
        st.markdown(f'<div class="kpi"><h3>Median Debt Ratio</h3><div class="val">{debt:.2f}</div></div>' if pd.notna(debt) else '<div class="kpi"><h3>Median Debt Ratio</h3><div class="val">N/A</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📉 Portfolio Risk Analysis")
    r1c1, r1c2, r1c3 = st.columns(3)
    
    with r1c1:
        tgt = dfd['SeriousDlqin2yrs'].value_counts(normalize=True).rename({0:'Non-Default',1:'Default'}).reset_index()
        tgt.columns = ['Status','Share']
        fig = go.Figure(data=[go.Pie(labels=tgt['Status'], values=tgt['Share'], hole=.6, marker=dict(colors=['#22c55e', '#ef4444']), textfont=dict(size=15, weight='bold'))])
        fig.update_traces(textposition='inside', texttemplate='%{label}<br><b>%{percent:.1%}</b>')
        fig.update_layout(title="<b>Default Distribution</b>", showlegend=True, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with r1c2:
        ab = pd.cut(dfd['age'], bins=[0,25,35,45,55,65,120], labels=['≤25','26-35','36-45','46-55','56-65','66+'], include_lowest=True)
        agg = pd.DataFrame({'age_band':ab, 'default':dfd['SeriousDlqin2yrs']}).groupby('age_band', observed=True)['default'].mean().reset_index()
        bar = px.bar(agg, x='age_band', y='default', title="<b>Default Rate by Age Cohort</b>", labels={'age_band': 'Age Band', 'default': 'Default Rate'}, color='default', color_continuous_scale='Reds')
        bar.update_traces(texttemplate='%{y:.1%}', textposition='outside')
        bar.update_layout(yaxis_tickformat='.0%', showlegend=False, height=350)
        st.plotly_chart(bar, use_container_width=True)
    
    with r1c3:
        late_bins = pd.cut(dfd['NumberOfTimes90DaysLate'], bins=[-0.1,0,1,2,5,100], labels=['0','1','2','3-5','6+'])
        t = pd.DataFrame({'late_band':late_bins, 'default':dfd['SeriousDlqin2yrs']}).groupby('late_band', observed=True)['default'].mean().reset_index()
        area = px.area(t, x='late_band', y='default', title="<b>Default by Late Payments</b>", labels={'late_band': '90+ Days Late Count', 'default': 'Default Rate'})
        area.update_traces(line=dict(color='#ef4444', width=3), fillcolor='rgba(239,68,68,0.2)')
        area.update_layout(yaxis_tickformat='.0%', height=350)
        st.plotly_chart(area, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💰 Financial Metrics Deep Dive")
    r2c1, r2c2, r2c3 = st.columns(3)
    
    with r2c1:
        bubble = px.scatter(dfd.sample(min(len(dfd), 6000), random_state=12), x='DebtRatio', y='MonthlyIncome', size='NumberOfOpenCreditLinesAndLoans', color='SeriousDlqin2yrs', opacity=0.5, title="<b>Debt Ratio vs Income</b>", color_discrete_map={0: '#22c55e', 1: '#ef4444'}, labels={'SeriousDlqin2yrs': 'Default'})
        bubble.update_layout(height=350)
        st.plotly_chart(bubble, use_container_width=True)
    
    with r2c2:
        dec = pd.qcut(dfd['RevolvingUtilizationOfUnsecuredLines'], q=10, duplicates='drop').astype(str)
        agg = pd.DataFrame({'util_decile':dec, 'default': dfd['SeriousDlqin2yrs']}).groupby('util_decile', observed=True)['default'].mean().reset_index()
        line_util = px.line(agg, x='util_decile', y='default', markers=True, title="<b>Default by Utilization Decile</b>", labels={'util_decile': 'Utilization Decile', 'default': 'Default Rate'})
        line_util.update_traces(line=dict(color='#8b5cf6', width=3), marker=dict(size=10))
        line_util.update_layout(yaxis_tickformat='.0%', height=350)
        st.plotly_chart(line_util, use_container_width=True)
    
    with r2c3:
        debt_bins = pd.cut(dfd['DebtRatio'], bins=[0,0.3,0.6,1.0,2.0,100], labels=['Low','Medium','High','V.High','Extreme'])
        t = pd.DataFrame({'debt_cat':debt_bins, 'default':dfd['SeriousDlqin2yrs']}).groupby('debt_cat', observed=True)['default'].mean().reset_index()
        bar_debt = px.bar(t, x='debt_cat', y='default', title="<b>Default by Debt Category</b>", labels={'debt_cat': 'Debt Ratio Category', 'default': 'Default Rate'}, color='default', color_continuous_scale='Oranges')
        bar_debt.update_traces(texttemplate='%{y:.1%}', textposition='outside')
        bar_debt.update_layout(yaxis_tickformat='.0%', showlegend=False, height=350)
        st.plotly_chart(bar_debt, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Distribution Analysis")
    r3c1, r3c2, r3c3 = st.columns(3)
    
    with r3c1:
        box_income = px.box(dfd, y='MonthlyIncome', color='SeriousDlqin2yrs', title="<b>Income Distribution by Default</b>", color_discrete_map={0: '#22c55e', 1: '#ef4444'}, labels={'SeriousDlqin2yrs': 'Default'})
        box_income.update_layout(showlegend=True, height=350)
        st.plotly_chart(box_income, use_container_width=True)
    
    with r3c2:
        hist_debt = px.histogram(dfd, x='DebtRatio', nbins=60, title="<b>Debt Ratio Distribution</b>", labels={'DebtRatio': 'Debt Ratio'})
        hist_debt.update_traces(marker_color='#4f46e5')
        hist_debt.update_layout(height=350)
        st.plotly_chart(hist_debt, use_container_width=True)
    
    with r3c3:
        violin_util = px.violin(dfd.sample(min(len(dfd), 8000), random_state=7), y='RevolvingUtilizationOfUnsecuredLines', box=True, points=False, title="<b>Credit Utilization Distribution</b>", labels={'RevolvingUtilizationOfUnsecuredLines': 'Utilization'})
        violin_util.update_traces(fillcolor='#f59e0b', line_color='#d97706')
        violin_util.update_layout(height=350)
        st.plotly_chart(violin_util, use_container_width=True)

elif section == "🧠 Predictor":
    st.title("🧠 Credit Risk Scoring")
    st.markdown("---")
    
    model_choice = st.selectbox("Select Prediction Model", ["Random Forest", "XGBoost", "LightGBM"], index=0)
    sel_key = {"Random Forest": "rf", "XGBoost": "xgb", "LightGBM": "lgb"}[model_choice]

    st.markdown("### 📝 Applicant Information")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown("**Credit Utilization**")
        RevolvingUtilizationOfUnsecuredLines = st.number_input("Revolving Utilization", 0.0, 1.0, 0.10, 0.01, help="Ratio of revolving debt to credit limit (0-1)")
        NumberOfOpenCreditLinesAndLoans = st.number_input("Open Credit Lines & Loans", 0, 60, 5, 1, help="Total number of active credit accounts")
    
    with c2:
        st.markdown("**Personal Information**")
        age = st.number_input("Age (years)", 18, 100, 35, 1)
        NumberOfDependents = st.number_input("Dependents", 0, 20, 0, 1, help="Excluding self")
    
    with c3:
        st.markdown("**Financial Position**")
        DebtRatio = st.number_input("Debt Ratio", 0.0, 5.0, 0.30, 0.01, help="Monthly debt payments / gross income")
        MonthlyIncome = st.number_input("Monthly Income ($)", 0.0, 1_000_000.0, 5000.0, 100.0)
    
    with c4:
        st.markdown("**Credit History**")
        NumberOfTimes90DaysLate = st.number_input("Times 90+ Days Late", 0, 50, 0, 1, help="In the past 2 years")
        NumberRealEstateLoansOrLines = st.number_input("Real Estate Loans/Lines", 0, 20, 1, 1, help="Mortgage and home equity lines")

    util_val = float(np.clip(RevolvingUtilizationOfUnsecuredLines, 0.0, 1.0))
    if RevolvingUtilizationOfUnsecuredLines != util_val:
        st.warning("⚠️ Revolving Utilization clamped to valid range [0, 1]", icon="⚠️")

    raw = pd.DataFrame({'RevolvingUtilizationOfUnsecuredLines': [util_val], 'age': [age], 'DebtRatio': [DebtRatio], 'MonthlyIncome': [MonthlyIncome],
                        'NumberOfOpenCreditLinesAndLoans': [NumberOfOpenCreditLinesAndLoans], 'NumberOfTimes90DaysLate': [NumberOfTimes90DaysLate],
                        'NumberRealEstateLoansOrLines': [NumberRealEstateLoansOrLines], 'NumberOfDependents': [NumberOfDependents]})

    proc = enforce_order(apply_preproc(raw, params or {}), feature_order)
    months_input = proc.drop(columns=['Income_missing','Deps_missing'], errors='ignore')

    st.markdown("---")
    
    if st.button("🚀 Generate Risk Score", use_container_width=True, type="primary"):
        with st.spinner("Analyzing applicant risk profile..."):
            p_sel = predict_model(proc, sel_key, models)
            p_rf, p_xgb, p_lgb = [predict_model(proc, k, models) for k in ['rf', 'xgb', 'lgb']]
            months = float(models['months'].predict(months_input)[0]) if 'months' in models else np.nan
            band, color = risk_band(p_sel)
            thr = 0.5

            st.markdown("### 🎯 Prediction Results")
            top = st.columns([1.5, 1])
            with top[0]:
                st.markdown('<div class="gauge">', unsafe_allow_html=True)
                st.plotly_chart(gauge(p_sel, title=f"{model_choice}: Default Probability"), use_container_width=True, key="main_gauge")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with top[1]:
                st.markdown(f'<div class="tile"><div class="name">Risk Classification</div><div class="p" style="color:{color}">{band}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="tile"><div class="name">Est. Months to Repay</div><div class="p">{months:.1f} months</div></div>', unsafe_allow_html=True)
                decision = "🚫 Reject" if p_sel >= thr else "✅ Approve"
                dec_color = "#ef4444" if p_sel >= thr else "#22c55e"
                st.markdown(f'<div class="tile"><div class="name">Decision</div><div class="p" style="color:{dec_color}">{decision}</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 Multi-Model Comparison")
            st.markdown('<div class="comparison-box"><b>All models evaluated on the same applicant data for robust decision-making</b></div>', unsafe_allow_html=True)
            
            comp_cols = st.columns(3)
            for col, (mname, prob, mcolor) in zip(comp_cols, [("Random Forest", p_rf, "#22c55e"), ("XGBoost", p_xgb, "#3b82f6"), ("LightGBM", p_lgb, "#f59e0b")]):
                decision_icon = "✅" if prob < thr else "🚫"
                with col:
                    st.markdown(f'<div class="metric-card" style="border-left: 4px solid {mcolor}"><h3 style="margin:0 0 8px 0; font-size:14px; color:#1e293b">{mname}</h3>'
                               f'<div style="font-size:26px; font-weight:800; color:{mcolor}; margin:6px 0">{prob*100:.2f}%</div>'
                               f'<div style="font-size:13px; color:#64748b">{decision_icon} {("Approve" if prob < thr else "Reject")}</div></div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div style="text-align:center; color:#94a3b8; font-size:13px; padding:20px 0"><b>RiskScore v2.0</b> | Credit Risk Analytics Platform</div>', unsafe_allow_html=True)
