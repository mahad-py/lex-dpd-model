import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import plotly.express as px

st.set_page_config(page_title="LEX — DPD Behavioural Scorecard", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;}
h1,h2,h3{color:#0f2138;}
.flag-ok  {background:#e3f5ec;color:#2f9e6e;padding:10px 16px;border-radius:8px;font-weight:600;}
.flag-warn{background:#fdf3e7;color:#d98c2b;padding:10px 16px;border-radius:8px;font-weight:600;}
.flag-bad {background:#fbeae6;color:#c2452f;padding:10px 16px;border-radius:8px;font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.title("📊 RADIAN8 LEX — DPD Behavioural Scorecard")
st.caption("Trained exclusively on approved applicants. Predicts post-approval repayment behaviour using A+B+C data.")

# ─── FEATURE DEFINITIONS ─────────────────────────────────────
CAT_COLS = ['gender','nationality_group','city','employment_status',
            'employer_sector','channel','product_type','risk_band']

A_COLS = ['age','months_in_job','monthly_salary_sar','other_monthly_income_sar',
          'total_monthly_income_sar','existing_monthly_obligations_sar',
          'nafath_verified','yakeen_match','approved_amount_or_limit_sar',
          'requested_tenor_months','annual_profit_rate',
          'approved_monthly_payment_est_sar','policy_dbr_cap','dbr_post',
          'max_affordable_new_payment_sar','policy_salary_multiple_cap',
          'max_approvable_limit_sar','risk_score_300_900']

B_COLS = ['salary_consistency_score','avg_eom_balance_sar','cash_withdrawal_ratio',
          'monthly_spend_volatility','credit_utilization_ratio','missed_payments_6m',
          'digital_txn_ratio','avg_monthly_inflow_sar','avg_monthly_outflow_sar']

B_ENHANCED = ['salary_day_variation','times_negative_6m',
              'outflow_inflow_ratio','late_payment_freq_score']

C_COLS = ['prev_loans_count','prev_defaults_count','max_dpd_last_loan',
          'ever_restructured','ever_written_off','simah_enquiries_3m',
          'active_credit_facilities']

ALL_FEATURES_BASE     = CAT_COLS + A_COLS + B_COLS + C_COLS
ALL_FEATURES_ENHANCED = CAT_COLS + A_COLS + B_COLS + B_ENHANCED + C_COLS

DPD_CLASSES = ['current','dpd_30','dpd_60','dpd_90plus','written_off']
DPD_ICONS   = {'current':'✅','dpd_30':'⚠️','dpd_60':'🟡','dpd_90plus':'🔴','written_off':'❌'}
DPD_LABELS  = {
    'current':    'Paying on time — no issues',
    'dpd_30':     'Likely to miss a payment by 30 days',
    'dpd_60':     'Likely to go 60 days late — needs monitoring',
    'dpd_90plus': 'Serious delinquency risk — 90+ days late',
    'written_off':'High probability of total loss'
}

CLASS_WEIGHTS = {'current':1,'dpd_30':3,'dpd_60':8,'dpd_90plus':12,'written_off':12}

# ─── SIDEBAR ─────────────────────────────────────────────────
st.sidebar.header("1. Training Data")
uploaded = st.sidebar.file_uploader(
    "Upload approved applicants dataset (.xlsx or .csv)",
    type=["xlsx","csv"], accept_multiple_files=False)

@st.cache_data(show_spinner=False)
def load_data(fname):
    if fname.endswith('.csv'):
        return pd.read_csv(uploaded)
    try:
        return pd.read_excel(uploaded, sheet_name=0)
    except:
        return pd.read_excel(uploaded)

def detect_enhanced(df):
    return all(c in df.columns for c in B_ENHANCED)

@st.cache_resource(show_spinner=False)
def train_model(df_hash, is_enhanced):
    feats = ALL_FEATURES_ENHANCED if is_enhanced else ALL_FEATURES_BASE
    X = df[[c for c in feats if c in df.columns]].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype('category')
    y = df['dpd_outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        random_state=42, verbosity=-1,
        class_weight=CLASS_WEIGHTS if is_enhanced else 'balanced')
    clf.fit(X_train, y_train, categorical_feature=[c for c in CAT_COLS if c in X.columns])
    pred   = clf.predict(X_test)
    acc    = accuracy_score(y_test, pred)
    f1     = f1_score(y_test, pred, average='macro')
    imp    = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    report = classification_report(y_test, pred, output_dict=True)
    return clf, acc, f1, imp, report, list(X.columns)

if uploaded:
    df = load_data(uploaded.name)
    is_enhanced = detect_enhanced(df)
    ALL_FEATURES = ALL_FEATURES_ENHANCED if is_enhanced else ALL_FEATURES_BASE
    schema_label = "50K Enhanced (A+B+B_Enhanced+C)" if is_enhanced else "10K Standard (A+B+C)"
    st.sidebar.success(f"✅ Loaded {len(df):,} approved applicants")
    st.sidebar.info(f"Schema: **{schema_label}**")
    with st.spinner("Training DPD behavioural scorecard model..."):
        clf, acc, f1, importance, report, used_features = train_model(str(len(df))+str(is_enhanced), is_enhanced)
    st.sidebar.metric("Model accuracy", f"{acc*100:.2f}%")
    st.sidebar.metric("Macro F1 score", f"{f1*100:.2f}%")
    st.sidebar.caption("Trained on approved applicants only")
else:
    st.info("👈 Upload GCC_Approved_50K_ABC_Enhanced (CSV or XLSX) to train the enhanced model.")
    st.stop()

# ─── TABS ────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Model Overview","🧪 Score a New Applicant","📁 Batch Scoring"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — MODEL OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total approved applicants", f"{len(df):,}")
    c2.metric("Model accuracy",            f"{acc*100:.2f}%")
    c3.metric("Macro F1",                  f"{f1*100:.2f}%")
    c4.metric("NPL rate (90++writeoff)",   f"{(df['dpd_outcome'].isin(['dpd_90plus','written_off'])).mean()*100:.1f}%")

    if is_enhanced:
        st.success("✅ Enhanced model active — 50K dataset with 4 additional behavioural features and fine-tuned class weights")

    col1,col2 = st.columns(2)
    with col1:
        dc = df['dpd_outcome'].value_counts().reset_index()
        dc.columns = ['DPD Outcome','Count']
        fig1 = px.bar(dc, x='DPD Outcome', y='Count', title="Applicant Count by DPD Outcome",
                      color='DPD Outcome',
                      color_discrete_map={'current':'#2f9e6e','dpd_30':'#c9a35a',
                                          'dpd_60':'#d98c2b','dpd_90plus':'#c2452f','written_off':'#0f2138'})
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        db = df.groupby('risk_band')['dpd_outcome'].apply(
            lambda x:(x.isin(['dpd_90plus','written_off'])).mean()*100).reset_index()
        db.columns = ['Risk Band','NPL %']
        fig2 = px.bar(db, x='Risk Band', y='NPL %', title="NPL Rate by Risk Band",
                      color_discrete_sequence=['#c2452f'])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Per-Class Model Performance")
    perf = []
    for cls in DPD_CLASSES:
        if cls in report:
            r = report[cls]
            perf.append({'DPD Class':f"{DPD_ICONS.get(cls,'')} {cls}",
                         'Precision':f"{r['precision']*100:.1f}%",
                         'Recall':f"{r['recall']*100:.1f}%",
                         'F1 Score':f"{r['f1-score']*100:.1f}%",
                         'Support':int(r['support'])})
    st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)

    st.subheader("Top Decision Drivers")
    imp_df = importance.head(12).reset_index()
    imp_df.columns = ['Feature','Importance']
    fig3 = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                  color_discrete_sequence=['#0f2138'])
    fig3.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — SCORE A NEW APPLICANT
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Enter Approved Applicant Details")
    st.caption("This applicant has already been approved. This model predicts their post-approval repayment behaviour.")

    with st.form("dpd_form"):
        st.markdown("**A — Application Data**")
        a1,a2,a3 = st.columns(3)
        with a1:
            gender            = st.selectbox("Gender", sorted(df['gender'].unique()))
            age               = st.number_input("Age", 21, 65, 34)
            nationality_group = st.selectbox("Nationality Group", sorted(df['nationality_group'].unique()))
            city              = st.selectbox("City", sorted(df['city'].unique()))
            employment_status = st.selectbox("Employment Status", sorted(df['employment_status'].unique()))
            employer_sector   = st.selectbox("Employer Sector", sorted(df['employer_sector'].unique()))
        with a2:
            months_in_job                    = st.number_input("Months in Job", 1, 360, 36)
            monthly_salary_sar               = st.number_input("Monthly Salary (SAR)", 3500.0, 80000.0, 12000.0, step=500.0)
            other_monthly_income_sar         = st.number_input("Other Monthly Income (SAR)", 0.0, 15000.0, 0.0, step=200.0)
            existing_monthly_obligations_sar = st.number_input("Existing Monthly Obligations (SAR)", 0.0, 30000.0, 2000.0, step=200.0)
            yakeen_match                     = int(st.selectbox("Yakeen Match", ["Yes","No"]) == "Yes")
        with a3:
            channel                = st.selectbox("Channel", sorted(df['channel'].unique()))
            product_type           = st.selectbox("Product Type", sorted(df['product_type'].unique()))
            approved_amount        = st.number_input("Approved Amount (SAR)", 1000.0, 300000.0, 40000.0, step=1000.0)
            requested_tenor_months = st.number_input("Tenor (months)", 6, 60, 24)
            annual_profit_rate     = st.number_input("Annual Profit Rate", 0.10, 0.40, 0.25, step=0.01)
            risk_score_300_900     = st.slider("Risk Score (560-900)", 560, 900, 670)

        st.divider()
        st.markdown("**B — Behavioural Data**")
        b1,b2,b3 = st.columns(3)
        with b1:
            salary_consistency_score = st.slider("Salary Consistency (0-1)", 0.30, 1.0, 0.85, 0.01,
                                                  help="Combined measure of salary arrival regularity and amount stability")
            avg_eom_balance_sar      = st.number_input("Avg End-of-Month Balance (SAR)", -5000.0, 50000.0, 2000.0, step=500.0)
            cash_withdrawal_ratio    = st.slider("Cash Withdrawal Ratio (0-1)", 0.02, 0.90, 0.20, 0.01)
        with b2:
            monthly_spend_volatility = st.slider("Spending Volatility (0-1)", 0.05, 0.70, 0.20, 0.01)
            credit_utilization_ratio = st.slider("Credit Utilization (0-1)", 0.05, 0.98, 0.40, 0.01)
            missed_payments_6m       = st.selectbox("Missed Payments (last 6m)", [0,1,2,3,4])
        with b3:
            digital_txn_ratio      = st.slider("Digital Transaction Ratio (0-1)", 0.20, 0.99, 0.70, 0.01)
            avg_monthly_inflow_sar  = st.number_input("Avg Monthly Inflow (SAR)", 0.0, 100000.0, monthly_salary_sar, step=500.0)
            avg_monthly_outflow_sar = st.number_input("Avg Monthly Outflow (SAR)", 0.0, 100000.0, monthly_salary_sar*0.75, step=500.0)

        if is_enhanced:
            st.divider()
            st.markdown("**B — Enhanced Behavioural Signals**")
            be1,be2 = st.columns(2)
            with be1:
                salary_day_variation    = st.slider("Salary Arrival Day Variation (0-1)", 0.0, 1.0, 0.15, 0.01,
                                                     help="0 = always same date, 1 = completely random")
                times_negative_6m       = st.selectbox("Times Account Went Negative (last 6m)", [0,1,2,3,4,5,6])
            with be2:
                outflow_inflow_ratio    = st.number_input("Outflow / Inflow Ratio", 0.10, 2.0,
                                                           round(avg_monthly_outflow_sar/max(avg_monthly_inflow_sar,1),2), step=0.01,
                                                           help="Above 1.0 means spending more than earning")
                late_payment_freq_score = st.slider("Late Payment Frequency Score (0-1)", 0.0, 1.0,
                                                     round(min(missed_payments_6m/4,1.0),2), 0.01,
                                                     help="0 = never late, 1 = always late")

        st.divider()
        st.markdown("**C — Collection Data**")
        cc1,cc2,cc3 = st.columns(3)
        with cc1:
            prev_loans_count    = st.selectbox("Previous Loans Count", [0,1,2,3,4,5])
            prev_defaults_count = st.selectbox("Previous Defaults Count", [0,1,2,3])
            max_dpd_last_loan   = st.selectbox("Max DPD on Last Loan", [0,15,30,60,90,120,180])
        with cc2:
            ever_restructured = int(st.selectbox("Ever Restructured?", ["No","Yes"]) == "Yes")
            ever_written_off  = int(st.selectbox("Ever Written Off?",   ["No","Yes"]) == "Yes")
        with cc3:
            simah_enquiries_3m     = st.selectbox("SIMAH Enquiries (3m)", [0,1,2,3,4,5])
            active_credit_facilities = st.selectbox("Active Credit Facilities", [0,1,2,3,4])

        submitted = st.form_submit_button("🔍 Predict DPD Outcome", use_container_width=True)

    if submitted:
        total_income     = monthly_salary_sar + other_monthly_income_sar
        approved_payment = (approved_amount*(1+annual_profit_rate*requested_tenor_months/12)/requested_tenor_months)
        dbr_post_val     = (existing_monthly_obligations_sar+approved_payment)/total_income if total_income>0 else 0
        max_afford       = max(0.45*total_income-existing_monthly_obligations_sar, 0)
        max_approvable   = min(max_afford*requested_tenor_months/(1+annual_profit_rate*requested_tenor_months/12), 15*monthly_salary_sar)
        risk_band_val    = 'A' if risk_score_300_900>=700 else 'B' if risk_score_300_900>=650 else 'C' if risk_score_300_900>=600 else 'D'

        base = {
            'gender':gender,'age':age,'nationality_group':nationality_group,'city':city,
            'employment_status':employment_status,'employer_sector':employer_sector,
            'months_in_job':months_in_job,'monthly_salary_sar':monthly_salary_sar,
            'other_monthly_income_sar':other_monthly_income_sar,'total_monthly_income_sar':total_income,
            'existing_monthly_obligations_sar':existing_monthly_obligations_sar,
            'nafath_verified':1,'yakeen_match':yakeen_match,
            'channel':channel,'product_type':product_type,
            'approved_amount_or_limit_sar':approved_amount,
            'requested_tenor_months':requested_tenor_months,'annual_profit_rate':annual_profit_rate,
            'approved_monthly_payment_est_sar':round(approved_payment,2),
            'policy_dbr_cap':0.45,'dbr_post':round(dbr_post_val,4),
            'max_affordable_new_payment_sar':round(max_afford,2),
            'policy_salary_multiple_cap':15,'max_approvable_limit_sar':round(max_approvable,2),
            'risk_score_300_900':risk_score_300_900,'risk_band':risk_band_val,
            'salary_consistency_score':salary_consistency_score,
            'avg_eom_balance_sar':avg_eom_balance_sar,'cash_withdrawal_ratio':cash_withdrawal_ratio,
            'monthly_spend_volatility':monthly_spend_volatility,
            'credit_utilization_ratio':credit_utilization_ratio,'missed_payments_6m':missed_payments_6m,
            'digital_txn_ratio':digital_txn_ratio,'avg_monthly_inflow_sar':avg_monthly_inflow_sar,
            'avg_monthly_outflow_sar':avg_monthly_outflow_sar,
            'prev_loans_count':prev_loans_count,'prev_defaults_count':prev_defaults_count,
            'max_dpd_last_loan':max_dpd_last_loan,'ever_restructured':ever_restructured,
            'ever_written_off':ever_written_off,'simah_enquiries_3m':simah_enquiries_3m,
            'active_credit_facilities':active_credit_facilities
        }
        if is_enhanced:
            base.update({
                'salary_day_variation':salary_day_variation,
                'times_negative_6m':times_negative_6m,
                'outflow_inflow_ratio':outflow_inflow_ratio,
                'late_payment_freq_score':late_payment_freq_score
            })

        row = pd.DataFrame([base])
        X_new = row[used_features].copy()
        for c in CAT_COLS:
            if c in X_new.columns:
                X_new[c] = X_new[c].astype('category').cat.set_categories(
                    df[c].astype('category').cat.categories)

        pred       = clf.predict(X_new)[0]
        proba      = clf.predict_proba(X_new)[0]
        classes    = list(clf.classes_)
        confidence = proba.max()*100
        npl_idx    = [classes.index(c) for c in ['dpd_90plus','written_off'] if c in classes]
        npl_prob   = proba[npl_idx].sum()*100
        risk_label = "🟢 Low Risk" if npl_prob<5 else "🟡 Medium Risk" if npl_prob<15 else "🔴 High Risk"

        st.divider()
        r1,r2,r3,r4 = st.columns(4)
        with r1:
            st.markdown(f"### {DPD_ICONS.get(pred,'⚪')} DPD Prediction")
            st.markdown(f"## **{pred.replace('_',' ').upper()}**")
            st.caption(f"Confidence: {confidence:.1f}%")
        with r2:
            st.markdown("### ⚠️ NPL Probability")
            st.markdown(f"## {npl_prob:.1f}%")
            st.caption("Probability of 90+ DPD or writeoff")
        with r3:
            st.markdown("### 🎯 Risk Label")
            st.markdown(f"## {risk_label}")
        with r4:
            st.markdown("### 📈 Risk Band")
            st.markdown(f"## {risk_band_val}")
            st.caption(f"SIMAH Score: {risk_score_300_900}/900")

        st.divider()
        st.subheader("📋 DPD Probability Breakdown")
        prob_rows = []
        for cls in DPD_CLASSES:
            if cls in classes:
                p = proba[classes.index(cls)]*100
                prob_rows.append({
                    'Outcome':f"{DPD_ICONS.get(cls,'')} {cls.replace('_',' ').title()}",
                    'Meaning':DPD_LABELS[cls],
                    'Probability':f"{p:.1f}%"
                })
        st.dataframe(pd.DataFrame(prob_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("📋 ABC Scorecard")

        def badge(s): return "🟢 Good" if s>=70 else "🟡 Moderate" if s>=50 else "🔴 Weak"

        a_risk    = round(((risk_score_300_900-560)/340)*100)
        a_afford  = round(max(0,min(100,(1-dbr_post_val/0.45)*100)))
        a_employ  = round(min(100,(months_in_job/60)*100))
        a_score   = round(a_risk*0.40+a_afford*0.35+a_employ*0.25)

        b_salary  = round(salary_consistency_score*100)
        b_balance = round(min(100,max(0,50+(avg_eom_balance_sar/total_income)*50))) if total_income>0 else 50
        b_cash    = round((1-cash_withdrawal_ratio)*100)
        b_util    = round((1-credit_utilization_ratio)*100)
        b_pay     = round(max(0,100-missed_payments_6m*25))
        b_score   = round(b_salary*0.25+b_balance*0.20+b_cash*0.15+b_util*0.25+b_pay*0.15)

        if is_enhanced:
            b_dayvar  = round((1-salary_day_variation)*100)
            b_neg     = round(max(0,100-times_negative_6m*16))
            b_oiratio = round(max(0,min(100,(1-(outflow_inflow_ratio-0.5))*100))) if outflow_inflow_ratio>0.5 else 100
            b_late    = round((1-late_payment_freq_score)*100)
            b_score   = round(b_score*0.60 + b_dayvar*0.10 + b_neg*0.10 + b_oiratio*0.10 + b_late*0.10)

        c_hist    = round(max(0,100-prev_defaults_count*40))
        c_dpd_s   = round(max(0,100-(max_dpd_last_loan/180)*100))
        c_wo      = 0 if ever_written_off else 100
        c_rest    = 60 if ever_restructured else 100
        c_enq     = round(max(0,100-simah_enquiries_3m*15))
        c_score   = round(c_hist*0.30+c_dpd_s*0.25+c_wo*0.25+c_rest*0.10+c_enq*0.10)

        aggregate = round(a_score*0.35+b_score*0.40+c_score*0.25)

        rows = [
            {"Layer":"A — Application","Component":"SIMAH Risk Score",        "Score":f"{a_risk}/100",   "Status":badge(a_risk)},
            {"Layer":"A — Application","Component":"Affordability (DBR Post)","Score":f"{a_afford}/100", "Status":badge(a_afford)},
            {"Layer":"A — Application","Component":"Employment Stability",     "Score":f"{a_employ}/100", "Status":badge(a_employ)},
            {"Layer":"A — Application","Component":"A Layer Score",            "Score":f"{a_score}/100",  "Status":badge(a_score)},
            {"Layer":"B — Behaviour",  "Component":"Salary Consistency",       "Score":f"{b_salary}/100", "Status":badge(b_salary)},
            {"Layer":"B — Behaviour",  "Component":"End-of-Month Balance",     "Score":f"{b_balance}/100","Status":badge(b_balance)},
            {"Layer":"B — Behaviour",  "Component":"Cash Withdrawal Pattern",  "Score":f"{b_cash}/100",   "Status":badge(b_cash)},
            {"Layer":"B — Behaviour",  "Component":"Credit Utilization",       "Score":f"{b_util}/100",   "Status":badge(b_util)},
            {"Layer":"B — Behaviour",  "Component":"Payment Consistency (6m)", "Score":f"{b_pay}/100",    "Status":badge(b_pay)},
        ]
        if is_enhanced:
            rows += [
                {"Layer":"B — Enhanced","Component":"Salary Arrival Regularity",  "Score":f"{b_dayvar}/100", "Status":badge(b_dayvar)},
                {"Layer":"B — Enhanced","Component":"Account Negative Frequency",  "Score":f"{b_neg}/100",    "Status":badge(b_neg)},
                {"Layer":"B — Enhanced","Component":"Outflow/Inflow Ratio",        "Score":f"{b_oiratio}/100","Status":badge(b_oiratio)},
                {"Layer":"B — Enhanced","Component":"Late Payment Frequency",      "Score":f"{b_late}/100",   "Status":badge(b_late)},
            ]
        rows += [
            {"Layer":"B — Behaviour",  "Component":"B Layer Score",               "Score":f"{b_score}/100",  "Status":badge(b_score)},
            {"Layer":"C — Collection", "Component":"Previous Default History",     "Score":f"{c_hist}/100",   "Status":badge(c_hist)},
            {"Layer":"C — Collection", "Component":"Max Days Past Due",            "Score":f"{c_dpd_s}/100",  "Status":badge(c_dpd_s)},
            {"Layer":"C — Collection", "Component":"Write-off History",            "Score":f"{c_wo}/100",     "Status":"✅ Clean" if not ever_written_off else "🔴 Written Off"},
            {"Layer":"C — Collection", "Component":"Loan Restructuring",           "Score":f"{c_rest}/100",   "Status":"✅ Clean" if not ever_restructured else "⚠️ Restructured"},
            {"Layer":"C — Collection", "Component":"SIMAH Enquiries (3m)",         "Score":f"{c_enq}/100",    "Status":badge(c_enq)},
            {"Layer":"C — Collection", "Component":"C Layer Score",                "Score":f"{c_score}/100",  "Status":badge(c_score)},
            {"Layer":"⭐ Overall",     "Component":"Aggregate Score (A+B+C)",      "Score":f"{aggregate}/100","Status":badge(aggregate)},
            {"Layer":"⭐ Overall",     "Component":"NPL Probability",              "Score":f"{npl_prob:.1f}%","Status":"🟢 Low" if npl_prob<5 else "🟡 Medium" if npl_prob<15 else "🔴 High"},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Review Flag")
        if npl_prob>=15:
            st.markdown(f'<div class="flag-bad">🚩 High NPL probability ({npl_prob:.1f}%) — recommend additional review before disbursement</div>', unsafe_allow_html=True)
        elif npl_prob>=5:
            st.markdown(f'<div class="flag-warn">⚠️ Moderate NPL probability ({npl_prob:.1f}%) — monitor account closely after disbursement</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="flag-ok">✅ Low NPL probability ({npl_prob:.1f}%) — proceed with standard monitoring</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — BATCH SCORING
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Batch DPD Scoring")
    st.caption("Upload a file of approved applicants. Policy fields are derived automatically if missing.")
    batch_file = st.file_uploader("Upload approved applicants batch", type=["csv","xlsx"], key="batch")

    if batch_file:
        bdf = pd.read_csv(batch_file) if batch_file.name.endswith('.csv') else pd.read_excel(batch_file)

        # derive missing policy fields if not present
        if 'total_monthly_income_sar' not in bdf.columns:
            bdf['total_monthly_income_sar'] = bdf['monthly_salary_sar'] + bdf.get('other_monthly_income_sar',0)
        if 'approved_monthly_payment_est_sar' not in bdf.columns:
            bdf['approved_monthly_payment_est_sar'] = (bdf['approved_amount_or_limit_sar']*
                (1+bdf['annual_profit_rate']*bdf['requested_tenor_months']/12)/bdf['requested_tenor_months']).round(2)
        if 'dbr_post' not in bdf.columns:
            bdf['dbr_post'] = ((bdf['existing_monthly_obligations_sar']+bdf['approved_monthly_payment_est_sar'])/
                               bdf['total_monthly_income_sar']).round(4)
        if 'policy_dbr_cap' not in bdf.columns: bdf['policy_dbr_cap'] = 0.45
        if 'policy_salary_multiple_cap' not in bdf.columns: bdf['policy_salary_multiple_cap'] = 15
        if 'max_affordable_new_payment_sar' not in bdf.columns:
            bdf['max_affordable_new_payment_sar'] = (0.45*bdf['total_monthly_income_sar']-bdf['existing_monthly_obligations_sar']).clip(lower=0).round(2)
        if 'max_approvable_limit_sar' not in bdf.columns:
            al = bdf['max_affordable_new_payment_sar']*bdf['requested_tenor_months']/(1+bdf['annual_profit_rate']*bdf['requested_tenor_months']/12)
            sl = 15*bdf['monthly_salary_sar']
            bdf['max_approvable_limit_sar'] = pd.concat([al,sl],axis=1).min(axis=1).round(2)
        if 'nafath_verified' not in bdf.columns: bdf['nafath_verified'] = 1
        if 'risk_band' not in bdf.columns:
            bdf['risk_band'] = bdf['risk_score_300_900'].apply(
                lambda s:'A' if s>=700 else 'B' if s>=650 else 'C' if s>=600 else 'D')
        if 'outflow_inflow_ratio' not in bdf.columns and is_enhanced:
            bdf['outflow_inflow_ratio'] = (bdf['avg_monthly_outflow_sar']/bdf['avg_monthly_inflow_sar'].replace(0,1)).round(3)
        if 'late_payment_freq_score' not in bdf.columns and is_enhanced:
            bdf['late_payment_freq_score'] = (bdf['missed_payments_6m']/4).clip(0,1).round(3)

        missing = [c for c in used_features if c not in bdf.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        X_b = bdf[used_features].copy()
        for c in CAT_COLS:
            if c in X_b.columns:
                X_b[c] = X_b[c].astype('category').cat.set_categories(
                    df[c].astype('category').cat.categories)

        bdf['dpd_prediction'] = clf.predict(X_b)
        proba_b = clf.predict_proba(X_b)
        classes = list(clf.classes_)
        bdf['confidence_%'] = (proba_b.max(axis=1)*100).round(1)
        npl_idx = [classes.index(c) for c in ['dpd_90plus','written_off'] if c in classes]
        bdf['npl_probability_%'] = (proba_b[:,npl_idx].sum(axis=1)*100).round(1)
        bdf['risk_label'] = bdf['npl_probability_%'].apply(
            lambda p:'🟢 Low' if p<5 else '🟡 Medium' if p<15 else '🔴 High')

        st.success(f"Scored {len(bdf):,} approved applicants")

        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Current (on time)",  int((bdf['dpd_prediction']=='current').sum()))
        s2.metric("DPD 30/60",          int(bdf['dpd_prediction'].isin(['dpd_30','dpd_60']).sum()))
        s3.metric("DPD 90+/Writeoff",   int(bdf['dpd_prediction'].isin(['dpd_90plus','written_off']).sum()))
        s4.metric("High NPL Risk",       int((bdf['npl_probability_%']>=15).sum()))

        fig_b = px.histogram(bdf, x='npl_probability_%', nbins=20,
                             title="NPL Probability Distribution",
                             color_discrete_sequence=['#c2452f'])
        st.plotly_chart(fig_b, use_container_width=True)

        disp = []
        if 'application_id' in bdf.columns: disp.append('application_id')
        if 'profile' in bdf.columns: disp.append('profile')
        disp += ['nationality_group','city','monthly_salary_sar','risk_score_300_900',
                 'risk_band','dpd_prediction','npl_probability_%','risk_label','confidence_%']
        st.dataframe(bdf[[c for c in disp if c in bdf.columns]], use_container_width=True)
        st.download_button("⬇️ Download DPD Scores (CSV)", bdf.to_csv(index=False), "dpd_scores.csv")
