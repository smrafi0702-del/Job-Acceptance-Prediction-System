from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="🚀",
    layout="wide",
)

APP_DIR = Path(__file__).parent


# --------------------------------------------------------------------------------
# LOAD DATA & MODEL
# --------------------------------------------------------------------------------
@st.cache_data
def load_data():
  df = pd.read_csv(APP_DIR / "cleaned_job_placement_data_after_ml.csv")
  return df


@st.cache_resource
def load_model():
  with open(APP_DIR / "model.pkl", "rb") as f:
    package = pickle.load(f)
  return package


try:
  df = load_data()
  model_package = load_model()
  model = model_package["model"]
  feature_columns = model_package["feature_columns"]
except Exception as e:
  st.error(
      f"Error loading files: {e}. Please ensure"
      " 'cleaned_job_placement_data_after_ml.csv' and 'model.pkl' are in the"
      " same directory."
  )
  st.stop()

# --------------------------------------------------------------------------------
# HEADER SECTION
# --------------------------------------------------------------------------------
st.markdown(
    """
    <h1 style="margin-bottom: 0;">
        <span style="color: #EC4899;">🚀</span> Job Acceptance Prediction System
    </h1>
    """,
    unsafe_allow_html=True,
)
st.write(
    "Predict whether a selected candidate accepts the offer and understand"
    " what influences their decision using real data."
)
st.markdown("---")

# --------------------------------------------------------------------------------
# TABS SETUP
# --------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏢 Overview",
    "🔮 Predict Acceptance",
    "📈 Dashboard & KPIs",
    "⚡ Model Insights",
])

# ================================================================================
# TAB 1 — OVERVIEW
# ================================================================================
with tab1:
  total_candidates = len(df)
  placement_rate = (
      (df["status"] == "Placed").mean() * 100 if "status" in df.columns else 0.0
  )
  avg_interview = (
      df["interview_avg_score"].mean()
      if "interview_avg_score" in df.columns
      else 0.0
  )
  avg_skills = (
      df["skills_match_percentage"].mean()
      if "skills_match_percentage" in df.columns
      else 0.0
  )

  col1, col2, col3, col4 = st.columns(4)

  with col1:
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.05); text-align: center;">
            <p style="margin: 0; font-size: 17px; color: #ccc;">Total Candidates</p>
            <h2 style="margin: 5px 0 0 0; color: #FF007F; font-size: 32px;">{total_candidates:,}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

  with col2:
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.05); text-align: center;">
            <p style="margin: 0; font-size: 17px; color: #ccc;">Placement Rate</p>
            <h2 style="margin: 5px 0 0 0; color: #FFD93D; font-size: 32px;">{placement_rate:.1f}%</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

  with col3:
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.05); text-align: center;">
            <p style="margin: 0; font-size: 17px; color: #ccc;">Avg Interview Score</p>
            <h2 style="margin: 5px 0 0 0; color: #00FF7F; font-size: 32px;">{avg_interview:.1f}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

  with col4:
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.05); text-align: center;">
            <p style="margin: 0; font-size: 17px; color: #ccc;">Avg Skills Match %</p>
            <h2 style="margin: 5px 0 0 0; color: #00E5FF; font-size: 32px;">{avg_skills:.1f}%</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

  st.markdown("---")

  st.subheader("📌 About This Project")
  st.info(
      "Recruitment teams handle huge volumes of candidate data — academics,"
      " skills, interviews, and job-market conditions. Not every eligible"
      " candidate accepts an offer! This app analyzes candidate data to predict"
      " whether a candidate will accept or reject a job offer, surface the key"
      " factors driving that decision, and give recruiters clear, actionable"
      " insights."
  )

  col_a, col_b = st.columns(2)

  with col_a:
    st.subheader("💼 Business Use Cases")
    st.markdown(
        """
        * ✅ Predict offer-acceptance probability before extending an offer
        * ✅ Reduce offer dropouts and shorten hiring cycle time
        * ✅ Identify the factors that most impact placement success
        * ✅ Optimize interview and evaluation strategies
        * ✅ Support data-driven decisions for HR managers
        """
    )

  with col_b:
    st.subheader("🧠 How It Works")
    st.markdown(
        """
        1. 1️⃣ Enter a candidate's academic, skills, interview & offer details
        2. 2️⃣ The app engineers the same features used in training
        3. 3️⃣ A trained **Random Forest** model predicts Accept / Reject
        4. 4️⃣ Explore dataset-wide trends in the Dashboard tab
        5. 5️⃣ See which features matter most in Model Insights
        """
    )

# ================================================================================
# TAB 2 — PREDICT ACCEPTANCE
# ================================================================================
with tab2:
  st.subheader("📋 Enter Candidate Details")
  st.info("Fill in the candidate's details below and click predict.")

  with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
      st.markdown("#### 🧑‍💼 Candidate Profile")
      age_years = st.number_input("Age (years)", 18, 60, 25)
      gender = st.selectbox("Gender", ["Male", "Female"])
      degree_specialization = st.selectbox(
          "Degree Specialization",
          [
              "Computer Science",
              "Electronics",
              "Information Technology",
              "Mechanical",
              "Others",
          ],
      )
      years_of_experience = st.number_input("Years of Experience", 0, 40, 1)
      academic_score = st.slider(
          "Overall Academic Score (%)",
          0.0,
          100.0,
          70.0,
          help="Average of 10th, 12th & Degree marks",
      )

    with col2:
      st.markdown("#### ⭐ Fit for the Role")
      interview_score = st.slider(
          "Interview Performance (%)",
          0.0,
          100.0,
          65.0,
          help="Average of technical, aptitude & communication scores",
      )
      skills_match_percentage = st.slider(
          "Skills Match with Job (%)", 0.0, 100.0, 70.0
      )
      certifications_count = st.number_input("Certifications Count", 0, 10, 1)
      company_tier = st.selectbox(
          "Company Tier", ["Tier 1", "Tier 2", "Tier 3"]
      )
      competition_level = st.selectbox(
          "Competition Level", ["Low", "Medium", "High"]
      )

    submitted = st.form_submit_button(
        "⚡ Analyze Offer Acceptance Probability", use_container_width=True
    )

  if submitted:
    ssc_percentage = academic_score
    hsc_percentage = academic_score
    degree_percentage = academic_score
    technical_score = interview_score
    aptitude_score = interview_score
    communication_score = interview_score
  else:
    ssc_percentage = 0.0
    hsc_percentage = 0.0
    degree_percentage = 0.0
    technical_score = 0.0
    aptitude_score = 0.0
    communication_score = 0.0

  internship_experience = "No"
  relevant_experience = "Relevant"
  job_role_match = "Matched"
  career_switch_willingness = "Not Willing"
  relocation_willingness = "Not Willing"
  previous_ctc_lpa = 4.0
  expected_ctc_lpa = 6.0
  bond_requirement = "Not Required"
  notice_period_days = 0
  layoff_history = "No"
  employment_gap_months = 0.0

  if submitted:

    def experience_category(years):
      if years == 0:
        return "Fresher"
      elif years <= 3:
        return "Junior"
      else:
        return "Senior"

    academic_avg = (ssc_percentage + hsc_percentage + degree_percentage) / 3

    def academic_band(avg):
      if avg < 60:
        return "Average"
      elif avg < 75:
        return "Good"
      else:
        return "Excellent"

    def skill_level(pct):
      if pct < 50:
        return "Low"
      elif pct < 75:
        return "Medium"
      else:
        return "High"

    interview_avg_score = (
        technical_score + aptitude_score + communication_score
    ) / 3

    def interview_category(score):
      if score < 50:
        return "Weak"
      elif score < 75:
        return "Average"
      else:
        return "Strong"

    def experience_score(years):
      return min(years * 15, 100)

    competition_adjustment = {"Low": 5, "Medium": 0, "High": -5}

    placement_probability_score = np.clip(
        0.35 * interview_avg_score
        + 0.25 * skills_match_percentage
        + 0.15 * academic_avg
        + 0.10 * min(certifications_count, 10) * 10
        + 0.15 * experience_score(years_of_experience)
        + competition_adjustment[competition_level],
        0,
        100,
    )

    def probability_band(score):
      if score < 50:
        return "Low"
      elif score < 75:
        return "Medium"
      else:
        return "High"

    exp_cat = experience_category(years_of_experience)
    acad_band = academic_band(academic_avg)
    skl_level = skill_level(skills_match_percentage)
    interview_cat = interview_category(interview_avg_score)
    prob_band = probability_band(placement_probability_score)

    row = {col: 0 for col in feature_columns}

    numeric_values = {
        "age_years": age_years,
        "ssc_percentage": ssc_percentage,
        "hsc_percentage": hsc_percentage,
        "degree_percentage": degree_percentage,
        "technical_score": technical_score,
        "aptitude_score": aptitude_score,
        "communication_score": communication_score,
        "skills_match_percentage": skills_match_percentage,
        "certifications_count": certifications_count,
        "years_of_experience": years_of_experience,
        "previous_ctc_lpa": previous_ctc_lpa,
        "expected_ctc_lpa": expected_ctc_lpa,
        "notice_period_days": notice_period_days,
        "employment_gap_months": employment_gap_months,
        "academic_avg": academic_avg,
        "interview_avg_score": interview_avg_score,
        "placement_probability_score": placement_probability_score,
    }
    for k, v in numeric_values.items():
      if k in row:
        row[k] = v

    dummy_map = {
        ("gender", "Male"): "gender_Male",
        ("degree_specialization", "Electronics"): (
            "degree_specialization_Electronics"
        ),
        ("degree_specialization", "Information Technology"): (
            "degree_specialization_Information Technology"
        ),
        ("degree_specialization", "Mechanical"): (
            "degree_specialization_Mechanical"
        ),
        ("degree_specialization", "Others"): "degree_specialization_Others",
        ("internship_experience", "Yes"): "internship_experience_Yes",
        ("career_switch_willingness", "Willing"): (
            "career_switch_willingness_Willing"
        ),
        ("relevant_experience", "Relevant"): "relevant_experience_Relevant",
        ("company_tier", "Tier 2"): "company_tier_Tier 2",
        ("company_tier", "Tier 3"): "company_tier_Tier 3",
        ("job_role_match", "Not Matched"): "job_role_match_Not Matched",
        ("competition_level", "Low"): "competition_level_Low",
        ("competition_level", "Medium"): "competition_level_Medium",
        ("bond_requirement", "Required"): "bond_requirement_Required",
        ("layoff_history", "Yes"): "layoff_history_Yes",
        ("relocation_willingness", "Willing"): "relocation_willingness_Willing",
        ("experience_category", "Junior"): "experience_category_Junior",
        ("experience_category", "Senior"): "experience_category_Senior",
        ("academic_band", "Excellent"): "academic_band_Excellent",
        ("academic_band", "Good"): "academic_band_Good",
        ("skills_match_level", "Low"): "skills_match_level_Low",
        ("skills_match_level", "Medium"): "skills_match_level_Medium",
        ("interview_performance_category", "Strong"): (
            "interview_performance_category_Strong"
        ),
        ("interview_performance_category", "Weak"): (
            "interview_performance_category_Weak"
        ),
        ("placement_probability_band", "Low"): "placement_probability_band_Low",
        ("placement_probability_band", "Medium"): (
            "placement_probability_band_Medium"
        ),
    }

    selections = {
        "gender": gender,
        "degree_specialization": degree_specialization,
        "internship_experience": internship_experience,
        "career_switch_willingness": career_switch_willingness,
        "relevant_experience": relevant_experience,
        "company_tier": company_tier,
        "job_role_match": job_role_match,
        "competition_level": competition_level,
        "bond_requirement": bond_requirement,
        "layoff_history": layoff_history,
        "relocation_willingness": relocation_willingness,
        "experience_category": exp_cat,
        "academic_band": acad_band,
        "skills_match_level": skl_level,
        "interview_performance_category": interview_cat,
        "placement_probability_band": prob_band,
    }

    for field, value in selections.items():
      key = (field, value)
      if key in dummy_map and dummy_map[key] in row:
        row[dummy_map[key]] = 1

    X_input = pd.DataFrame([row])[feature_columns]

    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    st.markdown("---")
    st.subheader("📢 Prediction Result")

    res_col, gauge_col = st.columns([1, 1.2])
    with res_col:
      if pred == 1:
        st.success(
            f"✅ Candidate is likely to ACCEPT the job offer!\n\nConfidence:"
            f" {proba*100:.1f}%"
        )
      else:
        st.error(
            f"❌ Candidate is likely to REJECT the job offer.\n\nConfidence:"
            f" {(1-proba)*100:.1f}%"
        )

      st.write(f"**Interview Avg:** {interview_avg_score:.1f}")
      st.write(f"**Academic Avg:** {academic_avg:.1f}")
      st.write(f"**Placement Score:** {placement_probability_score:.1f}")

    with gauge_col:
      fig = go.Figure(
          go.Indicator(
              mode="gauge+number",
              value=proba * 100,
              number={"suffix": "%"},
              title={
                  "text": "Acceptance Probability",
                  "font": {"size": 20},
              },
              gauge={
                  "axis": {"range": [0, 100]},
                  "bar": {"color": "darkblue"},
                  "steps": [
                      {"range": [0, 40], "color": "lightcoral"},
                      {"range": [40, 70], "color": "gold"},
                      {"range": [70, 100], "color": "lightgreen"},
                  ],
              },
          )
      )
      fig.update_layout(height=320, margin=dict(t=50, b=20, l=20, r=20))
      st.plotly_chart(fig, use_container_width=True)

# ================================================================================
# TAB 3 — DASHBOARD & KPIS
# ================================================================================
with tab3:
  st.subheader("📊 Key Performance Indicators")

  total_candidates = len(df)
  placement_rate = (
      (df["status"] == "Placed").mean() * 100 if "status" in df.columns else 0.0
  )
  acceptance_rate = placement_rate
  avg_interview = (
      df["interview_avg_score"].mean()
      if "interview_avg_score" in df.columns
      else 0.0
  )
  avg_skills = (
      df["skills_match_percentage"].mean()
      if "skills_match_percentage" in df.columns
      else 0.0
  )
  dropout_rate = 100 - placement_rate
  high_risk_pct = (
      (df["placement_probability_band"] == "Low").mean() * 100
      if "placement_probability_band" in df.columns
      else 0.0
  )
  avg_certs = (
      df["certifications_count"].mean()
      if "certifications_count" in df.columns
      else 0.0
  )

  # Helper function for styled KPI cards with reduced font size (24px)
  def render_kpi_card(col, label, value, color):
    col.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.05); text-align: center;">
            <p style="margin: 0; font-size: 15px; color: #ccc;">{label}</p>
            <h2 style="margin: 5px 0 0 0; color: {color}; font-size: 24px;">{value}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

  # Row 1 KPIs (Top 4 professional corporate colors)
  r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
  render_kpi_card(r1_c1, "Total Candidates", f"{total_candidates:,}", "#3B82F6")
  render_kpi_card(r1_c2, "Placement Rate", f"{placement_rate:.1f}%", "#10B981")
  render_kpi_card(
      r1_c3, "Job Acceptance Rate", f"{acceptance_rate:.1f}%", "#06B6D4"
  )
  render_kpi_card(r1_c4, "Avg Interview Score", f"{avg_interview:.1f}", "#8B5CF6")

  st.markdown(
      '<div style="margin-top: 15px; margin-bottom: 15px;"></div>',
      unsafe_allow_html=True,
  )

  # Row 2 KPIs (Bottom 4 professional corporate colors)
  r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
  render_kpi_card(
      r2_c1,
      "Avg Skills Match %",
      f"{avg_skills:.1f}%" if "skills_match_percentage" in df.columns else "0.0%",
      "#F59E0B",
  )
  render_kpi_card(r2_c2, "Offer Dropout Rate", f"{dropout_rate:.1f}%", "#EF4444")
  render_kpi_card(r2_c3, "High-Risk Candidates", f"{high_risk_pct:.1f}%", "#EC4899")
  render_kpi_card(r2_c4, "Avg Certifications", f"{avg_certs:.1f}", "#6366F1")

  st.markdown("---")
  st.subheader("📈 Candidate Performance Analysis")

  c1, c2 = st.columns(2)
  with c1:
    if "status" in df.columns and "interview_avg_score" in df.columns:
      fig1 = px.box(
          df,
          x="status",
          y="interview_avg_score",
          color="status",
          color_discrete_map={"Placed": "#00D2A0", "Not Placed": "#FF6B6B"},
          title="Interview Score vs Job Acceptance",
      )
      st.plotly_chart(fig1, use_container_width=True)

  with c2:
    if "status" in df.columns and "skills_match_percentage" in df.columns:
      fig2 = px.histogram(
          df,
          x="skills_match_percentage",
          color="status",
          barmode="overlay",
          nbins=40,
          color_discrete_map={"Placed": "#00D2A0", "Not Placed": "#FF6B6B"},
          opacity=0.7,
          title="Skills Match % Impact on Placement",
      )
      st.plotly_chart(fig2, use_container_width=True)

  st.markdown("---")
  st.subheader("🏢 Placement & Acceptance Analysis")
  c3, c4 = st.columns(2)
  with c3:
    if "company_tier" in df.columns and "status" in df.columns:
      tier_rate = (
          df.groupby("company_tier")["status"]
          .apply(lambda x: (x == "Placed").mean() * 100)
          .reindex(["Tier 1", "Tier 2", "Tier 3"])
          .reset_index(name="acceptance_rate")
      )
      fig3 = px.bar(
          tier_rate,
          x="company_tier",
          y="acceptance_rate",
          color="company_tier",
          color_discrete_sequence=["#3B82F6", "#06B6D4", "#10B981"],
          text_auto=".1f",
          title="Acceptance Rate by Company Tier",
      )
      fig3.update_yaxes(title="Acceptance Rate (%)")
      st.plotly_chart(fig3, use_container_width=True)

  with c4:
    if "experience_category" in df.columns and "status" in df.columns:
      exp_rate = (
          df.groupby("experience_category")["status"]
          .apply(lambda x: (x == "Placed").mean() * 100)
          .reindex(["Fresher", "Junior", "Senior"])
          .reset_index(name="placement_rate")
      )
      fig4 = px.bar(
          exp_rate,
          x="experience_category",
          y="placement_rate",
          color="experience_category",
          color_discrete_sequence=["#F59E0B", "#EC4899", "#8B5CF6"],
          text_auto=".1f",
          title="Experience vs Placement Probability",
      )
      fig4.update_yaxes(title="Placement Rate (%)")
      st.plotly_chart(fig4, use_container_width=True)

  st.markdown("---")
  st.subheader("🎤 Interview & Evaluation Analysis")
  c5, c6 = st.columns(2)
  with c5:
    if "competition_level" in df.columns and "status" in df.columns:
      comp_rate = (
          df.groupby("competition_level")["status"]
          .apply(lambda x: (x == "Placed").mean() * 100)
          .reindex(["Low", "Medium", "High"])
          .reset_index(name="acceptance_rate")
      )
      fig5 = px.bar(
          comp_rate,
          x="competition_level",
          y="acceptance_rate",
          color="competition_level",
          color_discrete_sequence=["#10B981", "#3B82F6", "#EF4444"],
          text_auto=".1f",
          title="Competition Level vs Job Acceptance",
      )
      fig5.update_yaxes(title="Acceptance Rate (%)")
      st.plotly_chart(fig5, use_container_width=True)

  with c6:
    numeric_cols = [
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage",
        "technical_score",
        "aptitude_score",
        "communication_score",
        "skills_match_percentage",
        "certifications_count",
        "years_of_experience",
        "interview_avg_score",
        "placement_probability_score",
    ]
    valid_numeric_cols = [c for c in numeric_cols if c in df.columns]
    if valid_numeric_cols:
      corr = df[valid_numeric_cols].corr()
      fig6 = px.imshow(
          corr,
          text_auto=".2f",
          color_continuous_scale="Blues",
          aspect="auto",
          title="Correlation Among Numeric Features",
      )
      fig6.update_layout(height=480)
      st.plotly_chart(fig6, use_container_width=True)

# ================================================================================
# TAB 4 — MODEL INSIGHTS
# ================================================================================
with tab4:
  st.subheader("📌 What Drives Candidate Job Acceptance?")
  st.info(
      "Trained on 50,000 candidate records, our Random Forest model analyzes"
      " historical patterns to figure out what influences people to accept or"
      " reject job offers. The chart below ranks the most important features"
      " the model relies on."
  )

  importance = pd.Series(model.feature_importances_, index=feature_columns)
  importance = importance.sort_values(ascending=False).head(15).sort_values()

  fig_model = px.bar(
      importance,
      x=importance.values,
      y=importance.index,
      orientation="h",
      color=importance.values,
      color_continuous_scale=[
          "#FEF08A",
          "#F59E0B",
          "#C2410C",
      ],  # Soft Yellow to Amber to Warm Orange transition
  )
  fig_model.update_layout(
      title="Top 15 Most Important Factors",
      xaxis_title="Importance Score",
      yaxis_title="",
      coloraxis_showscale=False,
      height=520,
  )
  st.plotly_chart(fig_model, use_container_width=True)

  st.subheader("💡 Key Takeaways & Tips")
  top_features = importance.sort_values(ascending=False).head(5).index.tolist()
  custom_bullets = "".join([f"<li><b>{f}</b></li>" for f in top_features])

  st.markdown(
      f"""
        <div style="padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.05);">
        The top 5 factors that influence whether a candidate accepts an offer are:
        <ul>{custom_bullets}</ul>
        <b>Quick Tips for Recruiters:</b>
        <ul style="margin-top: 5px;">
            <li>Focus heavily on candidate technical skills and interview preparation to improve acceptance rates.</li>
            <li>Keep track of company tiers and market competition so you can make competitive job offers.</li>
            <li>Ensure clear communication early in the hiring process to prevent offer dropouts.</li>
        </ul>
        </div>
        """,
      unsafe_allow_html=True,
  )

# --------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------
st.markdown(
    """
    <hr style="border-color:rgba(255,255,255,0.15); margin-top:30px;">
    <p style="text-align:center; color:rgba(245,246,255,0.6); font-size:13px;">
    Job Acceptance Prediction System | Developed by MUHAMMAD RAFI S.M. | Powered by Streamlit & Python · 2026
    </p>
    """,
    unsafe_allow_html=True,
)