import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Page Config
st.set_page_config(
    page_title="Glaucoma Risk Analytics & Prediction",
    page_icon="👁️",
    layout="wide"
)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_glaucoma_risk_dataset.csv")
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Error loading 'cleaned_glaucoma_risk_dataset.csv'. Please make sure the CSV file is in the same directory as app.py.")
    st.stop()

# Sidebar Navigation
st.sidebar.title("👁️ Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Patient Explorer", "Glaucoma Predictor"])

# ---------------------------------------------------------
# PAGE 1: DASHBOARD OVERVIEW
# ---------------------------------------------------------
if page == "Dashboard Overview":
    st.title("👁️ Glaucoma Risk Analytics Dashboard")
    st.markdown("Explore distribution metrics and risk stages across the dataset.")

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{len(df):,}")
    col2.metric("Glaucoma Cases", f"{df['glaucoma_label'].sum():,}")
    col3.metric("Avg Age", f"{df['age'].mean():.1f} yrs")
    col4.metric("Avg IOP (mmHg)", f"{df['iop_mmhg'].mean():.2f}")

    st.markdown("---")

    # Visualizations
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Stage Distribution")
        fig_stage = px.pie(
            df, 
            names='risk_stage', 
            title="Patient Breakdown by Risk Stage",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_stage, use_container_width=True)

    with col_right:
        st.subheader("IOP vs. RNFL Thickness")
        fig_scatter = px.scatter(
            df,
            x="iop_mmhg",
            y="rnfl_thickness_um",
            color="risk_stage",
            hover_data=["patient_id", "age", "gender"],
            labels={"iop_mmhg": "IOP (mmHg)", "rnfl_thickness_um": "RNFL Thickness (µm)"},
            title="IOP vs. RNFL Thickness by Risk Stage"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Feature Comparisons Across Risk Stages")
    feature = st.selectbox("Select Feature to Compare", ["iop_mmhg", "vertical_cdr", "rnfl_thickness_um", "visual_field_md_db", "age"])
    fig_box = px.box(df, x="risk_stage", y=feature, color="risk_stage", title=f"{feature.upper()} Distribution by Stage")
    st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: PATIENT EXPLORER
# ---------------------------------------------------------
elif page == "Patient Explorer":
    st.title("📋 Patient Record Explorer")

    # Filters
    st.sidebar.subheader("Filter Data")
    gender_filter = st.sidebar.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())
    stage_filter = st.sidebar.multiselect("Risk Stage", df['risk_stage'].unique(), default=df['risk_stage'].unique())
    age_range = st.sidebar.slider("Age Range", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))

    filtered_df = df[
        (df['gender'].isin(gender_filter)) &
        (df['risk_stage'].isin(stage_filter)) &
        (df['age'].between(age_range[0], age_range[1]))
    ]

    st.dataframe(filtered_df, use_container_width=True)
    st.caption(f"Showing {len(filtered_df)} of {len(df)} records.")

# ---------------------------------------------------------
# PAGE 3: GLAUCOMA PREDICTOR
# ---------------------------------------------------------
elif page == "Glaucoma Predictor":
    st.title("🤖 Real-Time Glaucoma Risk Assessment")
    st.markdown("Enter patient diagnostic metrics to predict glaucoma likelihood.")

    # Train a quick ML Model
    @st.cache_resource
    def train_model(data):
        X = data[['age', 'iop_mmhg', 'vertical_cdr', 'rnfl_thickness_um', 'visual_field_md_db', 'family_history']]
        y = data['glaucoma_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return model, acc

    model, accuracy = train_model(df)
    st.sidebar.success(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=60)
        iop = st.number_input("IOP - Intraocular Pressure (mmHg)", min_value=5.0, max_value=50.0, value=19.5)
        vertical_cdr = st.number_input("Vertical Cup-to-Disc Ratio (CDR)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

    with col2:
        rnfl = st.number_input("RNFL Thickness (µm)", min_value=30.0, max_value=150.0, value=85.0)
        visual_field = st.number_input("Visual Field MD (dB)", min_value=-30.0, max_value=5.0, value=-5.0)
        family_history = st.selectbox("Family History of Glaucoma", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    if st.button("Assess Risk", type="primary"):
        input_data = pd.DataFrame([[age, iop, vertical_cdr, rnfl, visual_field, family_history]], 
                                  columns=['age', 'iop_mmhg', 'vertical_cdr', 'rnfl_thickness_um', 'visual_field_md_db', 'family_history'])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.markdown("---")
        st.subheader("Assessment Result")
        
        if prediction == 1:
            st.error(f"⚠️ **High Risk of Glaucoma** (Probability: {probability * 100:.1f}%)")
        else:
            st.success(f"✅ **Low Risk / Normal** (Glaucoma Probability: {probability * 100:.1f}%)")
