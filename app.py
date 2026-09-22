import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 1100px; }
    .main-title { font-size: 2.4rem; font-weight: 800; margin-bottom: 0; }
    .subtitle { color: #9aa0a6; font-size: 1.05rem; margin-top: 0.2rem; margin-bottom: 1.5rem; }
    .card {
        background-color: rgba(127, 127, 127, 0.06);
        border: 1px solid rgba(127, 127, 127, 0.15);
        border-radius: 14px;
        padding: 1.4rem 1.6rem 0.6rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(127, 127, 127, 0.08);
        border-radius: 12px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">📉 Customer Churn Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Fill in a customer\'s profile below to predict whether they are likely to churn, '
    'using a Logistic Regression model trained on the Telco Customer Churn dataset.</p>',
    unsafe_allow_html=True,
)

with st.form("churn_form"):
    st.markdown('<div class="card"><div class="card-title">👤 Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    gender = c1.selectbox("Gender", ["Male", "Female"])
    senior_citizen = c2.selectbox("Senior Citizen", ["No", "Yes"])
    partner = c3.selectbox("Partner", ["No", "Yes"])
    dependents = c4.selectbox("Dependents", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">💳 Account &amp; Billing</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    tenure = c1.slider("Tenure (months)", 0, 72, 12)
    contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    c1, c2 = st.columns(2)
    monthly_charges = c1.slider("Monthly Charges ($)", 0.0, 150.0, 70.0)
    total_charges = c2.slider("Total Charges ($)", 0.0, 9000.0, 1000.0)
    c1, c2 = st.columns(2)
    paperless_billing = c1.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = c2.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📡 Services</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    phone_service = c1.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = c2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = c3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    c1, c2, c3 = st.columns(3)
    online_security = c1.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = c2.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = c3.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    c1, c2, c3 = st.columns(3)
    tech_support = c1.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = c2.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = c3.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

if submitted:
    raw_input = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    encoded_input = pd.get_dummies(raw_input, drop_first=True)
    encoded_input = encoded_input.reindex(columns=columns, fill_value=0)

    scaled_input = scaler.transform(encoded_input)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.divider()
    col1, col2 = st.columns([1, 1.4])

    with col1:
        if prediction == 1:
            st.error("### ⚠️ Likely to CHURN")
        else:
            st.success("### ✅ Likely to STAY")
        st.metric("Churn Probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))

    with col2:
        st.markdown("**Key Signals**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tenure", f"{tenure} mo")
        c2.metric("Monthly Charges", f"${monthly_charges:,.0f}")
        c3.metric("Contract", contract)
        st.caption(
            "Longer tenure, two-year contracts, and lower monthly charges are associated "
            "with a lower likelihood of churn based on this model."
        )

with st.expander("About this model"):
    st.write(
        "Logistic Regression trained on the Telco Customer Churn dataset (7,043 customers). "
        "Outperformed a Random Forest baseline on accuracy, F1-score, and ROC-AUC "
        "(0.81 accuracy, 0.84 ROC-AUC)."
    )
