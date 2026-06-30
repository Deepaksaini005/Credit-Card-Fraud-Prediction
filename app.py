import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main{
    background:#F5F7FA;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:#0F172A;
}

.subtitle{
    color:#6B7280;
    font-size:18px;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,.10);
}

div.stButton > button{
    width:100%;
    height:50px;
    font-size:18px;
    font-weight:bold;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    "<p class='title'>💳 Credit Card Fraud Detection</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>CatBoost Machine Learning Model</p>",
    unsafe_allow_html=True
)
st.divider()

# --------------------------------------------------
# Input Form
# --------------------------------------------------

st.subheader("Enter Transaction Details")

left, right = st.columns(2)

with left:

    Time = st.number_input("Time", value=0.0)

    V1 = st.number_input("V1", value=0.0)
    V2 = st.number_input("V2", value=0.0)
    V3 = st.number_input("V3", value=0.0)
    V4 = st.number_input("V4", value=0.0)
    V5 = st.number_input("V5", value=0.0)
    V6 = st.number_input("V6", value=0.0)
    V7 = st.number_input("V7", value=0.0)
    V8 = st.number_input("V8", value=0.0)
    V9 = st.number_input("V9", value=0.0)
    V10 = st.number_input("V10", value=0.0)
    V11 = st.number_input("V11", value=0.0)
    V12 = st.number_input("V12", value=0.0)
    V13 = st.number_input("V13", value=0.0)
    V14 = st.number_input("V14", value=0.0)

with right:

    V15 = st.number_input("V15", value=0.0)
    V16 = st.number_input("V16", value=0.0)
    V17 = st.number_input("V17", value=0.0)
    V18 = st.number_input("V18", value=0.0)
    V19 = st.number_input("V19", value=0.0)
    V20 = st.number_input("V20", value=0.0)
    V21 = st.number_input("V21", value=0.0)
    V22 = st.number_input("V22", value=0.0)
    V23 = st.number_input("V23", value=0.0)
    V24 = st.number_input("V24", value=0.0)
    V25 = st.number_input("V25", value=0.0)
    V26 = st.number_input("V26", value=0.0)
    V27 = st.number_input("V27", value=0.0)
    V28 = st.number_input("V28", value=0.0)

    Amount = st.number_input("Amount", value=0.0)

predict = st.button("🚀 Predict Transaction")

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------

if predict:

    input_data = np.array([[
        Time,
        V1, V2, V3, V4, V5, V6, V7,
        V8, V9, V10, V11, V12, V13,
        V14, V15, V16, V17, V18,
        V19, V20, V21, V22, V23,
        V24, V25, V26, V27, V28,
        Amount
    ]])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    genuine_prob = probability[0] * 100
    fraud_prob = probability[1] * 100

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("🚨 Fraudulent Transaction Detected!")

        st.progress(min(fraud_prob / 100, 1.0))

    else:

        st.success("✅ Genuine Transaction")

        st.progress(min(genuine_prob / 100, 1.0))

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Genuine Probability",
            f"{genuine_prob:.2f}%"
        )

    with col2:
        st.metric(
            "Fraud Probability",
            f"{fraud_prob:.2f}%"
        )

    st.divider()

    # -----------------------------
    # Prediction Probability
    # -----------------------------

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame({

        "Class":[
            "Genuine",
            "Fraud"
        ],

        "Probability":[
            genuine_prob,
            fraud_prob
        ]

    })

    st.bar_chart(
        probability_df.set_index("Class")
    )

    # -----------------------------
    # Input Summary
    # -----------------------------

    st.subheader("Transaction Summary")

    summary = pd.DataFrame({

        "Feature":[

            "Time","V1","V2","V3","V4","V5","V6","V7",
            "V8","V9","V10","V11","V12","V13",
            "V14","V15","V16","V17","V18","V19",
            "V20","V21","V22","V23","V24","V25",
            "V26","V27","V28","Amount"

        ],

        "Value":input_data.flatten()

    })

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.divider()

    # -----------------------------
    # Download Report
    # -----------------------------

    report = pd.DataFrame({

        "Parameter":[
            "Prediction",
            "Genuine Probability",
            "Fraud Probability"
        ],

        "Value":[

            "Fraud" if prediction==1 else "Genuine",

            f"{genuine_prob:.2f}%",

            f"{fraud_prob:.2f}%"

        ]

    })

    csv = report.to_csv(index=False)

    st.download_button(

        "📥 Download Prediction Report",

        csv,

        "prediction_report.csv",

        "text/csv",

        use_container_width=True

    )

    # --------------------------------------------------
# Feature Importance
# --------------------------------------------------

st.divider()

st.subheader("⭐ Top 10 Important Features")

feature_names = [

    "Time","V1","V2","V3","V4","V5","V6","V7",
    "V8","V9","V10","V11","V12","V13",
    "V14","V15","V16","V17","V18","V19",
    "V20","V21","V22","V23","V24","V25",
    "V26","V27","V28","Amount"

]

importance = pd.DataFrame({

    "Feature":feature_names,

    "Importance":model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

st.dataframe(

    importance.head(10),

    use_container_width=True

)

st.bar_chart(

    importance.head(10).set_index("Feature")

)


# --------------------------------------------------
# About Model
# --------------------------------------------------

st.divider()

st.subheader("🤖 Model Information")

st.info("""

**Algorithm : CatBoost Classifier**

Accuracy : **99.95%**

Precision : **97.26%**

Recall : **74.74%**

F1 Score : **84.52%**

""")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.markdown("""

<center>

## 💳 Credit Card Fraud Detection Dashboard

Developed by **Deepak Saini**

Python • CatBoost • Streamlit • Scikit-Learn

</center>

""",unsafe_allow_html=True)