import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }

        .title {
            font-size: 40px;
            font-weight: bold;
            color: #1f4e79;
            text-align: center;
        }

        .subtitle {
            text-align: center;
            color: #555;
            font-size: 18px;
        }

        .card {
            padding: 20px;
            border-radius: 15px;
            background-color: white;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }

        .success-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #e8f5e9;
            color: #2e7d32;
            font-size: 20px;
            font-weight: bold;
        }

        .danger-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #ffebee;
            color: #c62828;
            font-size: 20px;
            font-weight: bold;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# Load model
with open(
    'Diabetes-Risk-Prediction-System-Using-Machine-Learning/best_model.pkl',
    'rb'
) as file:
    model = pickle.load(file)


# Header
st.markdown(
    "<div class='title'>🩺 Diabetes Risk Prediction System</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='subtitle'>
    Machine Learning based web application to predict diabetes risk
    based on patient health information.
    </div>
    """,
    unsafe_allow_html=True
)


st.write("")


# Sidebar
st.sidebar.title("Patient Information")
st.sidebar.write("Enter patient data below:")


gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

age = st.sidebar.slider(
    "Age",
    0,
    120,
    30
)

hypertension = st.sidebar.selectbox(
    "Hypertension (0 = No, 1 = Yes)",
    [0, 1]
)

heart_disease = st.sidebar.selectbox(
    "Heart Disease (0 = No, 1 = Yes)",
    [0, 1]
)


smoking_history = st.sidebar.selectbox(
    "Smoking History",
    [
        "No Info",
        "current",
        "ever",
        "former",
        "never",
        "not current"
    ]
)


bmi = st.sidebar.slider(
    "BMI",
    10.0,
    50.0,
    25.0
)


hba1c_level = st.sidebar.slider(
    "HbA1c Level",
    2.0,
    15.0,
    5.5
)


blood_glucose_level = st.sidebar.slider(
    "Blood Glucose Level",
    50,
    300,
    120
)



# Encoding (same logic)
gender_encoded = [
    1 if gender == "Female" else 0,
    1 if gender == "Male" else 0,
    1 if gender == "Other" else 0
]


smoking_history_encoded = [
    1 if smoking_history == "No Info" else 0,
    1 if smoking_history == "current" else 0,
    1 if smoking_history == "ever" else 0,
    1 if smoking_history == "former" else 0,
    1 if smoking_history == "never" else 0,
    1 if smoking_history == "not current" else 0
]


# Prepare input
input_data = pd.DataFrame(
    [
        [
            age,
            hypertension,
            heart_disease,
            bmi,
            hba1c_level,
            blood_glucose_level
        ]
        + gender_encoded
        + smoking_history_encoded
    ],
    columns=[
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "gender_Female",
        "gender_Male",
        "gender_Other",
        "smoking_history_No Info",
        "smoking_history_current",
        "smoking_history_ever",
        "smoking_history_former",
        "smoking_history_never",
        "smoking_history_not current"
    ]
)



# Display input
st.markdown(
    "<div class='card'>",
    unsafe_allow_html=True
)

st.subheader("📋 Patient Data Preview")

st.dataframe(
    input_data,
    use_container_width=True
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)



# Prediction
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    prediction = model.predict(input_data)[0]

    prediction_proba = model.predict_proba(input_data)[0][1]


    st.write("")


    if prediction == 1:

        st.markdown(
            f"""
            <div class='danger-box'>
            ⚠️ Prediction Result: Positive for Diabetes
            <br><br>
            Risk Probability: {prediction_proba*100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.markdown(
            f"""
            <div class='success-box'>
            ✅ Prediction Result: Negative for Diabetes
            <br><br>
            Healthy Probability: {(1-prediction_proba)*100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )
