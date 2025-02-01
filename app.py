import streamlit as st
import pickle
import pandas as pd

# Load the model
@st.cache_resource
def load_model():
    with open("logistic_regression_model.pkl", "rb") as f:
        return pickle.load(f)


# Prediction function
def predict_depression(new_data):
    model = load_model()

    # Predefined order and names of features from training
    selected_features = [
        'Have you ever had suicidal thoughts ?', 'Academic Pressure',
        'Financial Stress', 'Dietary Habits', 'Study Satisfaction',
        'Family History of Mental Illness', 'Sleep Duration',
        'Work/Study Hours', 'Age', 'CGPA'
    ]

    # Load label encoders
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    # Load sleep duration mapping
    with open('sleep_duration_mapping.pkl', 'rb') as f:
        sleep_duration_mapping = pickle.load(f)

    # Copy data to avoid modifying the original
    processed_data = new_data.copy()

    # Encode categorical features safely
    for feature in ['Have you ever had suicidal thoughts ?', 'Dietary Habits', 'Family History of Mental Illness']:
        if feature in processed_data:
            processed_data[feature] = label_encoders[feature].transform([processed_data[feature]])[0]

    # Map Sleep Duration
    processed_data['Sleep Duration'] = processed_data['Sleep Duration'].map(sleep_duration_mapping).fillna(-1).astype(int)

    # Ensure input follows training feature order
    input_data = pd.DataFrame([{feature: processed_data[feature] for feature in selected_features}])

    # Debug
    print("Expected Features:", model.feature_names_in_)
    print("Current Features:", input_data.columns.tolist())

    # Predict
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    return {
        'prediction': prediction[0],  # Class label
        'probability': prediction_proba[0].max()  # Confidence score
    }


# Streamlit App
st.set_page_config(
    page_title="Depression Risk Assessment",
    # page_icon="favicon.png",
    # layout="wide",
)

st.title("Depression Risk Assessment With Machine Learning")

tab1, tab2, tab3 = st.tabs([
    "Risk Assessment",
    "Dashboard",
    "About"
])

with tab1:
    st.header("Depression Risk Assessment")

    # Collect user input
    name = st.text_input("Name")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, step=1)
    with col2:
        gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])

    col3, col4 = st.columns(2)
    with col3:
        profession = st.text_input("Profession")
    with col4:
        city = st.text_input("City of Domicile")

    # Prediction features
    academic_pressure = st.slider("Academic Pressure (5 = Most Pressure)", 0, 5, 0)
    work_pressure = st.slider("Work Pressure", 0, 5, 0)
    cgpa = st.number_input("CGPA (0.0 - 10.0)", min_value=0.0, max_value=10.0, step=0.1)
    study_satisfaction = st.slider("Study Satisfaction", 0, 10, 0)
    job_satisfaction = st.slider("Job Satisfaction", 0, 10, 0)

    sleep_duration = st.selectbox("Sleep Duration", [
        "Less than 5 hours",
        "5-6 hours",
        "7-8 hours",
        "More than 8 hours"
    ])

    family_history = st.radio("Family History of Mental Illness", ["No", "Yes"])
    feeling_depressed = st.radio("Are You Feeling Depressed?", ["No", "Yes"])

    dietary_habits = st.selectbox("Dietary Habits", [
        "Healthy", "Moderate", "Unhealthy", "Others"
    ])

    suicidal_thoughts = st.radio("Have You Ever Had Suicidal Thoughts?", ["No", "Yes"])

    work_study_hours = st.number_input("Work/Study Hours per Day", min_value=0, max_value=12, step=1)
    financial_stress = st.slider("Financial Stress", 1, 5, 1)

    if st.button("Predict Depression Risk"):
        # Prepare data for prediction
        prediction_data = pd.DataFrame({
            'Have you ever had suicidal thoughts ?': [suicidal_thoughts],
            'Academic Pressure': [academic_pressure],
            'Financial Stress': [financial_stress],
            'Dietary Habits': [dietary_habits],
            'Study Satisfaction': [study_satisfaction],
            'Family History of Mental Illness': [family_history],
            'Sleep Duration': [sleep_duration],
            'Work/Study Hours': [work_study_hours],
            'Age': [age],
            'CGPA': [cgpa]
        })

        result = predict_depression(prediction_data)

        # Display prediction
        if result['prediction'] == 1:
            st.error("Higher Risk of Depression Detected")
        else:
            st.success("Lower Risk of Depression Detected")
        st.write(f"Confidence: {result['probability']*100:.2f}%")

with tab2:
    st.header("Student Depression Dashboard")
    # st.components.v1.iframe("https://lookerstudio.google.com/reporting/8fee9fb4-8d82-4460-9943-61726e9887ad", height=900)
    st.markdown("[View Dashboard on Looker Studio](https://lookerstudio.google.com/reporting/8fee9fb4-8d82-4460-9943-61726e9887ad)")

with tab3:
    st.header("About This Project")
    st.markdown("""
    **Disclaimer:** This is a machine learning demo for depression risk assessment. The model is trained on a dataset of [student-depression-dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) and may not be suitable for all individuals.
    - **Not a Medical Diagnosis:** This tool CANNOT replace professional medical advice.
    - If you're experiencing depression or mental health challenges, please consult a healthcare professional.

    **Project Details:**
    - Developed by 4 Data Science students from Dicoding Indonesia Bootcamp 2024-2025
    - Purpose: Demonstrate machine learning application in mental health screening especially for student depression risk assessment.

    **Remember:** Your mental health is important. Seek professional help if needed.
    """
    )

    st.markdown("[Project Brief (ID)](https://drive.google.com/file/d/1umXcOXX_fML-y_UdQEZ2eGLYNk7C_VZi/view?usp=sharing)")
    st.markdown("[Github Repository](https://github.com/arguto1993/depression-risk-classification/tree/main)")