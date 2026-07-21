import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Load Model and Label Encoders
# -----------------------------------
model = joblib.load("salary_prediction_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Helper function to get encoder keys safely regardless of case
def get_encoder_key(key_name):
    for k in label_encoders.keys():
        if k.lower() == key_name.lower():
            return k
    return None

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Salary Prediction System",
    page_icon="💼",
    layout="wide"
)

# -----------------------------------
# Custom CSS
# -----------------------------------
st.markdown("""
<style>

.main{
    background-color:#F4F6F8;
}

h1{
    color:#0B5394;
    text-align:center;
}

h2{
    color:#0B5394;
}

h3{
    color:#1F4E79;
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:50px;
    font-size:18px;
    background:#0B5394;
    color:white;
}

.stButton>button:hover{
    background:#1C86EE;
    color:white;
}

div[data-testid="stMetric"]{
    border:2px solid #0B5394;
    border-radius:10px;
    padding:15px;
    background:white;
}

</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image(
    "https://img.icons8.com/color/96/businessman.png",
    width=100
)

st.sidebar.title("Salary Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "💼 Predict Salary",
        "📖 About Project"
    ]
)

# -----------------------------------
# Home Page
# -----------------------------------
if page == "🏠 Home":

    st.title("💼 Salary Prediction Using Ensemble Learning")

    st.markdown("---")

    st.header("Project Overview")

    st.write("""
This project predicts employee salary using Ensemble Machine Learning techniques.

The application is developed using Python, Streamlit and Scikit-Learn.
The prediction model is trained on employee salary data and compares multiple
ensemble algorithms to identify the best-performing model.

The deployed model provides fast and accurate salary predictions based on employee
details entered by the user.
""")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Project Features")

        st.success("""
✔ Salary Prediction

✔ Ensemble Learning

✔ Random Forest Model

✔ Interactive Dashboard

✔ User Friendly Interface

✔ Real-Time Prediction
""")

    with col2:

        st.subheader("Technologies Used")

        st.info("""
• Python

• Streamlit

• Pandas

• NumPy

• Scikit-Learn

• Joblib

• Ensemble Learning
""")

    st.markdown("---")

    st.subheader("Workflow")

    st.write("""
1. User enters employee details.

2. Data is encoded using Label Encoders.

3. Input is passed to the trained Random Forest model.

4. Model predicts the salary.

5. Predicted salary is displayed instantly.
""")

# -----------------------------------
# Salary Prediction Page
# -----------------------------------
elif page == "💼 Predict Salary":

    st.title("💼 Employee Salary Prediction")

    st.write("Fill in the employee details below and click **Predict Salary**.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Resolve Encoder Keys safely
    exp_key = get_encoder_key("experience_level")
    emp_type_key = get_encoder_key("employment_type")
    job_key = get_encoder_key("job_title")
    curr_key = get_encoder_key("salary_currency")
    res_key = get_encoder_key("employee_residence")
    loc_key = get_encoder_key("company_location")
    size_key = get_encoder_key("company_size")

    with col1:

        work_year = st.number_input(
            "Work Year",
            min_value=2020,
            max_value=2035,
            value=2024
        )

        experience_level = st.selectbox(
            "Experience Level",
            label_encoders[exp_key].classes_ if exp_key else []
        )

        employment_type = st.selectbox(
            "Employment Type",
            label_encoders[emp_type_key].classes_ if emp_type_key else []
        )

        job_title = st.selectbox(
            "Job Title",
            label_encoders[job_key].classes_ if job_key else []
        )

    with col2:

        remote_ratio = st.selectbox(
            "Remote Ratio (%)",
            [0, 50, 100],
            help="0: No remote work, 50: Hybrid, 100: Fully Remote"
        )

        salary_currency = st.selectbox(
            "Salary Currency",
            label_encoders[curr_key].classes_ if curr_key else []
        )

        employee_residence = st.selectbox(
            "Employee Residence",
            label_encoders[res_key].classes_ if res_key else []
        )

        company_location = st.selectbox(
            "Company Location",
            label_encoders[loc_key].classes_ if loc_key else []
        )

        company_size = st.selectbox(
            "Company Size",
            label_encoders[size_key].classes_ if size_key else []
        )

    st.markdown("---")

    if st.button("Predict Salary"):

        input_df = pd.DataFrame({
            "work_year": [work_year],
            "experience_level": [label_encoders[exp_key].transform([experience_level])[0]] if exp_key else [0],
            "employment_type": [label_encoders[emp_type_key].transform([employment_type])[0]] if emp_type_key else [0],
            "job_title": [label_encoders[job_key].transform([job_title])[0]] if job_key else [0],
            "salary_currency": [label_encoders[curr_key].transform([salary_currency])[0]] if curr_key else [0],
            "employee_residence": [label_encoders[res_key].transform([employee_residence])[0]] if res_key else [0],
            "remote_ratio": [remote_ratio],
            "company_location": [label_encoders[loc_key].transform([company_location])[0]] if loc_key else [0],
            "company_size": [label_encoders[size_key].transform([company_size])[0]] if size_key else [0]
        })

        # Match feature order expected by model
        if hasattr(model, "feature_names_in_"):
            for feature in model.feature_names_in_:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df.values)

        st.success("Prediction Successful!")

        st.metric(
            label="Predicted Salary (USD)",
            value=f"${prediction[0]:,.2f}"
        )

        st.balloons()

# -----------------------------------
# About Project Page
# -----------------------------------
elif page == "📖 About Project":

    st.title("📖 About This Project")

    st.markdown("---")

    st.header("Project Title")

    st.write("""
**Salary Prediction Using Ensemble Learning**
""")

    st.markdown("---")

    st.header("Project Objective")

    st.write("""
The objective of this project is to predict employee salaries using
Machine Learning techniques. The project applies Ensemble Learning
algorithms to improve prediction accuracy and help organizations
estimate employee salaries based on job-related information.
""")

    st.markdown("---")

    st.header("Machine Learning Algorithm")

    st.info("""
✔ Random Forest Regressor

✔ Ensemble Learning

✔ Trained using Scikit-Learn
""")

    st.markdown("---")

    st.header("Input Features")

    st.write("""
• Work Year

• Experience Level

• Employment Type

• Job Title

• Salary Currency

• Employee Residence

• Company Location

• Company Size
""")

    st.markdown("---")

    st.header("Software Used")

    st.write("""
• Python

• Streamlit

• Pandas

• NumPy

• Scikit-Learn

• Joblib

• VS Code
""")

    st.markdown("---")

    st.header("Project Outcome")

    st.success("""
The developed web application predicts employee salary quickly
and accurately using a trained Random Forest model.
The system provides an easy-to-use interface for users to
enter employee details and obtain salary predictions instantly.
""")

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:gray;'>
        Developed by <b>Gurmannat Singh</b><br>
        Salary Prediction Using Ensemble Learning<br>
        Powered by Streamlit & Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True
)