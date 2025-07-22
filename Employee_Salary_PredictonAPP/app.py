import streamlit as st
import pandas as pd
import random
from sklearn.linear_model import LinearRegression

st.title("💼 Employee Salary Predictor")

@st.cache_data
def load_data():
    num_samples = 100
    experience = [random.randint(0, 20) for _ in range(num_samples)]
    education_levels = [random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0] for _ in range(num_samples)]
    
    def calculate_salary(exp, edu):
        base = 25000
        exp_factor = exp * 3000
        edu_bonus = {0: 0, 1: 10000, 2: 20000}[edu]
        noise = random.randint(-3000, 3000)
        return base + exp_factor + edu_bonus + noise

    salaries = [calculate_salary(exp, edu) for exp, edu in zip(experience, education_levels)]

    education_map_reverse = {0: "Bachelor", 1: "Master", 2: "PhD"}
    df = pd.DataFrame({
        "Experience": experience,
        "Education_Level": [education_map_reverse[edu] for edu in education_levels],
        "Salary": salaries
    })
    return df

df = load_data()

education_mapping = {"Bachelor": 0, "Master": 1, "PhD": 2}
df["Education_Level"] = df["Education_Level"].map(education_mapping)
X = df[["Experience", "Education_Level"]]
y = df["Salary"]

model = LinearRegression()
model.fit(X, y)

st.subheader("📋 Enter Employee Details:")
exp = st.slider("Experience (Years)", 0, 30, 5)
edu = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
edu_num = education_mapping[edu]

if st.button("🔮 Predict Salary"):
    prediction = model.predict([[exp, edu_num]])
    st.success(f"Estimated Salary: ₹{int(prediction[0]):,}")
