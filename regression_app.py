import streamlit as st
import pandas as pd
import tensorflow.keras as keras
import pickle

# Load the regression model
model = keras.models.load_model("salary_regression_model.h5")

# Load the scaler and encoders for regression
with open("scaler_regression.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder_gender_regression.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("one_hot_encoder_geo_regression.pkl", "rb") as f:
    one_hot_encoder_geo = pickle.load(f)


# Streamlit app
st.title("Salary Prediction")
st.write("Enter customer details to predict estimated salary.")

# Input fields
geography = st.selectbox("Geography", one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=92)
balance = st.number_input("Balance", value=0.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
tenure = st.slider("Tenure (in years)", min_value=0, max_value=10)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
exited = st.selectbox("Exited (Churned)", [0, 1])

# Prepare the input data
input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "Exited": [exited],
    }
)

# One-hot encode Geography
geo_encoded = one_hot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(
    geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(["Geography"])
)

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict salary
predicted_salary = model.predict(input_data_scaled)
predicted_salary_value = predicted_salary[0][0]

st.write(f"**Predicted Estimated Salary: ${predicted_salary_value:,.2f}**")

# Display input summary
st.subheader("Input Summary:")
col1, col2 = st.columns(2)

with col1:
    st.write(f"**Geography:** {geography}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Age:** {age}")
    st.write(f"**Tenure:** {tenure} years")
    st.write(f"**Credit Score:** {credit_score}")

with col2:
    st.write(f"**Balance:** ${balance:,.2f}")
    st.write(f"**Number of Products:** {num_of_products}")
    st.write(f"**Has Credit Card:** {'Yes' if has_cr_card else 'No'}")
    st.write(f"**Is Active Member:** {'Yes' if is_active_member else 'No'}")
    st.write(f"**Exited (Churned):** {'Yes' if exited else 'No'}")

# Add some insights
st.subheader("Insights:")
if predicted_salary_value > 100000:
    st.success(
        "💰 High salary prediction - this customer profile suggests above-average earning potential!"
    )
elif predicted_salary_value > 50000:
    st.info(
        "💼 Moderate salary prediction - this profile indicates average earning potential."
    )
else:
    st.warning(
        "📊 Lower salary prediction - this profile suggests below-average earning potential."
    )
