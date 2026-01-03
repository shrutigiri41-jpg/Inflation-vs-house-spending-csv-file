import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Simple Salary Prediction App")

# Read CSV
df = pd.read_csv("salary.csv")
st.subheader("Salary Dataset")
st.dataframe(df)

# Model
model = LinearRegression()
model.fit(df[["Experience"]], df["Salary"])

# Input
exp = st.number_input("Enter Experience (in years):", 1, 50)

# Prediction
pred = model.predict([[exp]])

st.subheader("Predicted Salary")
st.write(int(pred[0]))
