import streamlit as st
import pickle
import numpy as np

# Page Config
st.set_page_config(
    page_title="KNN Prediction App",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for colors
st.markdown("""
<style>
body {
    background-color: #f0f8ff;
}
h1 {
    color: #4B0082;
    text-align: center;
}
h2 {
    color: #008080;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 18px;
}
.stButton>button:hover {
    background-color: #ff0000;
}
.result-box {
    background-color: #d1ffd6;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🌟 KNN Prediction App</h1>", unsafe_allow_html=True)

# Load Model
with open("knn_model.pkl", "rb") as file:
    model = pickle.load(file)

# Sidebar
st.sidebar.header("Input Values")

f1 = st.sidebar.number_input("Feature 1")
f2 = st.sidebar.number_input("Feature 2")
f3 = st.sidebar.number_input("Feature 3")
f4 = st.sidebar.number_input("Feature 4")

# Predict Button
if st.sidebar.button("Predict"):
    input_data = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(input_data)

    st.markdown(
        f"<div class='result-box'>Prediction Result: {prediction[0]}</div>",
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ using Streamlit</center>", unsafe_allow_html=True)
