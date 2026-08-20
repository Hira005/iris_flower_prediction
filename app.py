import streamlit as st
import joblib
import numpy as np

try:
    model = joblib.load('iris_model.pkl')
    scaler=joblib.load('iris_scaler.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure iris_model.pkl and iris_scaler.pkl are in this folder.")
    st.stop()

st.title("Iris Flower Prediction")
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)),
                           url("https://cdn.mos.cms.futurecdn.net/Pg3t8pRx7sodr5oRv3NqDe.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
sepal_length = st.number_input("Sepal Length (cm)")
sepal_width = st.number_input("Sepal Width (cm)")
petal_length = st.number_input("Petal Length (cm)")
petal_width = st.number_input("Petal Width (cm)")

def validate_input(sepal_length, sepal_width, petal_length, petal_width):
    errors = []
    for name, value in [("Sepal Length", sepal_length), ("Sepal Width", sepal_width),
                         ("Petal Length", petal_length), ("Petal Width", petal_width)]:
        if value<0:
            errors.append(f"{name} cannot be negative")
    if not(3.5<=sepal_length<=8.5):
        errors.append("Sepal Length seems out of realistic range (3.5–8.5 cm).")
    if not (1.5 <= sepal_width <= 5.0):
        errors.append("Sepal Width seems out of realistic range (1.5–5.0 cm).")
    if not (0.5 <= petal_length <= 7.5):
        errors.append("Petal Length seems out of realistic range (0.5–7.5 cm).")
    if not (0.05 <= petal_width <= 3.0):
        errors.append("Petal Width seems out of realistic range (0.05–3.0 cm).")

    return errors
if st.button("Predict"):
    errors = validate_input(sepal_length, sepal_width, petal_length, petal_width)

    if errors:
        for e in errors:
            st.error(e)
    else:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
    
        species_names = ['Setosa', 'Versicolor', 'Virginica']
        predicted_species = species_names[prediction[0]]
    
        st.write("Predicted species:", predicted_species)