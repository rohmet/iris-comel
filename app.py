import streamlit as st
import pickle
import numpy as np

with open("model-iris.pkcls", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

st.markdown("""
<style>
    .main {
        background: linear-gradient(to right, #f7faff, #fff4fb);
    }
    .stButton>button {
        background-color: #ff86c1;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.3em;
        border: none;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff5ca8;
        color: white;
        transform: scale(1.03);
    }
    .input-card {
        background: white;
        padding: 20px 30px;
        border-radius: 12px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
        border: 1px solid #ffb4d4;
    }
</style>
""", unsafe_allow_html=True)

label_map = {
    0.0: "Iris-Setosa",
    1.0: "Iris-Versicolor",
    2.0: "Iris-Virginica"
}

st.title("*Iris Flower Classifier*")

with st.container():
    st.markdown("### Masukkan Data Bunga 🌿")
    
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    with col2:
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)    

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Prediksi 🌼"):
    prediction = model(features)[0]
    flower_name = label_map.get(float(prediction), "Tidak diketahui")

    st.subheader("Hasil Prediksi 🌷")
    st.success(f"**Jenis Bunga:** {flower_name}  \n**Label Angka:** {prediction}")
