import streamlit as st
import pandas as pd
from model import train_and_predict
from utils import create_features

st.title("🔮 Six-Digit Lottery Predictor")
st.write("อัปโหลดไฟล์ `six_digit_numbers.csv` ที่มีเลข 6 หลักย้อนหลัง")

uploaded_file = st.file_uploader("📤 อัปโหลดไฟล์ CSV", type=["csv"])
model_name = st.selectbox("🧠 เลือกโมเดล", ["XGBoost", "Prophet"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'numbers' not in data.columns:
        st.error("❌ ต้องมีคอลัมน์ชื่อ 'numbers' ในไฟล์ CSV")
    else:
        st.write("📊 ข้อมูลที่อัปโหลด:")
        st.dataframe(data.tail(10))

        if model_name == "XGBoost":
            X, y = create_features(data)
            predicted = train_and_predict("XGBoost", X, y)
        elif model_name == "Prophet":
            predicted = train_and_predict("Prophet", None, None, df_raw=data)

        st.success(f"🎯 โมเดล {model_name} ทำนายเลขถัดไป: **{predicted:06d}**")
