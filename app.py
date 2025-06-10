import streamlit as st
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Dự đoán doanh số", layout="centered")

# Load scaler và mô hình
scaler = joblib.load("scaler.pkl")  # scaler phải fit trên tập có cột 'Sales'
model = load_model("model.h5", compile=False)

st.title("📊 Dự đoán doanh số bán hàng (RNN - với 14 ngày dữ liệu)")

uploaded_file = st.file_uploader("📁 Tải lên file CSV chứa 14 dòng dữ liệu", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.shape[0] != 14:
            st.error("❌ File CSV phải chứa đúng 14 dòng dữ liệu lịch sử.")
        else:
            # Scale dữ liệu đầu vào
            df_scaled = scaler.transform(df)

            # Định dạng lại shape cho RNN
            df_scaled = df_scaled.reshape(1, 14, df_scaled.shape[1])  # (batch, time, features)

            # Dự đoán
            pred = model.predict(df_scaled)

            # Tạo lại mảng có cùng số chiều như input scaler để inverse
            inverse_input = np.zeros((1, df.shape[1]))
            inverse_input[0, 0] = pred[0][0]  # giả sử cột 'Sales' là cột đầu tiên khi scaler fit

            original = scaler.inverse_transform(inverse_input)
            original_sales = original[0][0]

            st.success(f"🔮 Dự đoán doanh số (chuẩn hóa): {pred[0][0]:,.4f}")
            st.success(f"💰 Doanh số thực tế ước tính: {original_sales:,.2f} đơn vị")

    except Exception as e:
        st.error(f"⚠️ Lỗi khi xử lý file: {str(e)}")
