import streamlit as st
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Dự đoán doanh số", layout="centered")

# Load scaler và mô hình
scaler = joblib.load("scaler.pkl")  # scaler phải fit trên tập có cột 'Sales'
model = load_model("model.h5", compile=False)

st.title("📊 Dự đoán doanh số bán hàng (RNN - với dữ liệu 14 ngày gần nhất)")

uploaded_file = st.file_uploader("📁 Tải lên file CSV chứa dữ liệu bán hàng", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Kiểm tra cột Date
        if 'Date' not in df.columns:
            st.error("❌ File phải chứa cột 'Date'.")
        else:
            # Chuyển cột Date về datetime
            df['Date'] = pd.to_datetime(df['Date'])

            # Sắp xếp theo ngày tăng dần
            df = df.sort_values('Date')

            # Lấy 14 ngày gần nhất (theo giá trị duy nhất của ngày)
            unique_dates = df['Date'].dt.normalize().unique()
            if len(unique_dates) < 14:
                st.error("❌ Dữ liệu phải chứa ít nhất 14 ngày khác nhau.")
            else:
                last_14_days = unique_dates[-14:]

                # Lọc dữ liệu tương ứng
                df_14 = df[df['Date'].dt.normalize().isin(last_14_days)]

                # Kiểm tra đủ 14 ngày
                if df_14['Date'].dt.normalize().nunique() != 14:
                    st.error("❌ Dữ liệu không đủ 14 ngày liên tiếp.")
                else:
                    # Xóa cột Date nếu không nằm trong features
                    if 'Date' in df_14.columns:
                        df_14 = df_14.drop(columns=['Date'])

                    # Scale dữ liệu
                    df_scaled = scaler.transform(df_14)

                    # Reshape cho RNN
                    df_scaled = df_scaled.reshape(1, 14, df_scaled.shape[1])

                    # Dự đoán
                    pred = model.predict(df_scaled)

                    # Inverse transform
                    inverse_input = np.zeros((1, df_14.shape[1]))
                    inverse_input[0, 0] = pred[0][0]  # giả định cột Sales là đầu tiên

                    original = scaler.inverse_transform(inverse_input)
                    original_sales = original[0][0]

                    st.success(f"💰 Doanh số thực tế ước tính: {original_sales:,.2f} $")

    except Exception as e:
        st.error(f"⚠️ Lỗi khi xử lý file: {str(e)}")
