import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src.data_loader import FEATURE_NAMES

def predict_sample_house():
    scaler_path = "models/scaler.joblib"
    model_path = "models/mlp_model.keras"

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print("Lỗi: Chưa tìm thấy mô hình! Hãy chạy 'python run_pipeline.py' trước.")
        return

    scaler = joblib.load(scaler_path)
    model = load_model(model_path)

    # Thông tin giả lập của một căn nhà tại Boston (13 đặc trưng)
    sample_house = {
        "CRIM": 0.02731,     # Tỷ lệ tội phạm bình quân đầu người
        "ZN": 0.0,           # Tỷ lệ đất ở quy hoạch lô lớn
        "INDUS": 7.07,       # Tỷ lệ diện tích kinh doanh phi bán lẻ
        "CHAS": 0.0,         # Giáp sông Charles (1 = Có, 0 = Không)
        "NOX": 0.469,        # Nồng độ khí oxit nitơ
        "RM": 6.421,         # Số phòng trung bình mỗi căn hộ
        "AGE": 78.9,         # Tỷ lệ nhà xây trước năm 1940
        "DIS": 4.9671,       # Khoảng cách tới các trung tâm việc làm Boston
        "RAD": 2.0,          # Chỉ số tiếp cận đường cao tốc
        "TAX": 242.0,        # Thuế suất bất động sản trên $10,000
        "PTRATIO": 17.8,     # Tỷ lệ học sinh / giáo viên
        "B": 396.90,         # Chỉ số nhân khẩu học
        "LSTAT": 9.14        # Tỷ lệ dân cư thu nhập thấp (%)
    }

    input_df = pd.DataFrame([sample_house])[FEATURE_NAMES]
    input_scaled = scaler.transform(input_df.values)

    predicted_value = model.predict(input_scaled, verbose=0)[0][0]
    predicted_usd = predicted_value * 1000  # Đơn vị gốc của Boston Housing là $1,000

    print("\n--- DEMO DỰ ĐOÁN GIÁ NHÀ BẰNG MÔ HÌNH MLP ---")
    print("Thông số đầu vào căn nhà mẫu:")
    for k, v in sample_house.items():
        print(f"  - {k}: {v}")
    print(f"\n=> Giá nhà dự đoán (MEDV): {predicted_value:.2f} ($1000s) ~ ${predicted_usd:,.2f} USD")

if __name__ == "__main__":
    predict_sample_house()