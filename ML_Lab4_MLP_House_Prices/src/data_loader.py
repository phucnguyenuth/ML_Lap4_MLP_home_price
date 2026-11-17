import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

def _download_boston_dataset() -> pd.DataFrame:
    """Tải dữ liệu Boston Housing từ các nguồn HTTPS ổn định (OpenML / GitHub)."""
    # Nguồn 1: OpenML chính thức tích hợp sẵn trong scikit-learn (HTTPS)
    try:
        print("[Task 2] Đang tải dữ liệu Boston Housing từ OpenML (HTTPS)...")
        boston = fetch_openml(data_id=531, as_frame=True, parser="auto")
        df = boston.frame.copy()
        df.columns = [col.upper() for col in df.columns]
        return df[FEATURE_NAMES + ["MEDV"]].astype(float)
    except Exception as e:
        print(f"[Task 2] OpenML gián đoạn ({e}), đang chuyển sang nguồn dự phòng GitHub...")

    # Nguồn 2: Kho lưu trữ dự phòng trên GitHub (HTTPS)
    github_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(github_url)
    df.columns = [col.upper() for col in df.columns]
    return df[FEATURE_NAMES + ["MEDV"]].astype(float)

def load_and_preprocess_data(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Task 1 & 2: Tải dữ liệu Boston Housing, tách X/y, chia Train/Test và chuẩn hóa StandardScaler.
    """
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    raw_csv_path = os.path.join(raw_dir, "boston_housing.csv")

    if os.path.exists(raw_csv_path):
        print(f"[Task 2] Đang nạp dữ liệu từ file cục bộ: {raw_csv_path}")
        df = pd.read_csv(raw_csv_path)
    else:
        df = _download_boston_dataset()
        df.to_csv(raw_csv_path, index=False)
        print(f"[Task 2] Đã lưu dữ liệu thô tại: {raw_csv_path}")

    X = df[FEATURE_NAMES].values
    y = df["MEDV"].values

    # Chia tập Train (80%) và Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Chuẩn hóa đặc trưng (Fit trên Train, Transform trên cả Train & Test để tránh Data Leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Lưu scaler và dữ liệu đã xử lý
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    pd.DataFrame(X_train_scaled, columns=FEATURE_NAMES).to_csv(
        os.path.join(processed_dir, "X_train_scaled.csv"), index=False
    )
    pd.DataFrame(X_test_scaled, columns=FEATURE_NAMES).to_csv(
        os.path.join(processed_dir, "X_test_scaled.csv"), index=False
    )
    pd.Series(y_train, name="MEDV").to_csv(
        os.path.join(processed_dir, "y_train.csv"), index=False
    )
    pd.Series(y_test, name="MEDV").to_csv(
        os.path.join(processed_dir, "y_test.csv"), index=False
    )

    print(f"[Task 2] Hoàn tất! Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data()