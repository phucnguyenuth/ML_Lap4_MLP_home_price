from src.data_loader import load_and_preprocess_data
from src.mlp_model import train_mlp_model
from src.evaluate_mlp import evaluate_mlp_model
from src.compare_models import compare_and_analyze_models

def main():
    print("==========================================================")
    print("   PRACTICE EXERCISE 4: MLP HOUSE PRICE PREDICTION PIPELINE")
    print("==========================================================")

    # Task 1 & 2: Tải và tiền xử lý dữ liệu
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data()

    # Task 3 & 4: Khởi tạo và huấn luyện mạng MLP
    mlp_model, _ = train_mlp_model(
        X_train_scaled, y_train, X_test_scaled, y_test,
        epochs=100, batch_size=32
    )

    # Task 5: Đánh giá và trực quan hóa kết quả MLP
    mlp_metrics = evaluate_mlp_model(mlp_model, X_test_scaled, y_test)

    # Task 6: So sánh với Linear Regression, Decision Tree, Random Forest
    compare_and_analyze_models(
        X_train_scaled, y_train, X_test_scaled, y_test,
        mlp_metrics=mlp_metrics
    )

    print("\n==========================================================")
    print("   HOÀN THÀNH TOÀN BỘ PIPELINE! KIỂM TRA THƯ MỤC figures/ & doc/")
    print("==========================================================")

if __name__ == "__main__":
    main()