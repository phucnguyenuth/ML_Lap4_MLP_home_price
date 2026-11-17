import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_mlp_model(
    model,
    X_test_scaled,
    y_test,
    figures_dir: str = "figures"
) -> dict:
    """
    Task 5: Đánh giá mô hình MLP trên tập Test và trực quan hóa kết quả dự đoán.
    """
    os.makedirs(figures_dir, exist_ok=True)

    print("\n[Task 5] Đang đánh giá mô hình MLP trên tập Test...")
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f" -> Test Loss (MSE): {mse:.4f}")
    print(f" -> Test RMSE:       {rmse:.4f}")
    print(f" -> Test MAE:        {mae:.4f}")
    print(f" -> Test R-squared:  {r2:.4f}")

    # In thử 5 mẫu so sánh thực tế vs dự đoán
    comparison_sample = pd.DataFrame({
        "Actual Price ($1000s)": y_test[:5],
        "Predicted Price ($1000s)": np.round(y_pred[:5], 2),
        "Absolute Error": np.round(np.abs(y_test[:5] - y_pred[:5]), 2)
    })
    print("\n[Task 5] Ví dụ 5 mẫu dự đoán đầu tiên trên tập Test:")
    print(comparison_sample.to_string(index=False))

    # Vẽ biểu đồ Scatter: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.75, color="dodgerblue", edgecolors="k", label="Dự đoán (MLP)")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Đường lý tưởng (Actual = Predicted)")

    plt.xlabel("Actual Prices ($1000s)")
    plt.ylabel("Predicted Prices ($1000s)")
    plt.title(f"Actual vs. Predicted House Prices (MLP | R2 = {r2:.3f})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    scatter_path = os.path.join(figures_dir, "mlp_actual_vs_pred.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"[Task 5] Đã lưu biểu đồ Scatter tại: {scatter_path}")

    return {
        "Model": "MLP (Neural Network)",
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "R-squared (R2)": round(float(r2), 4)
    }