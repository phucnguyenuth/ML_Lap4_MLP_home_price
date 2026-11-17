import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_mlp_model(input_dim: int) -> Sequential:
    """
    Task 3: Xây dựng kiến trúc Multilayer Perceptron (MLP) cho bài toán hồi quy.
    """
    tf.random.set_seed(42)
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation="relu", name="hidden_layer_1"),
        Dense(16, activation="relu", name="hidden_layer_2"),
        Dense(1, activation="linear", name="output_layer")
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model

def train_mlp_model(
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    epochs: int = 100,
    batch_size: int = 32,
    models_dir: str = "models",
    figures_dir: str = "figures"
):
    """
    Task 4: Huấn luyện mô hình MLP, lưu trọng số và vẽ biểu đồ lịch sử huấn luyện.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("\n[Task 3] Đang khởi tạo kiến trúc mô hình MLP...")
    model = build_mlp_model(input_dim=X_train_scaled.shape[1])
    model.summary()

    print(f"\n[Task 4] Đang huấn luyện mô hình MLP ({epochs} epochs, batch_size={batch_size})...")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_scaled, y_test),
        verbose=1
    )

    # Lưu mô hình đã huấn luyện
    model_path = os.path.join(models_dir, "mlp_model.keras")
    model.save(model_path)
    print(f"[Task 4] Đã lưu mô hình MLP tại: {model_path}")

    # Vẽ và lưu biểu đồ Loss (MSE) và MAE qua từng Epoch
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["loss"], label="Train Loss (MSE)", color="royalblue")
    axes[0].plot(history.history["val_loss"], label="Validation Loss (MSE)", color="darkorange")
    axes[0].set_title("MLP Training vs Validation MSE")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Mean Squared Error")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(history.history["mae"], label="Train MAE", color="royalblue")
    axes[1].plot(history.history["val_mae"], label="Validation MAE", color="darkorange")
    axes[1].set_title("MLP Training vs Validation MAE")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Mean Absolute Error")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    loss_fig_path = os.path.join(figures_dir, "mlp_training_loss.png")
    plt.savefig(loss_fig_path, dpi=300)
    plt.close()
    print(f"[Task 4] Đã lưu biểu đồ quá trình học tại: {loss_fig_path}")

    return model, history