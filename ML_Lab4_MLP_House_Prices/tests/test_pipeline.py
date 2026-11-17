import os
import numpy as np
from src.data_loader import load_and_preprocess_data
from src.mlp_model import build_mlp_model

def test_data_shapes_and_scaling():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    assert X_train.shape[1] == 13, "Boston Housing phải có đúng 13 đặc trưng"
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    # Kiểm tra StandardScaler đưa trung bình tập Train về xấp xỉ 0
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6)

def test_mlp_architecture_output():
    model = build_mlp_model(input_dim=13)
    dummy_input = np.random.randn(4, 13)
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (4, 1), "Đầu ra của MLP Regression phải có dạng (batch_size, 1)"