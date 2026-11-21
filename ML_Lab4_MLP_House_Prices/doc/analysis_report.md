# Báo cáo phân tích và so sánh mô hình
## Practice Exercise 4 – Multilayer Perceptron Regression: Predicting House Prices

## 1. Mục tiêu
Trong bài thực hành này, em sử dụng mô hình Multilayer Perceptron (MLP) để dự đoán giá nhà dựa trên các thông tin của căn nhà và khu vực xung quanh. Đây là bài toán hồi quy vì kết quả cần dự đoán là một giá trị liên tục, cụ thể là giá nhà.

Ngoài mô hình MLP theo yêu cầu của đề bài, em còn thực hiện thêm ba mô hình hồi quy khác gồm Linear Regression, Decision Tree Regression và Random Forest Regression. Sau khi huấn luyện, em dùng cùng tập dữ liệu kiểm tra để đánh giá và so sánh kết quả của các mô hình.

## 2. Dữ liệu và tiền xử lý
- **Bộ dữ liệu:** Boston Housing.
- **Số lượng mẫu:** 506.
- **Số đặc trưng đầu vào:** 13.
- **Biến mục tiêu:** `MEDV` – giá trị trung bình của căn nhà, tính theo đơn vị nghìn USD ($1000s).
- **Chia dữ liệu:** 80% dùng để huấn luyện (404 mẫu) và 20% dùng để kiểm tra (102 mẫu), với `random_state=42`.
- **Chuẩn hóa:** sử dụng `StandardScaler`. Bộ chuẩn hóa được fit trên tập Train, sau đó dùng để transform cả tập Train và Test.

## 3. Xây dựng và huấn luyện mô hình MLP
Mô hình được xây dựng bằng Keras theo kiến trúc Sequential, gồm:
- Lớp đầu vào có 13 đặc trưng.
- Lớp ẩn thứ nhất có 32 neuron, sử dụng hàm kích hoạt ReLU.
- Lớp ẩn thứ hai có 16 neuron, sử dụng hàm kích hoạt ReLU.
- Lớp đầu ra có 1 neuron, sử dụng hàm kích hoạt tuyến tính để dự đoán giá nhà.

Mô hình sử dụng optimizer Adam và hàm mất mát Mean Squared Error (MSE). Quá trình huấn luyện được thiết lập 100 epochs, batch size 32 và theo dõi kết quả trên tập validation.

## 4. Kết quả đánh giá và so sánh
Các chỉ số dưới đây được lấy từ file `metrics_comparison.csv` trong thư mục `doc/`.

| Mô hình | MSE | RMSE | MAE | R-squared (R²) |
|---|---:|---:|---:|---:|
| Random Forest Regression | 7.8900 | 2.8089 | 2.0381 | 0.8924 |
| MLP (Neural Network) | 13.3113 | 3.6485 | 2.4872 | 0.8185 |
| Decision Tree Regression | 17.8156 | 4.2209 | 2.7245 | 0.7571 |
| Linear Regression | 24.2911 | 4.9286 | 3.1891 | 0.6688 |

**Lưu ý:** Giá trị `MEDV` được tính theo nghìn USD, vì vậy RMSE và MAE trong bảng cũng có đơn vị nghìn USD; MSE có đơn vị bình phương của nghìn USD.

## 5. Nhận xét kết quả
1. **Random Forest Regression:** Đây là mô hình có kết quả tốt nhất trong lần chạy được ghi nhận. Mô hình có MSE thấp nhất (7.8900), RMSE thấp nhất (2.8089), MAE thấp nhất (2.0381) và R² cao nhất (0.8924). Điều này cho thấy Random Forest dự đoán tốt nhất trong bốn mô hình trên tập Test.
2. **MLP (Neural Network):** MLP đạt MSE = 13.3113 và R² = 0.8185. Kết quả này tốt hơn Decision Tree và Linear Regression, cho thấy mạng MLP học được các mối quan hệ giữa những đặc trưng đầu vào và giá nhà. Tuy nhiên, MLP vẫn chưa đạt hiệu quả bằng Random Forest trong lần đánh giá này.
3. **Decision Tree Regression:** Mô hình đạt MSE = 17.8156 và R² = 0.7571. Kết quả thấp hơn MLP và Random Forest nhưng tốt hơn Linear Regression.
4. **Linear Regression:** Mô hình có MSE cao nhất (24.2911) và R² thấp nhất (0.6688) trong bốn mô hình. Với bộ dữ liệu này, mô hình tuyến tính thể hiện khả năng giải thích biến động giá nhà thấp hơn ba mô hình còn lại.

Nhìn chung, cả bốn mô hình đều có R² dương. Trong kết quả hiện tại, thứ tự hiệu quả theo MSE từ tốt đến thấp là: Random Forest, MLP, Decision Tree và Linear Regression.

## 6. Kết luận
Qua bài thực hành, em đã thực hiện các bước tải và tiền xử lý dữ liệu Boston Housing, chia dữ liệu thành tập Train và Test, chuẩn hóa đặc trưng, xây dựng và huấn luyện mô hình MLP, đánh giá kết quả dự đoán và so sánh với các mô hình hồi quy khác.

Kết quả cho thấy MLP đạt hiệu quả tốt hơn Decision Tree và Linear Regression, nhưng Random Forest là mô hình có kết quả tốt nhất trên tập Test trong lần chạy này. Vì vậy, với bộ dữ liệu và cách thiết lập hiện tại, Random Forest là mô hình phù hợp nhất trong bốn mô hình được so sánh.

## 7. Hình ảnh kết quả
Các biểu đồ được lưu trong thư mục `figures/`:
- `mlp_training_loss.png`: biểu đồ Loss của quá trình huấn luyện và validation.
- `mlp_actual_vs_pred.png`: biểu đồ so sánh giá nhà thực tế và giá nhà dự đoán bằng MLP.
- `model_comparison.png`: biểu đồ so sánh MSE và R² giữa các mô hình.
