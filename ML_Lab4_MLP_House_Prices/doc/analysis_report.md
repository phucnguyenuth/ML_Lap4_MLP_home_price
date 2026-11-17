# Báo cáo Phân tích & So sánh Mô hình (Practice Exercise 4)

## 1. Bảng tổng hợp chỉ số đánh giá trên tập Test
| Model                    |     MSE |   RMSE |    MAE |   R-squared (R2) |
|:-------------------------|--------:|-------:|-------:|-----------------:|
| Random Forest Regression |  7.89   | 2.8089 | 2.0381 |           0.8924 |
| MLP (Neural Network)     | 13.3113 | 3.6485 | 2.4872 |           0.8185 |
| Decision Tree Regression | 17.8156 | 4.2209 | 2.7245 |           0.7571 |
| Linear Regression        | 24.2911 | 4.9286 | 3.1891 |           0.6688 |

## 2. Phân tích chuyên sâu kết quả
1. **Mô hình đạt hiệu suất cao nhất (Random Forest Regression):** Đạt mức sai số thấp nhất (`MSE = 7.89`) và hệ số xác định cao nhất (`R2 = 0.8924`). Trên tập dữ liệu dạng bảng (tabular data) kích thước vừa và nhỏ (~506 mẫu), các thuật toán học tập hợp dựa trên cây quyết định (Random Forest) khai thác ranh giới đặc trưng rất hiệu quả và ít bị nhiễu.
2. **Hiệu suất của Multilayer Perceptron (MLP):** Nhờ cấu trúc 2 lớp ẩn (`Dense(32)` và `Dense(16)`) kết hợp hàm kích hoạt phi tuyến `ReLU`, mạng MLP học được các mối quan hệ phi tuyến giữa đặc trưng nhà ở (như số phòng `RM`, tỷ lệ thu nhập thấp `LSTAT`) và giá nhà, vượt trội rõ rệt so với hồi quy tuyến tính truyền thống. Tuy nhiên, do lượng mẫu huấn luyện chỉ có 404 dòng (80% của 506), mạng nơ-ron chưa khai thác hết lợi thế như trên dữ liệu lớn.
3. **Linear Regression:** Có chỉ số MSE cao hơn và R2 thấp hơn vì giả định mối quan hệ giữa các đặc trưng và giá nhà hoàn toàn là tuyến tính (đường thẳng), không nắm bắt được các tương tác phi tuyến phức tạp trong thực tế.
4. **Decision Tree Regression:** Cây quyết định đơn lẻ dễ xảy ra hiện tượng học vẹt (Overfitting) trên tập Train nên độ khái quát hóa trên tập Test kém ổn định hơn Random Forest và MLP.
