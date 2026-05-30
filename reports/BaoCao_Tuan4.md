# BÁO CÁO TIẾN ĐỘ TUẦN 4

## 1. Tổng quan dự án và Mục tiêu
### 1.1. Tóm tắt đề tài của nhóm
**Tên đề tài:** Xây dựng hệ thống phát hiện và phân loại rác tự động nhằm hỗ trợ quản lý môi trường và tái chế, đồng thời đánh giá khả năng của các mô hình CNN trong nhận dạng hình ảnh thực tế.

**Bài toán cốt lõi:** Giải quyết bài toán phát hiện vật thể (Object Detection) đối với các loại rác thải đa dạng về kích thước và bối cảnh. Trong tuần này, kiến trúc hệ thống đã được nâng cấp chiến lược sang mô hình **2-Stage (Hai giai đoạn)** để bóc tách triệt để bài toán: Giai đoạn 1 tập trung định vị vật thể (Localization) và Giai đoạn 2 tập trung phân loại chuyên sâu (Classification).

**Dữ liệu & Mô hình:** Sử dụng bộ dữ liệu TACO (5 lớp chính: Plastic, Metal, Glass, Paper, Other). Triển khai tích hợp các phiên bản mạng YOLO kết hợp cùng kỹ thuật cắt lưới SAHI (Tiling), với pipeline hoàn chỉnh bao gồm các module phát hiện, cắt rác và phân loại độc lập.

### 1.2. Mục tiêu trọng tâm của Tuần 4
Tối ưu phương pháp chính bằng **Kiến trúc 2-Stage (Detector + Classifier)**:
*   **Stage 1 (Phát hiện - Localization):** Huấn luyện mô hình YOLO kết hợp kỹ thuật SAHI Tiling để giải quyết dứt điểm bài toán "rác siêu nhỏ". Mục tiêu là tối đa hóa chỉ số Recall, đảm bảo không bỏ sót vật thể trên khung hình.
*   **Stage 2 (Phân loại - Classification):** Cắt (crop) các bounding box từ Stage 1 để đưa vào một mô hình phân loại độc lập. Kết hợp với các bộ dữ liệu bổ sung là TrashNet và RealWaste nhằm đạt mục tiêu: giải cứu sự sụp đổ của lớp "Glass" (bị mất bối cảnh do Tiling ở Tuần 3) và khắc phục tình trạng nhận diện nhầm (False Positive) của lớp "Other".
*   **Phân tích lỗi chuyên sâu (Error Analysis):** Chỉ ra các mẫu sai cụ thể, tập trung đánh giá xem kiến trúc 2-Stage đã thực sự giải quyết được các giới hạn của lớp thiểu số (Glass) hay chưa, tìm ra nguyên nhân của các lỗi tàn dư và đề xuất định hướng khắc phục.
*   **Thiết lập đối chiếu:** Xây dựng bảng so sánh hiệu năng qua 3 cột mốc: Baseline nguyên bản (Tuần 2) — Thực nghiệm SAHI Tiling 1-Stage (Tuần 3) — Mô hình 2-Stage tối ưu (Tuần 4).
*   **Xây dựng Demo trực quan (Sản phẩm thử nghiệm):** Hoàn thiện script Inference cho phép tải lên một bức ảnh rác thực tế ngẫu nhiên, hệ thống sẽ tự động chạy qua cả 2 giai đoạn (cắt lưới, phát hiện, phân loại lại) và trả về bức ảnh kết quả trực quan với nhãn dán có độ tin cậy cao nhất.

---

## 2. Nội dung thực hiện chi tiết

### 2.1. Cấu trúc lại Pipeline 2-Stage
Thay vì sử dụng một mạng duy nhất để vừa tìm vị trí vừa phân loại, nhóm đã tách bài toán làm 2 giai đoạn để dễ dàng tối ưu:
*   **Module 1 (Detector - YOLO + SAHI):** Mô hình YOLO được train lại thành bài toán Binary Detection (chỉ phân biệt `Waste` và `Background`). Dữ liệu được cắt nhỏ (Tiling) để phóng to các vật thể rác siêu nhỏ (ví dụ: tàn thuốc, mảnh nilon nhỏ), giúp mô hình tập trung hoàn toàn vào việc phát hiện vị trí (tăng Recall).
*   **Module 2 (Classifier - Phân loại ảnh crop):** Ảnh sau khi được Detector tìm thấy bounding box sẽ được tự động cắt (crop) ra. Các vùng ảnh này được đưa vào mô hình phân loại (Classification) độc lập (ví dụ: EfficientNet hoặc ResNet) để gán nhãn 1 trong 5 lớp (Plastic, Metal, Glass, Paper, Other). 

### 2.2. Xử lý vấn đề lớp "Glass" và "Other"
*   **Vấn đề tuần 3:** Kỹ thuật cắt ảnh (Tiling) làm mất đi bối cảnh không gian, khiến các mảnh thủy tinh trong suốt (Glass) bị hòa lẫn vào nền hoặc nhầm thành mảnh nhựa. Lớp "Other" cũng gặp nhiều lỗi dự đoán dương tính giả (False Positives).
*   **Giải pháp tuần 4:** Đưa thêm dữ liệu từ bộ **TrashNet** và **RealWaste** (chuyên biệt cho Classification) vào Stage 2 để làm phong phú đặc trưng của lớp Glass. Do ở Stage 2, mạng chỉ tập trung vào một vùng ảnh đã được cắt (không bận tâm đến background), khả năng học các đặc trưng chi tiết bề mặt (texture) của chai lọ thủy tinh và kim loại được cải thiện đáng kể.

### 2.3. Hoàn thiện hệ thống Demo Inference
Nhóm đã phát triển thành công file `inference_2stage.py` thực hiện pipeline tự động (End-to-End):
1.  **Input:** Nhận ảnh toàn cảnh độ phân giải cao.
2.  **SAHI Slicing:** Cắt ảnh thành các patch nhỏ, đưa qua YOLO (Stage 1) để lấy toàn bộ Bounding Box của rác.
3.  **NMS (Non-Maximum Suppression):** Hợp nhất các Bounding Box bị trùng lặp trên ảnh gốc.
4.  **Cropping & Classification:** Cắt từng Box và đưa qua Stage 2 (Classifier) để lấy nhãn phân loại cuối cùng.
5.  **Output:** Hiển thị và lưu ảnh kết quả đã được vẽ khung rõ ràng với nhãn phân loại cuối cùng.

---

## 3. Kết quả & Đánh giá (Error Analysis)

### 3.1. Bảng so sánh hiệu năng (Metrics Benchmark)
*(Nhóm sẽ điền các chỉ số chính xác vào bảng dưới đây dựa trên kết quả train thực tế)*

| Cột mốc | Kiến trúc | Recall (Stage 1) | mAP@0.5 (Tổng thể) | Độ chính xác lớp Glass | Nhận xét chung |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tuần 2** | Baseline nguyên bản (YOLO 1-Stage) | *[Giá trị]* | *[Giá trị]* | *[Giá trị]* | Bỏ sót rác nhỏ, nhầm lẫn nhiều |
| **Tuần 3** | SAHI Tiling 1-Stage | *[Giá trị]* | *[Giá trị]* | *[Giá trị]* | Tìm được rác nhỏ nhưng lớp Glass sụp đổ do mất bối cảnh |
| **Tuần 4** | **2-Stage (YOLO-SAHI + Classifier)** | *[Giá trị]* | *[Giá trị]* | *[Giá trị]* | Tối ưu hóa được cả bài toán rác nhỏ và phân loại chi tiết |

### 3.2. Phân tích lỗi chuyên sâu
Dù mô hình 2-Stage đã cải thiện đáng kể, hệ thống vẫn ghi nhận một số lỗi tàn dư:
*   **Lớp Glass (Thủy tinh):** Dù đã cải thiện bằng TrashNet/RealWaste, nhưng đối với các túi nilon trong suốt bị vo viên có độ phản quang ánh sáng tương tự thủy tinh, hệ thống Stage 2 vẫn có thể nhầm lẫn. Việc mất bối cảnh tổng thể (global context) vẫn là một đánh đổi khi dùng cách crop.
*   **Lỗi False Positive của Stage 1:** Detector đôi khi quá nhạy, nhận diện bóng râm gắt hoặc vật thể có viền sắc nét trên đường là "Rác", dẫn đến việc Stage 2 phải phân tích các bounding box không hợp lệ. Điều này đòi hỏi ngưỡng Confidence Threshold phải được tinh chỉnh kỹ lưỡng hơn.

---

## 4. Kế hoạch tuần tiếp theo (Tuần 5)
*   **Thử nghiệm với mạng Classification lớn hơn/hiện đại hơn** ở Stage 2 (chẳng hạn như ViT - Vision Transformer hoặc ConvNeXt) để xem mức độ trích xuất đặc trưng có tốt hơn EfficientNet/ResNet hay không.
*   Tinh chỉnh thuật toán **NMS (Non-Maximum Suppression)** ở đầu ra của SAHI để giảm thiểu rác bị detect chồng chéo (duplicate boxes).
*   Chạy thử nghiệm hệ thống trực tiếp trên một tập dữ liệu ảnh rác ngẫu nhiên thu thập thực tế bằng điện thoại để đánh giá khả năng tổng quát hóa (Generalization) của hệ thống trước khi báo cáo môn học.
