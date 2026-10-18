# Thư viện và nguyên lý

## PyTorch và torchvision

PyTorch thực hiện tensor, autograd, optimizer, checkpoint và inference.
`torchvision` cung cấp ImageFolder, transform và backbone pretrained.

Backbone ImageNet giúp khởi tạo feature thị giác tốt hơn train từ đầu. Tuy
nhiên domain thực phẩm khác ImageNet, nên mặc định dùng fine-tune toàn model.
`head_only` phù hợp làm baseline nhanh hoặc khi tài nguyên hạn chế.

## Dataset và transform

`NonEmptyImageFolder` bỏ các class directory không có ảnh nhưng vẫn giữ quy tắc
class theo tên thư mục. Train dùng augmentation; validation, test và inference
dùng transform xác định, không ngẫu nhiên.

Ba split được load độc lập. Đây là điều kiện bắt buộc để test không bị lấy từ
train trong mỗi lần chạy.

## Loss và tối ưu

Cross-entropy dùng trực tiếp logits và integer class index. Label smoothing
`0.05` giảm overconfidence nhẹ. AdamW là optimizer mặc định; SGD vẫn được hỗ
trợ. Plateau scheduler giảm learning rate khi validation loss không cải thiện;
CosineAnnealingLR là lựa chọn khác.

## Metric

- Accuracy: tỷ lệ dự đoán đúng tổng thể.
- Precision: trong các mẫu model dự đoán một lớp, bao nhiêu mẫu đúng.
- Recall: trong các mẫu thật của một lớp, bao nhiêu mẫu được tìm thấy.
- F1: trung bình điều hòa precision và recall.
- Macro-F1: trung bình F1 giữa các lớp, không để lớp lớn lấn át hoàn toàn.

Với bài toán độ tươi, recall `SPOILED` và confusion matrix quan trọng hơn chỉ
một accuracy tổng.

## Hash dữ liệu

SHA-256 phát hiện file giống byte. pHash phát hiện nội dung thị giác gần giống
ở mức thô. BK-tree giảm số phép so khoảng cách Hamming; union-find gom các cặp
liên thông thành cluster. Cả hai không xác định chắc chắn hai ảnh thuộc cùng
đối tượng hoặc cùng phiên chụp; metadata thật vẫn đáng tin cậy hơn.

## ONNX

`export_onnx.py` chuyển checkpoint sang graph ONNX. `onnx` kiểm tra graph và
`onnxruntime` chạy inference CPU. `verify_onnx.py` yêu cầu top-1 giống PyTorch
và sai khác xác suất tối đa dưới `1e-5` trên ảnh kiểm tra.

## Nguyên tắc an toàn thực nghiệm

- Không augmentation validation/test.
- Không chọn model bằng test.
- Không hard-code class order; đọc từ checkpoint/manifest.
- Không coi softmax confidence là xác suất đã calibration.
- Không coi freshness score là chứng nhận an toàn thực phẩm.
