# Nguyên lý thư viện và phương pháp

## PyTorch và torchvision

PyTorch thực hiện forward, tính loss, autograd và cập nhật trọng số. `torchvision.models` cung cấp MobileNetV3, EfficientNet-B0 và ResNet18 pretrained ImageNet. Dự án thay classifier cuối để phù hợp số lớp của từng task.

- `mobilenet_v3_large`: mặc định, ưu tiên chất lượng.
- `mobilenet_v3_small`: nhẹ hơn, cần benchmark về độ chính xác và độ trễ.
- `head_only`: chỉ train classifier, phù hợp baseline nhanh.
- `fine_tune`: cập nhật toàn bộ model, thích nghi tốt hơn nhưng tốn tài nguyên hơn.

Tài liệu:

- https://docs.pytorch.org/docs/stable/
- https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## Dataset và preprocessing

`ImageFolder` dùng tên thư mục làm nhãn và sắp xếp class alphabetic. Dataset hiện có 5 category, nhưng freshness expert có thể có 2 hoặc 3 lớp tùy checkpoint.

Preprocessing đánh giá/inference chuẩn:

```text
RGB
Resize(232)
CenterCrop(224)
ToTensor
Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

Train thêm random crop, horizontal flip và rotation. Augmentation không thay thế group split; ảnh gốc và biến thể gần giống vẫn phải nằm cùng một split.

## Loss và tối ưu

Model trả logits, không softmax trước `CrossEntropyLoss`. Label smoothing mặc định `0.05` giảm sự tự tin quá mức.

AdamW mặc định dùng learning rate `5e-4`, weight decay `1e-4`. SGD có momentum/Nesterov là lựa chọn thay thế. Scheduler hỗ trợ `ReduceLROnPlateau` và cosine.

AMP dùng `torch.autocast` và `GradScaler` khi có CUDA.

Tài liệu:

- https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
- https://docs.pytorch.org/docs/stable/notes/amp_examples.html

## Metric và diễn giải

- Accuracy: tỷ lệ đúng tổng thể.
- Precision: độ chính xác của các dự đoán một class.
- Recall: khả năng tìm đúng các mẫu thật của class.
- F1: trung bình điều hòa precision và recall.
- Macro-F1: trung bình F1 không theo trọng số số lượng mẫu.
- Confusion matrix: ma trận nhãn thật/dự đoán.
- Confidence: xác suất lớn nhất sau softmax; không đồng nghĩa xác suất đúng nếu chưa calibration.
- Entropy: mức phân tán của phân phối xác suất; entropy thấp không đảm bảo model đúng.

Với bài toán độ tươi, recall `SPOILED` và metric theo từng source/product quan trọng hơn accuracy tổng thể.

## Transfer learning và chống học vẹt

Pretrained ImageNet chỉ cung cấp đặc trưng khởi đầu, không ngăn model học thuộc dataset. Các biện pháp chính là:

1. Group split theo source/product/session/object.
2. Loại duplicate bằng SHA-256 và near-duplicate bằng perceptual hash/embedding.
3. Test độc lập từ nguồn hoặc đợt chụp chưa có trong train.
4. Báo cáo macro-F1, confusion matrix và recall `SPOILED`.
5. Không chọn checkpoint dựa trên test nhiều lần.

Dataset hiện tại đã có cross-split duplicate, nên metric cũ phải xem là chẩn đoán.

## ONNX và triển khai

ONNX export output logits, client áp dụng softmax và mapping class. Mapping phải lấy từ manifest/checkpoint contract, không nên duy trì danh sách nhãn riêng trong nhiều script. Khi ONNX dùng external data, file `.onnx.data` là một phần bắt buộc của package.
