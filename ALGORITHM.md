# Thuật toán

## 1. Chuẩn bị dữ liệu

`scripts/resplit_dataset.py` gộp các source, tính SHA-256 và pHash rồi thực hiện:

1. Nếu một SHA-256 xuất hiện ở nhiều model label, loại toàn bộ nhóm đó.
2. Nếu một SHA-256 lặp trong cùng model label, giữ một file đại diện.
3. Chỉ so pHash giữa ảnh cùng leaf class `category/freshness/product`.
4. Dùng BK-tree tìm các pHash có Hamming distance không vượt threshold.
5. Dùng union-find tạo cluster liên thông.
6. Gán nguyên cluster vào train, valid hoặc test bằng hàm chi phí sai lệch tỷ lệ.
7. Ép cluster quá lớn đối với target valid/test vào train.

Split hiện tại dùng threshold `0`, seed `42` và tỷ lệ `0.70/0.15/0.15`.

## 2. Tiền xử lý

Train augmentation:

```text
EXIF orientation → RGB
RandomResizedCrop(224, scale=0.85..1.0, ratio=0.85..1.15)
RandomHorizontalFlip
ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0)
ToTensor
ImageNet normalization
```

Không dùng rotation vì vùng đệm ở góc có thể trở thành tín hiệu giả. Không đổi
hue vì màu là đặc trưng quan trọng của độ tươi. Jitter chỉ mô phỏng sai khác ánh
sáng nhỏ giữa các thiết bị chụp.

Validation, test và inference:

```text
EXIF orientation → RGB
Resize(232)
CenterCrop(224)
ToTensor
ImageNet normalization
```

## 3. Huấn luyện

Mỗi batch train chạy:

```text
image → model → logits
      → CrossEntropyLoss(label_smoothing=0.05)
      → backward → optimizer.step
```

Validation không cập nhật trọng số. Checkpoint được chọn theo validation loss;
scheduler và early stopping cũng chỉ quan sát validation. Sau khi train xong,
code nạp lại checkpoint tốt nhất và chấm `data_split/test` đúng một lần.

Hai transfer-learning mode:

- `fine_tune`: cập nhật toàn bộ model.
- `head_only`: đóng băng backbone, chỉ cập nhật classifier.

## 4. Inference hai tầng

1. Preprocess ảnh thành tensor `[1, 3, 224, 224]`.
2. Router tạo logits 5 category; softmax và argmax chọn category.
3. Nạp expert tương ứng.
4. Expert tạo logits freshness theo class list trong checkpoint.
5. Softmax, argmax và tính freshness score.
6. Trả JSON gồm nhãn, confidence, score và cảnh báo.

## 5. Huấn luyện và inference unified

Unified dùng một backbone và ba head. Loss tổng là tổng có trọng số của:

- Food cross-entropy trên cả `SUPPORTED`, `UNSUPPORTED`, `NON_FOOD`.
- Category cross-entropy chỉ trên `SUPPORTED`.
- Freshness cross-entropy chỉ trên `SUPPORTED`.

Food loss mặc định dùng class weight `sqrt_inverse` để giảm thiên lệch giữa
66.115 positive train và 3.500 ảnh ở mỗi lớp negative. Checkpoint tốt nhất mặc
định chọn bằng validation food macro-F1. Inference tạo ba logits trong một lần
forward; category/freshness chỉ có ý nghĩa khi food status là `SUPPORTED`.

## 6. Metric

`train.py` lưu accuracy, loss, macro-F1, precision, recall, F1 từng lớp và
confusion matrix. Macro-F1 quan trọng khi class không cân bằng; cần xem riêng
recall `SPOILED`, không chỉ xem accuracy tổng.

`evaluate_end_to_end.py` báo ba tầng kết quả:

- Router category.
- Freshness khi dùng đúng category thật (oracle routing).
- Freshness khi category do router dự đoán (pipeline thực).

Các expert 2 lớp và 3 lớp được quy về tên lớp chung khi cộng confusion matrix.
