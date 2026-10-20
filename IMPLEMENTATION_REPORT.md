# Báo cáo hiện trạng triển khai

Ngày cập nhật: 2026-10-20.

## Dataset

- Gộp `data/train` và `data/valid`: 116.818 ảnh nguồn.
- Loại 673 file thuộc nhóm SHA-256 xung đột nhãn.
- Loại 21.696 exact duplicate dư thừa cùng nhãn.
- Dataset supported sạch còn 94.449 ảnh: 66.115 train, 14.172 valid và
  14.162 test.
- pHash cluster không bị tách giữa các split; 4 cluster lớn được ép vào train.
- Negative có 10.000 ảnh: 5.000 `UNSUPPORTED` từ Food-101 và 5.000
  `NON_FOOD`, chia thành 7.000/1.500/1.500.
- Unified manifest có 104.449 dòng và đủ ba food status trong cả ba split.
- Dữ liệu, cache, report, model và run đều nằm ngoài Git.

## Pipeline Legacy

- Router 5 lớp chọn một trong 5 freshness expert.
- `cu`, `qua`, `rau` dùng expert 2 lớp; `gia_cam`, `gia_suc` dùng expert 3 lớp.
- Train/valid/test được load độc lập; validation chọn checkpoint, test chỉ chấm
  sau khi nạp checkpoint tốt nhất.
- Hỗ trợ MobileNetV3-Small/Large, EfficientNet-B0 và ResNet18; hỗ trợ
  `fine_tune` và `head_only`.
- Snapshot baseline cũ nằm tại `models/legacy/original`.
- `export_onnx.py --bundle-name <id>` yêu cầu đủ router + 5 checkpoint, export
  sáu ONNX và `manifest.json` vào `models/legacy/<id>`.

## Pipeline Unified

- Một backbone dùng chung với ba head: food, category và freshness.
- Hỗ trợ cùng bốn backbone như Legacy.
- Category/freshness loss được mask trên negative; food loss mặc định dùng
  `sqrt_inverse` class weight.
- Best checkpoint mặc định chọn theo validation food macro-F1.
- Artifact mặc định nằm trong `models/unified`; artifact A/B riêng nằm trong
  `models/unified/exports/<id>.onnx` kèm sidecar manifest.
- Unified ONNX có input batch động và ba output logits; food gate hiện đã được
  train với `SUPPORTED`, `UNSUPPORTED`, `NON_FOOD`.

## Web đánh giá ONNX

- Lập chỉ mục `data_split/test` và `data_negative/test`.
- Chế độ `Legacy`, `Unified` hoặc `Cả hai`.
- Quét nhiều Unified artifact và nhiều Legacy bundle; hai danh sách model được
  chọn độc lập.
- Đánh giá một ảnh hoặc batch 25/100/500/1.000/5.000/toàn bộ.
- Mỗi batch dùng seed mới nên lần chạy sau chọn nhóm ảnh khác; trong
  cùng batch không lặp đường dẫn. Seed được lưu trong JSON report.
- Metric food áp dụng trên cả ba food status; category/freshness và Legacy chỉ
  áp dụng trên `SUPPORTED`.
- CSV lưu kết quả từng ảnh, model ID/tên/kiến trúc, confidence, probability và
  cờ đúng/sai; JSON lưu bộ lọc, seed và metric tổng hợp.

## Tiền xử lý và kiểm thử

- Train: EXIF → RGB → RandomResizedCrop(224) → flip → ColorJitter nhẹ →
  ImageNet normalization.
- Valid/test/inference: EXIF → RGB → Resize(232) → CenterCrop(224) → ImageNet
  normalization.
- Không dùng rotation/hue mạnh để tránh nền giả và làm sai tín hiệu màu.
- Regression test bao phủ contract lớp, split, dedupe, masked loss, class
  balance, output Unified, tên artifact và metric web.

## Việc tiếp theo

1. Train nhiều backbone Legacy và Unified trên cùng split/budget.
2. Export mỗi Unified checkpoint bằng `--name` riêng và mỗi Legacy pipeline
   bằng `--bundle-name` riêng.
3. Chọn model bằng validation; chỉ dùng test để báo cáo cuối.
4. So sánh accuracy/macro-F1/recall, dung lượng, CPU latency và peak memory.
5. Benchmark model được chọn trên thiết bị di động thật.
6. Mở rộng negative bằng ảnh thực tế và bổ sung metadata source/object/session
   để giảm leakage tốt hơn pHash.
