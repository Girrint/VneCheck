# Cấu trúc dự án

## Entry points

| File | Vai trò |
|---|---|
| `main.py` | Menu tương tác cho train, predict, dataset status và kết quả |
| `train.py` | Train router hoặc một freshness expert |
| `predict.py` | Inference PyTorch router → expert |
| `evaluate_router_uncertainty.py` | Accuracy, confidence, entropy và coverage của router |
| `evaluate_end_to_end.py` | Đánh giá router, oracle expert và pipeline thật |
| `tune.py` | Grid search nhiều kiến trúc/cấu hình |
| `benchmark_mobilenet.py` | So sánh các biến thể MobileNetV3 cố định |
| `export_onnx.py` | Export checkpoint PyTorch sang ONNX |
| `verify_onnx.py` | So output PyTorch và ONNX trên một ảnh |
| `package_models.py` | Tạo manifest checksum và tùy chọn copy artifact |
| `test_regression.py` | Regression tests cho contract, split và preprocessing |

## Package `food_ai`

| File | Vai trò |
|---|---|
| `config.py` | Category, freshness label, score anchor và model path |
| `data.py` | ImageFolder, transform và DataLoader train/valid/test |
| `model.py` | Model factory, freeze classifier và load checkpoint |
| `experiment.py` | Theo dõi run, CSV metric và classification report |

## Script dữ liệu

Các script dữ liệu hiện tại:

```text
scripts/resplit_dataset.py
scripts/dataset_stats.py
```

`resplit_dataset.py` đảm nhiệm SHA-256 dedupe, quarantine xung đột nhãn, pHash
clustering, chia 70/15/15 và sinh audit report. `dataset_stats.py` chỉ đếm file
theo split/category/freshness/product, không mở nội dung ảnh. Các script chia
valid, xóa duplicate và near-duplicate cũ đã được bỏ.

## Thư mục runtime

| Thư mục | Nội dung |
|---|---|
| `data/` | Dataset nguồn cũ, không dùng trực tiếp để train |
| `data_split/` | Dataset train/valid/test hiện tại |
| `reports/` | Cache và audit của quá trình resplit |
| `runs/` | Config, metric, checkpoint và summary từng lần train |
| `models/` | Checkpoint/ONNX đang được dùng khi inference |

`data/`, `data_split/`, `runs/`, `models/` và report sinh tự động không nên
commit vào Git.

## Tài liệu

| File | Nội dung |
|---|---|
| `README.md` | Bắt đầu nhanh và trạng thái hiện tại |
| `DATA.md` | Dataset và audit split |
| `TRAIN.md` | Train, resume, đánh giá |
| `ARCHITECTURE.md` | Kiến trúc router–expert |
| `ALGORITHM.md` | Thuật toán dữ liệu và model |
| `MODEL_COMPARISON.md` | Quy trình so sánh model |
| `LIBRARY_PRINCIPLES.md` | Vai trò các thư viện và nguyên lý ML |
| `IMPLEMENTATION_REPORT.md` | Báo cáo thay đổi hiện tại |
