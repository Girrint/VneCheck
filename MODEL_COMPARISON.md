# So sánh model

## Mục tiêu

So sánh kiến trúc và phương pháp transfer learning trên cùng dataset sạch,
cùng seed, transform, batch size và số epoch. Không so kết quả run cũ với run
mới nếu chúng dùng split khác nhau.

Các kiến trúc được hỗ trợ:

- `mobilenet_v3_small`
- `mobilenet_v3_large`
- `efficientnet_b0`
- `resnet18`

Phương pháp:

- `fine_tune`: cập nhật toàn model.
- `head_only`: chỉ cập nhật classifier.

## Quy trình hợp lệ

1. Chốt `data_split` và `data_negative`, không thay đổi giữa các trial.
2. Train các trial với cùng seed và budget.
3. Xếp hạng bằng validation loss và macro-F1.
4. Kiểm tra recall từng lớp, đặc biệt `SPOILED`.
5. Chọn cấu hình cuối.
6. Chỉ sau khi đã chốt cấu hình mới dùng test để báo cáo cuối.
7. Đánh giá cả router riêng, expert với oracle category và end-to-end.
8. Export mỗi Unified run bằng `--name` và mỗi Legacy pipeline bằng
   `--bundle-name`; tích các artifact cần so sánh trên web.

Không dùng test accuracy để chọn model, threshold, learning rate hoặc epoch.
Nếu đã xem test nhiều lần để ra quyết định, test đó đã trở thành validation và
cần một test mới.

## Công cụ

Benchmark MobileNetV3:

```powershell
python benchmark_mobilenet.py router --epochs 15 --batch-size 16 --workers 4
python benchmark_mobilenet.py freshness --category gia_cam --epochs 15
```

Grid search rộng hơn:

```powershell
python tune.py --task router --epochs 3
python tune.py --task freshness --category gia_cam --epochs 3
```

Hai công cụ gọi `train.py`, vì vậy mỗi trial vẫn sinh test metric. Khi chọn
cấu hình, chỉ đọc `best_valid_loss` và `best_valid_accuracy`; giữ test metric
ngoài tiêu chí xếp hạng.

## Bảng kết quả đề xuất

| Task | Model | Method | Seed | Best val loss | Val accuracy | Test macro-F1 | SPOILED recall | Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Router | ... | ... | 42 | ... | ... | ... | n/a | ... |
| Expert | ... | ... | 42 | ... | ... | ... | ... | ... |

Snapshot Legacy `models/legacy/original` được train trước split sạch, vì vậy
không dùng metric của snapshot này làm kết quả chính thức. Hãy train lại rồi
lấy số liệu từ `runs/<run>/summary.json`.

Router Legacy hiện có 7 lớp; ngoài macro-F1 cần xem recall riêng
`UNSUPPORTED` và `NON_FOOD`. Router 5 lớp trong snapshot `original` chỉ dùng
làm baseline lịch sử trên ảnh `SUPPORTED`.

Với Unified, dùng validation food macro-F1 để chọn checkpoint mặc định và xem
thêm recall từng food status. Web batch sinh seed mới ở mỗi lượt để duyệt các
mẫu khác nhau, nhưng không dùng việc chạy test lặp lại để tuning. Seed của lượt
đánh giá được lưu trong JSON report.

## Tiêu chí triển khai

Ngoài metric, cần kiểm tra:

- Kích thước checkpoint và ONNX.
- Latency trên thiết bị đích.
- Sai khác xác suất PyTorch/ONNX bằng `verify_onnx.py`.
- Class order và checksum trong manifest.
- Độ ổn định trên nguồn ảnh mới, không chỉ ảnh cùng dataset.
