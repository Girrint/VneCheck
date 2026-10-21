# VnGrocery AI

Hệ thống có hai pipeline chạy song song:

1. **Legacy:** router 7 lớp nhận diện 5 nhóm hoặc từ chối
   `UNSUPPORTED`/`NON_FOOD`; chỉ ảnh thuộc 5 nhóm mới chạy freshness expert.
2. **Unified:** một backbone với ba head food/category/freshness, có thể từ
   chối `UNSUPPORTED` và `NON_FOOD`.

Cả hai dùng ba split độc lập; không lấy một phần train làm test. Legacy giữ
nguyên kiến trúc 6 model để làm baseline, còn Unified hướng tới triển khai di
động bằng một ONNX. Web local cho phép tích chọn nhiều Unified artifact, nhiều
Legacy bundle hoặc chạy cả hai nhóm trên cùng ảnh. Xem [UNIFIED.md](UNIFIED.md).

## Trạng thái hiện tại

- Dataset sạch: 94.449 ảnh.
- Train: 66.115 ảnh; validation: 14.172 ảnh; test: 14.162 ảnh.
- Exact duplicate cùng nhãn chỉ giữ một bản.
- File SHA-256 giống nhau nhưng xung đột nhãn bị loại toàn bộ.
- pHash cluster không bị tách qua các split.
- Các cluster quá lớn đối với validation/test được ép vào train.
- Expert `gia_cam` và `gia_suc` có 3 lớp; `cu`, `qua`, `rau` có 2 lớp.
- Negative: 10.000 ảnh, gồm 5.000 `UNSUPPORTED` và 5.000 `NON_FOOD`.
- Unified manifest: 104.449 dòng trên ba split.
- Bốn backbone dùng được cho cả hai pipeline: MobileNetV3-Small/Large,
  EfficientNet-B0 và ResNet18.

Snapshot `models/legacy/original` được tạo trước split sạch nên chỉ là baseline
kỹ thuật. Unified artifact hiện tại có food gate ba lớp; mọi model mới vẫn phải
được chọn bằng validation và chỉ dùng test để báo cáo cuối.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quy trình chuẩn

Kiểm tra PyTorch có dùng CUDA trước khi train:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Quy trình copy-paste để train cả bốn backbone Legacy/Unified, export từng ONNX
và verify đúng cặp artifact nằm tại [TRAIN.md](TRAIN.md).

Kiểm tra hoặc tạo lại split:

```powershell
python scripts\resplit_dataset.py --threshold 0 --dry-run
python scripts\resplit_dataset.py --threshold 0
```

Thống kê số lượng ảnh mà không mở nội dung ảnh:

```powershell
python scripts\dataset_stats.py
python scripts\dataset_stats.py --products
python scripts\dataset_stats.py --csv reports\dataset_stats.csv
```

Train:

```powershell
python train.py router
python train.py freshness --category cu
python train.py freshness --category gia_cam
python train.py freshness --category gia_suc
python train.py freshness --category qua
python train.py freshness --category rau
```

Dự đoán bằng checkpoint PyTorch:

```powershell
python predict.py path\to\image.jpg
```

Đánh giá trên test độc lập:

```powershell
python evaluate_router_uncertainty.py
python evaluate_end_to_end.py
```

Export và kiểm tra ONNX legacy thành bundle riêng:

```powershell
python export_onnx.py --bundle-name mobilenet_v3_large_v1
python verify_onnx.py `
  --image path\to\image.jpg `
  --checkpoint-root models `
  --onnx-root models\legacy\mobilenet_v3_large_v1
```

Hướng dẫn chi tiết cho cả unified và legacy: [ONNX.md](ONNX.md).

Chạy kiểm thử:

```powershell
python -m unittest -v test_regression.py
```

Có thể dùng `python main.py` nếu muốn thao tác qua menu.

Unified với food gate:

```powershell
python scripts\build_unified_manifest.py --negative-root data_negative
python train_unified.py
python evaluate_unified.py
python export_unified_onnx.py
```

Để giữ nhiều unified ONNX cho A/B test, export checkpoint của từng run bằng
`--name` khác nhau. Web sẽ tự liệt kê tên, kiến trúc và training method:

```powershell
python export_unified_onnx.py `
  --checkpoint runs\<run>\best_model.pth `
  --name mobilenet_v3_small_v1 `
  --display-name "MobileNetV3 Small v1"
```

Legacy cũng có thể giữ nhiều bộ router + năm expert. Sau khi train đủ sáu task
của cùng một kiến trúc, export thành bundle riêng:

```powershell
python export_onnx.py `
  --bundle-name resnet18_v1 `
  --display-name "ResNet18 legacy v1"
```

Web local so sánh unified ONNX với pipeline legacy router → expert trực tiếp
trên cùng ảnh test positive và negative:

```powershell
python -m local_web.server
```

Sau đó mở `http://127.0.0.1:8000`. Web có chế độ xem từng ảnh và đánh giá
25/100/500/1.000/5.000 hoặc toàn bộ ảnh phù hợp bộ lọc trong
`data_split/test` và `data_negative/test` bằng ONNX Runtime. Snapshot
legacy mặc định lấy từ `models/legacy/original`; unified chưa export vẫn không
ảnh hưởng việc đánh giá model cũ. Mỗi lần đánh giá sinh CSV chi tiết từng cặp
ảnh–model và JSON metric riêng cho từng model trong `reports/web_evaluations/`,
có nút tải trên web. Hai danh sách checkbox cho phép chọn đồng thời nhiều
Unified ONNX và nhiều Legacy bundle; chế độ **Cả hai** chạy mọi lựa chọn trên
cùng tập ảnh.
Mỗi lượt batch dùng seed mới nên chạy lại sẽ lấy nhóm ảnh khác; seed
được lưu trong JSON report và trong một lượt không lặp cùng đường dẫn ảnh.
Kết quả batch còn benchmark tốc độ riêng cho từng model/bundle: thời gian
ONNX trung bình theo ảnh, P95 theo batch, throughput và dung lượng artifact.
Phép đo warm-up trước, chạy bằng CPU và loại thời gian đọc/tiền xử lý ảnh;
Legacy chỉ tính router cùng expert theo route dự đoán, không tính Oracle.
Đây là số liệu so sánh trên máy đang chạy web, không phải latency điện thoại.

## Tài liệu

- [DATA.md](DATA.md): cấu trúc, số lượng và cách tạo split.
- [TRAIN.md](TRAIN.md): huấn luyện, resume và output.
- [ARCHITECTURE.md](ARCHITECTURE.md): kiến trúc router–expert.
- [ALGORITHM.md](ALGORITHM.md): thuật toán dữ liệu, train và inference.
- [MODEL_COMPARISON.md](MODEL_COMPARISON.md): quy tắc so sánh model.
- [PROJECT_FILES.md](PROJECT_FILES.md): vai trò từng file còn lại.
- [UNIFIED.md](UNIFIED.md): pipeline một model chạy song song với legacy.
- [ONNX.md](ONNX.md): export, verify và chạy web với ONNX.

Kết quả chỉ là ước lượng từ hình ảnh, không thay thế kiểm nghiệm an toàn thực phẩm.
