# Export và kiểm tra ONNX

Tài liệu này áp dụng cho hai pipeline chạy song song:

- **Unified:** một checkpoint PyTorch → một ONNX có ba output.
- **Legacy:** router và năm freshness checkpoint → sáu ONNX riêng.

ONNX chỉ là bản inference của checkpoint `.pth`. Export không train model và
không làm model tốt hơn; cần có checkpoint đã train trước.

## 1. Chuẩn bị môi trường

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Kiểm tra các lệnh và tham số mà không chạy export:

```powershell
python export_unified_onnx.py --help
python export_onnx.py --help
python verify_unified_onnx.py --help
python verify_onnx.py --help
```

## 2. Export unified ONNX

Checkpoint mặc định phải tồn tại:

```text
models/unified/food_multitask.pth
```

Nếu chưa có, tạo manifest rồi train:

```powershell
python scripts\build_unified_manifest.py
python train_unified.py
```

Export:

```powershell
python export_unified_onnx.py
```

Output:

```text
models/unified/food_multitask.onnx
models/unified/food_multitask.manifest.json
models/unified/manifest.json
```

ONNX có input `image` với shape động `[batch, 3, 224, 224]` và ba output:

```text
food_logits       = SUPPORTED / UNSUPPORTED / NON_FOOD
category_logits   = cu / gia_cam / gia_suc / qua / rau
freshness_logits  = FRESH / HALF-FRESH / SPOILED
```

Export checkpoint hoặc output khác:

```powershell
python export_unified_onnx.py `
  --checkpoint runs\<run_unified>\best_model.pth `
  --output models\unified\food_multitask.onnx
```

Giữ nhiều artifact để chọn và so sánh trên web:

```powershell
python export_unified_onnx.py `
  --checkpoint runs\<run_unified>\best_model.pth `
  --name mobilenet_v3_small_v1 `
  --display-name "MobileNetV3 Small v1"
```

Lệnh trên tạo:

```text
models/unified/exports/mobilenet_v3_small_v1.onnx
models/unified/exports/mobilenet_v3_small_v1.manifest.json
```

Không dùng lại cùng `--name` nếu muốn giữ artifact cũ. Web tự quét đệ quy
`models/unified/**/*.onnx`; có thể đổi thư mục bằng `--onnx-root`.

So sánh output PyTorch và ONNX trên một ảnh test:

```powershell
python verify_unified_onnx.py `
  --image data_split\test\qua\FRESH\apple\<ten_anh>.jpg
```

Kết quả hợp lệ khi cả `food`, `category`, `freshness` đều báo `PASS`.

## 3. Export sáu legacy ONNX

Snapshot `models/legacy/original` đã có sẵn sáu ONNX, vì vậy web local không
cần export lại để đánh giá baseline cũ.

Muốn export checkpoint hiện hành trong `models/`:

```powershell
python export_onnx.py
```

Lệnh trên ghi hoặc ghi đè:

```text
models/router.onnx
models/freshness/cu.onnx
models/freshness/gia_cam.onnx
models/freshness/gia_suc.onnx
models/freshness/qua.onnx
models/freshness/rau.onnx
```

Để không ghi đè artifact hiện hành, export vào thư mục mới:

```powershell
python export_onnx.py `
  --checkpoint-root models\legacy\original `
  --output-root models\legacy\reexported
```

Kiểm tra sáu ONNX vừa export:

```powershell
python verify_onnx.py `
  --image data_split\test\cu\FRESH\carrot\<ten_anh>.jpg `
  --checkpoint-root models\legacy\original `
  --onnx-root models\legacy\reexported
```

`cu`, `qua`, `rau` có expert 2 lớp. `gia_cam`, `gia_suc` có expert 3 lớp.

## 4. Mở web đánh giá

Mặc định web dùng unified ONNX hiện hành và snapshot legacy nguyên bản:

```powershell
python -m local_web.server
```

Mở `http://127.0.0.1:8000`.

Dùng ONNX ở đường dẫn khác:

```powershell
python -m local_web.server `
  --onnx models\unified\food_multitask.onnx `
  --legacy-root models\legacy\reexported
```

Web vẫn hoạt động nếu chỉ một trong hai pipeline sẵn sàng.
Batch evaluation hỗ trợ 25/100/500/1.000/5.000 hoặc toàn bộ ảnh phù hợp bộ lọc.
Khi chạy toàn bộ 14.162 ảnh bằng CPU, request có thể mất vài phút; trình duyệt
giữ trạng thái chờ đến khi server cộng dồn xong metric.

Kết thúc đánh giá, dùng **Tải CSV chi tiết** để lấy một dòng kết quả cho từng
ảnh hoặc **Tải JSON tổng hợp** để lấy metric. File cũng được lưu tại:

```text
reports/web_evaluations/evaluation_<timestamp>_<id>.csv
reports/web_evaluations/evaluation_<timestamp>_<id>.json
```

Sau batch, khung bên phải hiển thị từng ảnh theo trang 20 ảnh, cùng truth,
dự đoán, confidence và probability như chế độ đánh giá một ảnh. Bộ lọc và nút
điều khiển nằm cố định ở cột trái trên màn hình desktop.

## 5. Lỗi thường gặp

### `Thiếu unified checkpoint`

Chưa có `models/unified/food_multitask.pth`. Chạy `train_unified.py` trước hoặc
truyền đúng `--checkpoint`.

### Web báo `Thiếu unified ONNX`

Chạy `python export_unified_onnx.py`. Legacy vẫn có thể được đánh giá độc lập.

### `Missing checkpoint` hoặc `Missing ONNX`

Kiểm tra `--checkpoint-root` và `--onnx-root`. Hai thư mục phải cùng đại diện
cho một phiên bản model để phép so sánh PyTorch/ONNX có ý nghĩa.

### Verify báo `FAIL`

Không dùng ONNX được export từ checkpoint khác. Export lại đúng checkpoint rồi
verify; sai số tối đa yêu cầu nhỏ hơn `1e-5`.

### Food gate hiện `NOT_TRAINED`

Checkpoint unified chưa được train với cả `UNSUPPORTED` và `NON_FOOD` trong ba
split. Category/freshness vẫn chạy, nhưng chưa được phép dùng head này để từ
chối ảnh không phù hợp.
