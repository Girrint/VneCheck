# VneCheck Model Lab

Web benchmark chạy ONNX trực tiếp trong trình duyệt. Ảnh không được tải lên server.
Model mặc định hiện hành là run detector `yolo11l_640_e30_b4` cùng 30
classifier có `training_max_epochs = 30`.

## Chạy web ngay khi model đang train

```powershell
.\.venv\Scripts\python.exe scripts\serve_web.py --open
```

Mở `http://127.0.0.1:8080`, bật **Demo giao diện**, chọn một hoặc nhiều ảnh và nhấn **Chạy toàn bộ pipeline**. Kết quả demo luôn mang nhãn `SIMULATED DATA` và không được dùng làm số liệu nghiên cứu.

## Đồng bộ model sau khi train

```powershell
.\.venv\Scripts\python.exe scripts\prepare_web_models.py --require-complete
```

Mặc định script tạo hard-link, vì vậy không nhân đôi khoảng 1,5 GB ONNX. Nếu model nằm trên ổ khác hoặc cần một bản độc lập:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_web_models.py --mode copy --require-complete
```

Sau đó tải lại trang, tắt **Demo giao diện** và chạy benchmark thật.

Lệnh đồng bộ thành công phải in:

```text
Detector: 1/1
Classifiers: 30/30
```

Sau khi thay model, dùng `Ctrl+F5` để trình duyệt tải lại manifest và artifact e30.

Nếu detector được train dưới tên run khác:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_web_models.py `
  --detector-run ten_thu_muc_run `
  --require-complete
```

## Chức năng

- Hai tab độc lập nhưng dùng chung ROI trong bộ nhớ: YOLO & Crop và Benchmark.
- YOLO11l ONNX: letterbox 640×640, decode output và class-aware NMS.
- Classifier ONNX: crop ROI có padding 3%, resize-pad 224×224, contrast, sharpening và ImageNet normalization.
- WebGPU với WASM fallback.
- Chọn một classifier hoặc so sánh cả ba.
- Warm-up một lần, sau đó đo riêng model load time và inference latency trung bình.
- Hiển thị kích thước ONNX và Validation Macro-F1 từ metadata.
- Confidence và IoU NMS điều chỉnh được.
- Vẽ bounding box trên ảnh.
- Chọn, kéo thả và đánh giá theo lô tối đa 20 ảnh; mỗi ảnh tối đa 20 MB.
- Xử lý tuần tự để tránh nạp nhiều lượt suy luận đồng thời vào GPU hoặc bộ nhớ trình duyệt.
- Danh sách ảnh cuộn riêng bên cạnh khung preview, cho phép chọn hoặc xóa từng ảnh mà không làm thay đổi vị trí nút chạy.
- Chuyển nhanh giữa từng ảnh trong lô và giữ riêng box, ROI cùng kết quả benchmark.
- Hiển thị từng ROI đã crop cùng nhãn YOLO và kết quả của các classifier được định tuyến.
- Xuất một báo cáo JSON hoặc CSV tổng hợp cho toàn bộ ảnh đã xử lý.

Tab **YOLO & Crop** nhận một lô ảnh và chạy tuần tự toàn bộ pipeline trong một lệnh: detector, vẽ box, crop ROI rồi chạy các classifier được chọn. Thanh ảnh thu nhỏ cho phép chọn kết quả của từng ảnh. Tab này trình bày ảnh đang chọn với box và gallery crop. Tab **Benchmark** không có nút chạy; nó đọc ngay kết quả tương ứng, đặt thông tin Fresh/Rotten cùng confidence, latency, load time, kích thước ONNX và Validation F1 ở bên trái, ảnh crop ở bên phải. Người dùng có thể chuyển qua lại giữa hai tab hoặc giữa các ảnh mà không mất kết quả.

Phần so sánh lớn ở đầu Benchmark chỉ hiển thị **ROI đang chọn**, mặc định là ROI 01.
Để xem ROI khác, chọn ROI trong tab **Kết quả AI** hoặc thanh **Đối tượng đã phát hiện**
ở tab **Nhận diện**, rồi quay lại Benchmark. Bảng **Chi tiết hiệu năng** phía dưới luôn
liệt kê prediction của tất cả ROI thuộc ảnh hiện tại. Với lô nhiều ảnh, chọn ảnh trong
**Danh sách ảnh** trước khi chọn ROI.

## Kiểm tra end-to-end

Một ca kiểm thử chỉ đạt khi hệ thống phát hiện đúng loại sản phẩm, crop đúng đối tượng và
phân loại đúng Fresh/Rotten. Nên kiểm tra ảnh một/nhiều đối tượng, nền phức tạp, vật thể
nhỏ hoặc che khuất, ảnh không có sản phẩm và nhiều mẫu `cucumber`. Xuất JSON/CSV sau mỗi
lô để đối chiếu với ground truth; không dùng dữ liệu `SIMULATED DATA` làm kết quả.

Lần chạy đầu tiên có thể chậm do trình duyệt phải tải và biên dịch ONNX. `Load` được tách riêng khỏi `Latency TB`; những lần dùng lại model sẽ hiển thị `cached`.

## Cấu trúc model

```text
web/models/
├── manifest.json
├── detector/
│   ├── yolo11l_final.onnx
│   └── export_info.json
└── classifiers/
    ├── mobilenet_v3_large/<produce>/
    ├── resnet18/<produce>/
    └── efficientnet_b0/<produce>/
```

Không mở `web/index.html` trực tiếp bằng `file://`, vì trình duyệt sẽ chặn việc tải ONNX và manifest. Luôn chạy `scripts/serve_web.py`.
