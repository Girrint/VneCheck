# VneCheck Training

VneCheck là dự án học sâu hai tầng dùng để phát hiện rau củ quả trong ảnh và đánh giá từng đối tượng là `fresh` hoặc `rotten`. Source được thiết kế để train trên Windows, lưu đầy đủ checkpoint và export tất cả mô hình sang ONNX để benchmark trên web.

## Kiến trúc hệ thống

```text
Ảnh đầu vào
    │
    ▼
YOLO11l pretrained
Phát hiện loại sản phẩm và bounding box
    │
    ├── apple ───────► Apple classifier ───────► fresh / rotten
    ├── banana ──────► Banana classifier ──────► fresh / rotten
    ├── bellpepper ──► Bellpepper classifier ──► fresh / rotten
    └── ...
```

Tầng 1 sử dụng một mô hình YOLO11l pretrained để phát hiện 10 loại sản phẩm:

```text
apple, banana, bellpepper, carrot, cucumber,
mango, orange, potato, strawberry, tomato
```

Tầng 2 sử dụng bộ phân loại nhị phân riêng cho từng loại. Ba kiến trúc được so sánh:

- MobileNetV3-Large
- ResNet-18
- EfficientNet-B0

Mỗi kiến trúc có 10 mô hình, tương ứng 10 loại sản phẩm. Toàn bộ thí nghiệm tạo ra:

```text
1 YOLO11l + 30 classifier = 31 mô hình ONNX
```

Các classifier được train độc lập trên cùng bộ ROI. Việc train hoặc thay đổi classifier không yêu cầu train lại YOLO.

## Trạng thái thực nghiệm hiện tại

Phiên bản đang hoạt động là **e30**:

- YOLO11l được fine-tune tối đa 30 epoch; run mặc định là `yolo11l_640_e30_b4`.
- Mỗi classifier được train tối đa 30 epoch, với early stopping patience 15.
- Model triển khai web gồm 1 detector và 30 classifier e30.
- Báo cáo chính thức nằm trong `reports/detection_test_e30/` và
  `reports/classification_test_e30/`.
- Artifact e15 và smoke đã được loại khỏi cây dự án; pipeline, báo cáo và web chỉ dùng e30.

## Dữ liệu hiện tại

Dataset nguồn đã chuẩn hóa đang hoạt động tại `data/`; dữ liệu cũ không còn nằm trong
cây dự án.

| Tập nguồn | Số ảnh | Số bounding box |
|---|---:|---:|
| Train | 8.185 | 14.900 |
| Validation | 1.751 | 3.197 |
| Test | 1.760 | 3.175 |
| Tổng | 11.696 | 21.272 |

Dataset detection được chia lại theo ảnh nguồn để các biến thể Roboflow của cùng một ảnh không xuất hiện ở nhiều split:

| Tập detection | Số ảnh |
|---|---:|
| Train | 8.137 |
| Validation | 1.786 |
| Test | 1.773 |

Dataset classification mới có 21.185 ROI. Có 87 box nhỏ hơn 24×24 pixel bị loại và
không còn box chưa xác định `fresh/rotten`.

## Cấu trúc dự án

```text
VneCheck_YoLo/
├── configs/
│   ├── classes.json                 # Mapping 19 ID nguồn về 10 lớp detection
│   ├── detector_yolo11l.json        # Cấu hình train YOLO11l
│   └── classifiers.json             # Cấu hình ba kiến trúc classification
├── data/                             # Dataset nguồn đang hoạt động
├── datasets/
│   ├── detection/                    # Dataset YOLO 10 lớp, đã chống leakage
│   ├── classification/               # ROI fresh/rotten cho từng loại
│   └── e2e/                          # Bộ ảnh và nhãn kiểm thử end-to-end
├── reports/                          # Audit dữ liệu và báo cáo e30 hiện hành
├── runs/
│   ├── detection/                   # Checkpoint, biểu đồ và ONNX YOLO
│   └── classification/              # 30 checkpoint và ONNX classifier
├── scripts/
│   ├── bootstrap.ps1                # Cài môi trường
│   ├── train_all.ps1                # Train và export toàn bộ bằng một lệnh
│   └── verify_environment.py        # Kiểm tra CUDA và thư viện
├── src/vnecheck/
│   ├── data/
│   │   ├── audit.py                 # Kiểm tra ảnh, label, box và leakage
│   │   ├── prepare_detection.py     # Chuyển nhãn và chia dataset detection
│   │   └── crop_rois.py             # Crop Ground Truth ROI cho tầng 2
│   ├── detection/
│   │   ├── train.py                 # Train và export YOLO11l
│   │   └── evaluate.py              # Đánh giá detector trên test
│   ├── classification/
│   │   ├── transforms.py            # Tiền xử lý và augmentation ROI
│   │   ├── train.py                 # Train classifier
│   │   └── export.py                # Export và kiểm tra classifier ONNX
│   └── train_all.py                 # Điều phối toàn bộ thí nghiệm
├── tests/
├── yolo11l.pt                       # Trọng số YOLO11l pretrained
└── pyproject.toml
```

`datasets/`, `runs/`, `reports/`, checkpoint và ONNX là artifact sinh tự động và không nên commit vào Git.

## Môi trường

Cấu hình đã kiểm tra:

- Windows 10
- Python 3.12
- NVIDIA GeForce GTX 1650 4 GB
- PyTorch 2.14.0 CUDA 12.6
- TorchVision 0.29.0
- Ultralytics 8.4.126
- OpenCV 5.0.0

Tạo môi trường và cài thư viện:

```powershell
cd C:\Project\VneCheck_YoLo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[dev]"
```

Hoặc sử dụng script:

```powershell
.\scripts\bootstrap.ps1 -TorchIndexUrl https://download.pytorch.org/whl/cu126
```

Kiểm tra CUDA:

```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

Kết quả bắt buộc phải có:

```text
cuda_available: true
gpu: NVIDIA GeForce GTX 1650
```

Checkpoint pretrained phải nằm tại:

```text
C:\Project\VneCheck_YoLo\yolo11l.pt
```

## Chuẩn bị dữ liệu

### 1. Audit dữ liệu nguồn

```powershell
.\.venv\Scripts\python.exe -m vnecheck.data.audit `
  --data data `
  --output reports\data_audit
```

Audit kiểm tra:

- ảnh và label tương ứng;
- định dạng YOLO;
- class ID;
- bounding box vượt biên;
- ảnh không có box;
- số instance từng lớp;
- ảnh nguồn xuất hiện ở nhiều split.

### 2. Tạo dataset detection

```powershell
.\.venv\Scripts\python.exe -m vnecheck.data.prepare_detection `
  --source data `
  --output datasets\detection
```

Pipeline ánh xạ 19 ID nguồn thành 10 lớp và chia lại theo `group_id`. Ảnh được hard-link khi có thể; nội dung ảnh gốc không bị chỉnh sửa.

### 3. Xác minh Fresh và Rotten

Các box không suy ra được độ tươi được ghi vào:

```text
datasets/detection/review_freshness.csv
```

Chỉ điền cột `freshness` bằng `fresh` hoặc `rotten`. Không sửa những cột còn lại.

### 4. Crop ROI cho classification

```powershell
.\.venv\Scripts\python.exe -m vnecheck.data.crop_rois
```

ROI được crop từ Ground Truth box, thêm padding 3% và bỏ crop nhỏ hơn 24×24 pixel.
Đầu ra mặc định là `datasets/classification`; split của ROI luôn giữ theo ảnh nguồn
để tránh leakage.

## Thuật toán tầng 1

YOLO11l sử dụng trọng số pretrained trên COCO rồi fine-tune với 10 lớp VneCheck.

Cấu hình mặc định trong `configs/detector_yolo11l.json`:

| Tham số | Giá trị |
|---|---:|
| Model | `yolo11l.pt` |
| Input | 640×640 |
| Epoch | 30 |
| Batch | 4 |
| Freeze | 10 |
| AMP | Có |
| Seed | 42 |
| Early stopping patience | 30 |

Ultralytics thực hiện resize giữ tỷ lệ, letterbox 640×640, padding màu `(114,114,114)` và chuẩn hóa ảnh trong batch.

Augmentation online:

- HSV hue 0,01;
- saturation và brightness 0,25;
- rotation ±10°;
- translation ±5%;
- scale ±30%;
- horizontal flip 50%;
- mosaic 50%, tắt trong 3 epoch cuối.

Không dùng Contrast Enhancement, Sharpening hoặc Random Erasing cho detector.

## Thuật toán tầng 2

Mỗi loại sản phẩm có một bài toán phân loại nhị phân độc lập:

```text
ROI → Fresh hoặc Rotten
```

Ba kiến trúc đều được khởi tạo ngẫu nhiên bằng `weights=None` để so sánh công bằng theo thiết kế nghiên cứu.

Cấu hình mặc định trong `configs/classifiers.json`:

| Tham số | Giá trị |
|---|---:|
| Input | 224×224 |
| Epoch | 30 |
| Batch | 8 |
| Optimizer | AdamW |
| Learning rate | 0,001 |
| Weight decay | 0,0001 |
| Loss | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |
| Early stopping patience | 15 |
| Tiêu chí checkpoint | Validation Macro-F1, sau đó Validation Loss |
| Gradient clipping | 5,0 |
| AMP | Tắt, dùng FP32 |

FP32 được dùng cho cả ba kiến trúc vì ResNet-18 sinh `NaN` với AMP trên GTX 1650. Giữ cùng chế độ số học giúp kết quả so sánh nhất quán.

Tiền xử lý cố định:

1. EXIF transpose và chuyển RGB.
2. Resize giữ tỷ lệ, padding `(114,114,114)` thành 224×224.
3. Contrast Enhancement với hệ số 1,2.
4. Sharpening với hệ số 1,3 và kernel `1/13 × [[1,1,1],[1,5,1],[1,1,1]]`.
5. Chuẩn hóa theo ImageNet mean/std.

Augmentation chỉ áp dụng cho train:

- rotation ±120°;
- brightness 0,9–1,5;
- contrast 0,9–1,5;
- horizontal flip 50%;
- Random Erasing 10 vùng, mỗi vùng 15%×15%.

Validation và test không có augmentation ngẫu nhiên.

## Smoke test

Smoke test YOLO với 2% dữ liệu:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11l.pt').train(data='datasets/detection/data.yaml', epochs=1, imgsz=640, batch=4, fraction=0.02, freeze=10, workers=0, amp=True, cache=False, device=0, project=r'C:\Project\VneCheck_YoLo\runs\detection', name='yolo11l_640_b4_smoke')"
```

Smoke test một classifier:

```powershell
.\.venv\Scripts\python.exe -m vnecheck.classification.train `
  --architecture resnet18 `
  --produce apple `
  --max-epochs 1 `
  --batch-size 8
```

Checkpoint smoke chỉ kiểm tra pipeline, không dùng làm kết quả nghiên cứu.

## Train và export toàn bộ bằng một lệnh

```powershell
cd C:\Project\VneCheck_YoLo
.\scripts\train_all.ps1
```

Script chạy tuần tự để không có hai tiến trình tranh chấp GTX 1650:

1. Train YOLO11l trên `datasets/detection` trong 30 epoch, batch 4.
2. Export và kiểm tra `yolo11l_final.onnx`.
3. Train 10 MobileNetV3-Large, mỗi mô hình 30 epoch, batch 8.
4. Export và kiểm tra từng ONNX.
5. Train và export 10 ResNet-18.
6. Train và export 10 EfficientNet-B0.

Sau mỗi classifier, script lập tức export ONNX trước khi chuyển sang mô hình kế tiếp. Nếu tiến trình bị dừng, chạy lại cùng lệnh; các ONNX đã hoàn thành đủ 30 epoch sẽ được bỏ qua.

Tùy chọn:

```powershell
# Chỉ train 30 classifier
.\scripts\train_all.ps1 --skip-detector

# Chỉ train YOLO
.\scripts\train_all.ps1 --skip-classifiers

# Train lại tất cả từ đầu
.\scripts\train_all.ps1 --force
```

Model và báo cáo được ghi vào `runs/` và `reports/`. Không chạy YOLO và classifier
đồng thời trên GPU 4 GB.

## Train riêng từng thành phần

Train YOLO theo cấu hình mặc định:

```powershell
.\.venv\Scripts\python.exe -m vnecheck.detection.train `
  --name yolo11l_640_e30_b4
```

Train một classifier và tự động export ONNX:

```powershell
.\.venv\Scripts\python.exe -m vnecheck.classification.train `
  --architecture efficientnet_b0 `
  --produce carrot
```

Train và export toàn bộ classifier mà không chạy YOLO:

```powershell
.\.venv\Scripts\python.exe -m vnecheck.classification.train `
  --architecture all `
  --produce all
```

## Đầu ra ONNX

YOLO:

```text
runs/detection/yolo11l_640_e30_b4/
├── weights/
│   ├── best.pt
│   └── last.pt
└── export/
    ├── yolo11l_final.onnx
    └── export_info.json
```

Classifier:

```text
runs/classification/<architecture>/<produce>/
├── best.pt
├── history.csv
├── <produce>_<architecture>.onnx
└── <produce>_<architecture>.json
```

ONNX detector:

```text
Input:  images [1, 3, 640, 640]
Output: [1, 14, 8400]
```

`14 = 4` tọa độ box `+ 10` điểm lớp.

ONNX classifier:

```text
Input:  images [1, 3, 224, 224]
Output: logits [1, 2]
Class order: [fresh, rotten]
```

Mỗi ONNX được kiểm tra cấu trúc bằng ONNX Checker. Classifier còn được chạy inference thử bằng ONNX Runtime để xác nhận output `[1,2]`. File JSON đi kèm lưu preprocessing, class order, input shape, checkpoint nguồn và Validation Macro-F1 để web benchmark đọc đúng cấu hình.

## Đánh giá

Detector:

```powershell
.\.venv\Scripts\python.exe -m vnecheck.detection.evaluate `
  --weights runs\detection\yolo11l_640_e30_b4\weights\best.pt `
  --data datasets\detection\data.yaml `
  --output reports\detection_test_e30 `
  --batch 4
```

Chỉ số detector:

- Precision và Recall;
- mAP50;
- mAP50-95;
- chỉ số từng lớp;
- confusion matrix và PR/F1 curve;
- thời gian suy luận.

Đánh giá toàn bộ 30 classifier trên split `test` đã đóng băng:

```powershell
.\.venv\Scripts\python.exe -m vnecheck.classification.evaluate `
  --data datasets\classification `
  --runs runs\classification `
  --output reports\classification_test_e30 `
  --batch-size 8
```

Kết quả được lưu tại:

```text
reports/classification_test_e30/
├── per_model.csv
├── architecture_summary.csv
├── summary.json
└── <architecture>/<produce>/
    ├── metrics.json
    ├── predictions.csv
    ├── confusion_matrix.csv
    └── confusion_matrix.png
```

Chỉ số classifier:

- Accuracy;
- Precision và Recall;
- Macro-F1;
- F1 Fresh và F1 Rotten;
- ROC-AUC;
- confusion matrix;
- test loss và thời gian suy luận trung bình mỗi ảnh.

`scripts/train_all.ps1` tự chạy cả detector test và classifier test ở bước cuối. Dùng `--skip-evaluation` khi chỉ muốn train/export mà chưa muốn mở tập test. Báo cáo test hoàn chỉnh sẽ được bỏ qua khi chạy lại, trừ khi dùng `--force`.

### Kết quả test e30 với `data`

Các số liệu dưới đây được đo sau khi train xong trên split `test` độc lập của bộ dữ liệu
mới. YOLO dùng 1.773 ảnh test; mỗi kiến trúc classifier dùng 3.181 ROI test đã đóng băng.

YOLO11l trên split test:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 94,90% | 94,68% | 97,59% | 88,83% |

Classifier trên 3.181 ROI test cho mỗi kiến trúc:

| Kiến trúc | Accuracy | Macro-F1 | Recall Rotten | ROC-AUC | Latency/ảnh |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 96,49% | 96,21% | 95,15% | 98,55% | 2,90 ms |
| MobileNetV3-Large | 96,10% | 95,85% | 95,74% | 98,30% | 2,71 ms |
| ResNet-18 | 95,16% | 94,83% | 94,05% | 98,35% | 2,80 ms |

EfficientNet-B0 đạt Accuracy và Macro-F1 cao nhất; MobileNetV3-Large đạt Recall Rotten
cao nhất. Latency được đo bằng PyTorch CUDA trên máy hiện tại, không phải latency ONNX
trong trình duyệt. Dữ liệu chi tiết theo từng loại sản phẩm nằm trong
`reports/classification_test_e30/per_model.csv`; kết quả YOLO theo lớp nằm trong
`reports/detection_test_e30/test_per_class.csv`.

Precision riêng lớp `cucumber` thấp hơn các lớp còn lại và cần được theo dõi trong các
lần đánh giá bổ sung.

## Luồng triển khai web

```text
Ảnh
→ YOLO11l ONNX
→ decode box + NMS
→ crop từng ROI
→ xác định loại sản phẩm
→ áp dụng preprocessing 224×224
→ classifier ONNX tương ứng
→ softmax logits
→ Fresh hoặc Rotten
```

Web benchmark phải đọc metadata JSON của từng classifier để áp dụng đúng contrast, sharpening, padding, normalization và class order.

## Web benchmark trong trình duyệt

Source web nằm trong `web/` và có thể phát triển, xem giao diện trong khi tiến trình train vẫn chạy. Web dùng ONNX Runtime Web, ưu tiên WebGPU và tự chuyển sang WebAssembly khi cần. Không chạy benchmark thật đồng thời với quá trình train nếu muốn số đo GPU chính xác.

Chạy giao diện local:

```powershell
.\.venv\Scripts\python.exe scripts\serve_web.py --open
```

Trong lúc chưa đủ model, bật **Demo giao diện**. Kết quả mô phỏng được đánh dấu rõ và không dùng làm benchmark.

Sau khi toàn bộ quá trình train hoàn tất, đồng bộ 31 ONNX bằng hard-link và kiểm tra đủ model:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_web_models.py --require-complete
```

Sau đó tải lại `http://127.0.0.1:8080`, tắt Demo và chạy phép đo thật. Web cho phép chọn hoặc kéo thả tối đa 20 ảnh, xử lý tuần tự, giữ kết quả riêng cho từng ảnh, vẽ box, so sánh ba classifier, đo thời gian tải và latency, đồng thời xuất JSON/CSV tổng hợp cho cả lô. Xem hướng dẫn chi tiết tại `web/README.md`.

## Đánh giá end-to-end

Đánh giá end-to-end sử dụng ảnh nguyên bản chưa xuất hiện trong train/validation và đo
toàn bộ chuỗi `ảnh → detection → crop ROI → classifier → Fresh/Rotten`. Một đối tượng
chỉ được tính đúng hoàn toàn khi được phát hiện đúng loại, bounding box đạt IoU tối thiểu
0,5 và trạng thái Fresh/Rotten đúng ground truth.

Benchmark hiện hành được chạy bằng `scripts/evaluate_e2e.py` trên `datasets/e2e/`:

- 371 ảnh, 1.350 ground-truth box và 1.292 predicted box.
- Ghép box class-agnostic tại IoU ≥ 0,5: Precision 97,29%, Recall 93,11%, F1 95,16%.
- Mean IoU của 1.257 box ghép được là 92,14%.
- Ba classifier đồng thuận trên 90,79% ROI được detector tạo ra.

Bộ nhãn E2E hiện có ID 0–14, gồm 47 box ngoài dải ID 0–9 của detector, và chưa có
ground truth Fresh/Rotten theo object. Vì vậy chưa thể báo cáo Accuracy end-to-end theo
lớp sản phẩm hoặc độ tươi. Kết quả đầy đủ nằm trong `reports/e2e_test_e30/`.

Quy trình chạy lại:

1. Đồng bộ đủ 31 model e30 bằng `scripts/prepare_web_models.py --require-complete`.
2. Chạy `scripts/evaluate_e2e.py` để tái tạo thống kê detector và kết quả inference của
   ba classifier.
3. Sau khi hoàn thiện mapping nhãn và ground truth Fresh/Rotten, bổ sung Accuracy theo
   object, Miss Rate, False Positive Rate và Freshness Accuracy có điều kiện.

Trong giao diện, phần đầu tab Benchmark chỉ hiển thị ROI đang chọn. Để chuyển ROI, chọn
đối tượng trong tab **Kết quả AI** hoặc thanh ROI của tab **Nhận diện**, sau đó quay lại
Benchmark. Bảng **Chi tiết hiệu năng** chứa prediction của tất cả ROI trong ảnh đang chọn.

## Kiểm thử source

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Không dùng kết quả từ smoke checkpoint để so sánh mô hình hoặc đưa lên web benchmark.
