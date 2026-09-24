# VnGrocery AI

Hệ thống Python phân loại nhóm thực phẩm và ước lượng độ tươi bằng mô hình CNN pretrained từ `torchvision`, mặc định là MobileNetV3-Large.

## Mục lục tài liệu

- [Kiến trúc](ARCHITECTURE.md): router, freshness expert và luồng suy luận.
- [Thuật toán](ALGORITHM.md): thuật toán hoạt động, tiền xử lý, forward và điểm tươi.
- [Dữ liệu](DATA.md): cấu trúc dataset, nguồn dữ liệu và kiểm soát leakage.
- [Huấn luyện](TRAIN.md): cài đặt, train, resume, đánh giá và export.
- [Nguyên lý thư viện](LIBRARY_PRINCIPLES.md): transfer learning, loss, optimizer và metric.
- [So sánh model](MODEL_COMPARISON.md): baseline, sơ đồ CNN, đánh giá đơn và đa tầng.

## Trạng thái hiện tại

- Router phân loại 5 nhóm: `cu`, `gia_cam`, `gia_suc`, `qua`, `rau`.
- Freshness expert hiện không đồng nhất số lớp:
  - `gia_cam`, `gia_suc`: `FRESH`, `HALF-FRESH`, `SPOILED`.
  - `cu`, `qua`, `rau`: `FRESH`, `SPOILED`.
- Preprocessing chuẩn cho validation/inference: RGB, `Resize(232)`, `CenterCrop(224)`, ImageNet normalization.
- Dataset hiện có ảnh trùng giữa `train` và `valid`; metric hiện tại chưa phải bằng chứng tổng quát hóa.
- `evaluate_end_to_end.py` hiện chưa chạy được với expert 2 lớp và 3 lớp trộn lẫn; cần sửa contract trước khi dùng.

## Cài đặt nhanh

PowerShell tại thư mục dự án:

```powershell
winget install --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra môi trường:

```powershell
python --version
python -m pip check
```

## Chạy các công cụ chính

Dự đoán PyTorch:

```powershell
python predict.py path\to\image.jpg
```

Kiểm tra ảnh trùng:

```powershell
python scripts\check_duplicates.py
```

Đánh giá end-to-end và uncertainty chỉ chạy sau khi contract freshness được thống nhất:

```powershell
python evaluate_end_to_end.py --root data\valid
python evaluate_router_uncertainty.py --root data\valid --threshold 0.8
```

Export và verify ONNX:

```powershell
python export_onnx.py
python verify_onnx.py --image path\to\image.jpg
```

Tạo manifest model:

```powershell
python package_models.py --require-onnx
```

Chạy regression tests:

```powershell
python -m unittest -v test_regression.py
```

## Rủi ro trước khi phát hành

1. Tạo lại split theo group/source/product; không chia ngẫu nhiên các ảnh cùng nguồn sang nhiều split.
2. Xử lý toàn bộ duplicate và near-duplicate, sau đó train lại.
3. Chốt freshness contract 2 lớp hoặc 3 lớp; checkpoint, evaluator, client và tài liệu phải cùng contract.
4. Sửa evaluator end-to-end để hỗ trợ contract đã chốt.
5. Chỉ chọn model MobileNetV3 sau khi đánh giá trên split sạch và test độc lập.
6. Tạo manifest và kiểm tra checksum trước khi copy sang Flutter.

Kết quả model chỉ là ước lượng từ hình ảnh, không thay thế kiểm nghiệm an toàn thực phẩm.
