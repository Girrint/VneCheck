# Hướng dẫn huấn luyện

## Menu tương tác

Khởi động menu chính:

```powershell
python main.py
```

Menu hỗ trợ train router, train từng freshness expert, dự đoán ảnh, kiểm tra
dataset và có trang trợ giúp bằng tiếng Anh.

## 1. Kích hoạt môi trường

Chạy trong PowerShell tại thư mục dự án:

```powershell
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Kiểm tra PyTorch có nhận GPU NVIDIA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Kết quả phải có `True` nếu muốn huấn luyện bằng CUDA.

## 2. Chuẩn bị dữ liệu

Dữ liệu phải có đúng cấu trúc hai tầng:

```text
data/
  train/
    gia_suc/
      FRESH/
      HALF-FRESH/
      SPOILED/
    gia_cam/
      FRESH/
      HALF-FRESH/
      SPOILED/
    rau/
      FRESH/
      HALF-FRESH/
      SPOILED/
    cu/
      FRESH/
      HALF-FRESH/
      SPOILED/
    qua/
      FRESH/
      HALF-FRESH/
      SPOILED/
  valid/
    ... cùng cấu trúc với train ...
```

Dữ liệu cũ chưa được xác định nhóm nằm trong:

```text
data/unclassified/train/
data/unclassified/valid/
```

Phải phân loại các ảnh này vào đúng nhóm trước khi train. Không đặt cùng một ảnh
hoặc các bản augmentation của cùng ảnh gốc ở cả `train` và `valid`.

Nếu dataset mới chỉ có train, có thể tách 15% theo từng sản phẩm và mức độ tươi:

```powershell
python split_validation.py --categories rau cu qua --ratio 0.15
```

Expert có thể học hai lớp `FRESH/SPOILED` khi dataset không có `HALF-FRESH`.
Lớp rỗng được bỏ qua tự động và có thể bổ sung sau bằng cách train lại expert.

## 3. Huấn luyện router

Router nhận diện năm nhóm thực phẩm:

```powershell
python train.py router
```

Model tốt nhất được lưu tại:

```text
models/router.pth
```

## 4. Huấn luyện các freshness expert

Mỗi nhóm có một model đánh giá độ tươi riêng:

```powershell
python train.py freshness --category gia_suc
python train.py freshness --category gia_cam
python train.py freshness --category rau
python train.py freshness --category cu
python train.py freshness --category qua
```

Các model được lưu trong:

```text
models/freshness/gia_suc.pth
models/freshness/gia_cam.pth
models/freshness/rau.pth
models/freshness/cu.pth
models/freshness/qua.pth
```

## 5. Tùy chỉnh quá trình train

Các tùy chọn chung phải đặt trước tên task:

```powershell
python train.py --epochs 20 --batch-size 32 --workers 4 --model mobilenet_v3_large --method fine_tune router
python train.py --epochs 20 --batch-size 32 --workers 4 --model efficientnet_b0 --method fine_tune freshness --category gia_suc
```

Ý nghĩa:

- `--epochs`: số epoch tối đa, mặc định `15`.
- `--batch-size`: số ảnh trong mỗi batch, mặc định `32`.
- `--workers`: số tiến trình nạp dữ liệu, mặc định tối đa `4`.
- `--model`: kiến trúc model.
- `--method`: `fine_tune` toàn mạng hoặc `head_only` chỉ classifier.
- `--optimizer`: `adamw` hoặc `sgd`.
- `--scheduler`: `plateau` hoặc `cosine`.
- `--learning-rate`: bước cập nhật trọng số.
- `--weight-decay`: regularization cho trọng số.
- `--label-smoothing`: giảm mức quá tự tin của nhãn.
- `--seed`: cố định phép tách dữ liệu và khởi tạo ngẫu nhiên.

Với GPU 4 GB VRAM, bắt đầu bằng batch size `16` hoặc `32`. Nếu gặp lỗi hết
VRAM, giảm xuống `8`:

```powershell
python train.py --batch-size 8 router
```

Model tự dừng nếu validation loss không cải thiện trong ba epoch.

## Kết quả của mỗi lần train

Mỗi lần train tạo một thư mục riêng trong `runs/`:

```text
runs/<timestamp>_<task>_<category>_<model>/
  config.json       # toàn bộ tham số đầu vào và phiên bản thư viện
  metrics.csv       # train/valid loss, accuracy, LR và thời gian từng epoch
  best_model.pth    # checkpoint tốt nhất của riêng run
  summary.json      # test accuracy, macro F1, precision/recall/F1 từng lớp
```

Checkpoint tốt nhất đồng thời được copy vào `models/` để pipeline dự đoán sử
dụng. Chọn mục `Training results` trong menu để so sánh các run đã hoàn thành.

## Tuning tham số

Chọn `Hyperparameter tuning` trong menu hoặc chạy:

```powershell
python tune.py --task freshness --category cu --epochs 3
```

Tuning chạy các baseline model/phương pháp khác nhau và tạo leaderboard CSV.
Ba epoch chỉ dùng sàng lọc nhanh; cấu hình thắng cần được train lại đủ epoch.

Khi bắt đầu train, chương trình dùng seed `42` để tách cố định 20% ảnh trong tập
train thành test nội bộ. Ảnh test không dùng augmentation và không tham gia cập
nhật trọng số, chọn checkpoint hoặc early stopping. Sau khi train xong, checkpoint
tốt nhất được nạp lại và chấm test đúng một lần.

Luồng dữ liệu thực tế:

```text
data/train -> 80% học + 20% test cuối
data/valid -> chọn model và early stopping
```

## 6. Resume an toàn

Mỗi run tạo các file:

```text
runs/<run>/last_checkpoint.pth  # trạng thái mới nhất, lưu sau mỗi epoch
runs/<run>/best_model.pth       # model tốt nhất theo validation loss
```

Resume bằng checkpoint mới nhất; `--epochs` là tổng số epoch mục tiêu:

```powershell
python train.py --epochs 15 --resume runs/<run>/last_checkpoint.pth router
```

Checkpoint lưu model, optimizer, scheduler, AMP scaler, epoch, metric tốt nhất và random state. Nếu process bị kill giữa epoch, mất tối đa epoch đang chạy; checkpoint cũ vẫn nguyên vẹn.

## 7. Thứ tự huấn luyện đề xuất

1. Phân loại toàn bộ dữ liệu theo năm nhóm.
2. Train router.
3. Train đủ năm freshness expert.
4. Kiểm tra từng model trên ảnh chưa từng xuất hiện trong train.
5. Chạy pipeline dự đoán hoàn chỉnh.

## 7. Dự đoán

Sau khi đã có router và expert tương ứng:

```powershell
python predict.py path\to\image.jpg
```

Kết quả gồm:

- Nhóm thực phẩm và độ tin cậy.
- Mức độ tươi và độ tin cậy.
- Chỉ số độ tươi ước tính từ `1–10`.

Chỉ số này được suy ra từ hình ảnh, không thay thế kiểm nghiệm an toàn thực phẩm.

## 8. Lỗi thường gặp

### `Dữ liệu chưa sẵn sàng`

Một hoặc nhiều thư mục lớp đang rỗng. Bổ sung ảnh vào đủ các thư mục trong
`data/train` và `data/valid`.

### Train hiển thị `Thiết bị: CPU`

PyTorch hiện tại không có CUDA. Kiểm tra:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### `CUDA out of memory`

Giảm batch size và đóng các ứng dụng đang sử dụng GPU:

```powershell
python train.py --batch-size 8 router
```

### Thiếu model khi dự đoán

Pipeline cần `models/router.pth` và expert của nhóm được router nhận diện. Hãy
train đủ sáu model trước khi chạy dự đoán tổng quát.
