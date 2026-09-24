# Dữ liệu và kiểm soát leakage

## 1. Cấu trúc

```text
data/
  train/
    cu/
      FRESH/
      SPOILED/
    gia_cam/
      FRESH/
      HALF-FRESH/
      SPOILED/
    gia_suc/
      FRESH/
      HALF-FRESH/
      SPOILED/
    qua/
      FRESH/
      SPOILED/
    rau/
      FRESH/
      SPOILED/
  valid/
    ... cùng category/freshness thực tế với train ...
```

`ImageFolder` suy ra label từ tên thư mục và bỏ qua class directory không có ảnh. Vì vậy không phải expert nào cũng có đủ ba output.

## 2. Số lượng hiện tại

Theo lần kiểm tra gần nhất:

| Category | Freshness có ảnh |
|---|---|
| `cu` | `FRESH`, `SPOILED` |
| `gia_cam` | `FRESH`, `HALF-FRESH`, `SPOILED` |
| `gia_suc` | `FRESH`, `HALF-FRESH`, `SPOILED` |
| `qua` | `FRESH`, `SPOILED` |
| `rau` | `FRESH`, `SPOILED` |

Số lượng train/valid và trạng thái lớp cần được kiểm tra lại trước mỗi lần train; không suy ra từ thư mục mẫu trong tài liệu.

## 3. Nguồn dữ liệu tham khảo

- https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset
- https://www.kaggle.com/datasets/namhoangtrinh/fresh-spoiled-meat
- https://www.kaggle.com/datasets/crowww/meat-quality-assessment-based-on-deep-learning
- https://www.kaggle.com/datasets/iwayanadisaputra26/beefyteamdataset
- https://universe.roboflow.com/reinsgundar/meat-freshness-xpmq5

Cần lưu nguồn, điều kiện chụp, product/object group và phép biến đổi của từng tập khi xây split sạch.

## 4. Vấn đề leakage đã xác nhận

SHA-256 đã phát hiện nhiều nhóm ảnh byte-identical xuất hiện đồng thời trong `data/train` và `data/valid`. Một số ảnh đổi tên hoặc nằm ở thư mục khác nhưng nội dung giống nhau. Ngoài ra, split theo file/product hiện tại không đảm bảo ảnh cùng nguồn hoặc cùng object chỉ xuất hiện trong một split.

Do đó:

- Không dùng metric hiện tại để tuyên bố model chống học vẹt.
- Không di chuyển/xóa ảnh tự động nếu chưa xác định bản gốc và nhãn đúng.
- Cần kiểm tra cả near-duplicate bằng perceptual hash hoặc embedding.

Công cụ kiểm tra byte-identical:

```powershell
python scripts\check_duplicates.py
```

## 5. Quy trình tạo split sạch

1. Lập metadata cho mỗi ảnh: source, product/object, session và label.
2. Gom ảnh cùng nguồn/object/augmentation vào một group.
3. Chia group, không chia từng file, vào train/validation/test độc lập.
4. Chạy SHA-256 trên mọi split.
5. Chạy perceptual hash hoặc embedding similarity để bắt near-duplicate.
6. Kiểm tra lại phân bố class và recall lớp `SPOILED`.
7. Chỉ sau đó mới train và đánh giá model.

`split_validation.py` hiện chia trong product folder; nó không chứng minh group leakage đã được loại bỏ.
