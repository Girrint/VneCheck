# Báo cáo hiện trạng triển khai

Ngày cập nhật: 2026-10-18.

## Đã hoàn thành

### Dataset

- Gộp `data/train` và `data/valid`: 116.818 ảnh.
- Tính và cache SHA-256 cùng pHash.
- Loại 673 file xung đột nhãn exact SHA-256.
- Loại 21.696 exact duplicate dư thừa cùng nhãn.
- Chia 94.449 ảnh thành 66.115 train, 14.172 valid và 14.162 test.
- Giữ nguyên pHash cluster giữa các split.
- Ép 4 cluster lớn vào train; không còn cluster trên 500 ảnh ở valid/test.
- Sinh audit CSV trong `reports/`.

### Pipeline train

- `DATA_DIR` trỏ đến `data_split`.
- Loader đọc train, valid và test độc lập.
- Kiểm tra class list giữa ba split.
- Validation chọn checkpoint; test chỉ chạy sau khi nạp checkpoint tốt nhất.
- Run config ghi rõ đường dẫn và số lượng từng split.
- Menu dataset status hiển thị đủ ba split.
- Preprocessing tự sửa EXIF orientation và chuyển RGB.
- Bỏ rotation có thể tạo góc đen; dùng crop và ColorJitter nhẹ, không đổi hue.
- Dùng chung hằng số kích thước/normalize cho loader, inference, ONNX và manifest.

### Đánh giá và inference

- `predict.py` đọc class list từ từng checkpoint.
- Evaluator router mặc định dùng `data_split/test`.
- Evaluator end-to-end hỗ trợ đồng thời expert 2 lớp và 3 lớp.
- Loại bỏ ONNX inference legacy từng hard-code sai class order.
- Giữ export, verify và package manifest cho ONNX.

### Dọn dự án

- Chỉ giữ `scripts/resplit_dataset.py` làm công cụ dữ liệu chính.
- Loại các script cũ dùng để move/delete valid, kiểm tra duplicate riêng lẻ và
  split pHash phiên bản cũ.
- Loại wrapper inference trùng lặp.
- Regression tests chuyển sang kiểm tra logic dedupe, conflict quarantine,
  tính toàn vẹn cluster và ba loader độc lập.
- Requirements bỏ thư viện không còn được import trực tiếp.

## Việc tiếp theo

1. Train lại router trên `data_split`.
2. Train lại đủ 5 freshness expert.
3. Chọn model bằng validation; không dùng test để tuning.
4. Chạy router uncertainty và end-to-end evaluation trên test đúng một lần cho
   cấu hình đã chốt.
5. Export ONNX, verify sai khác và tạo manifest mới.
6. Nếu có metadata source/object/session, tạo split group-aware mạnh hơn pHash.

## Lưu ý về artifact hiện tại

Các file `.pth` và `.onnx` hiện có được tạo trước dataset split mới. Chúng vẫn
có thể chạy về mặt kỹ thuật nhưng không đại diện cho kết quả trên test sạch.
Không dùng metric cũ làm số liệu phát hành sau thay đổi này.
