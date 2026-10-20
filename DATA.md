# Dữ liệu

## Dataset nguồn và dataset huấn luyện

`data/` là dữ liệu nguồn cũ. Pipeline train không đọc trực tiếp thư mục này.

`data_split/` là dataset sạch đang được dùng:

```text
data_split/
  train/<category>/<freshness>/<product>/...
  valid/<category>/<freshness>/<product>/...
  test/<category>/<freshness>/<product>/...
```

Ở tầng router, `category` là class. Ở tầng expert, `freshness` là class;
`product` chỉ là thư mục tổ chức và không trở thành output của model.

## Thống kê hiện tại

| Split | Số ảnh | Tỷ lệ thực tế |
|---|---:|---:|
| Train | 66.115 | 69,999% |
| Validation | 14.172 | 15,005% |
| Test | 14.162 | 14,995% |
| Tổng | 94.449 | 100% |

Số ảnh theo router category:

| Category | Train | Valid | Test | Freshness |
|---|---:|---:|---:|---|
| `cu` | 9.472 | 2.030 | 2.029 | FRESH, SPOILED |
| `gia_cam` | 6.368 | 1.365 | 1.363 | FRESH, HALF-FRESH, SPOILED |
| `gia_suc` | 18.792 | 4.027 | 4.027 | FRESH, HALF-FRESH, SPOILED |
| `qua` | 17.646 | 3.784 | 3.780 | FRESH, SPOILED |
| `rau` | 13.837 | 2.966 | 2.963 | FRESH, SPOILED |

## Cách split hiện tại được tạo

Lần tạo gần nhất dùng:

```powershell
python scripts\resplit_dataset.py --threshold 0 --dry-run
python scripts\resplit_dataset.py --threshold 0
```

Đầu vào gồm `data/train` và `data/valid`, tổng cộng 116.818 ảnh. Script:

1. Tính pHash và SHA-256, có cache trong `reports/`.
2. Loại 673 file thuộc các SHA-256 xung đột nhãn.
3. Loại 21.696 exact duplicate dư thừa cùng nhãn.
4. Giữ lại 94.449 ảnh.
5. Gom pHash trong cùng leaf class.
6. Chia nguyên cluster theo tỷ lệ mục tiêu 70/15/15.
7. Ép cluster quá lớn vào train để không làm validation/test lệch mạnh.

Kết quả cuối có 79.973 cluster, không có cluster nhiều class và không còn
cluster trên 500 ảnh trong validation/test.

`threshold=0` chỉ nối ảnh có pHash giống hệt. Nó giảm leakage do ảnh giống nhau
nhưng không thay thế metadata source/object/session. Nếu bổ sung được metadata,
nên group theo đối tượng hoặc phiên chụp trước khi chia.

## Dữ liệu negative cho unified model

Ảnh vừa thu thập đặt vào `_staging`, chưa đưa thẳng vào split:

```text
data_negative/
  _staging/
    UNSUPPORTED/
    NON_FOOD/
  _quarantine/
  metadata/
  train/{UNSUPPORTED,NON_FOOD}/
  valid/{UNSUPPORTED,NON_FOOD}/
  test/{UNSUPPORTED,NON_FOOD}/
```

`_quarantine` chứa ảnh lỗi, xung đột hoặc cần duyệt lại. `metadata` lưu source
ID, URL và license. `build_unified_manifest.py --negative-root data_negative`
chỉ đọc ba thư mục `train`, `valid`, `test`; nó bỏ qua staging và quarantine.
Toàn bộ `data_negative/` được ignore khỏi Git.

### Parquet hiện có

`data_negative/_parquet` hiện chứa một phần Food-101 dưới dạng Hugging Face
Parquet. Schema là `image.bytes`, `image.path`, `label`; metadata ánh xạ 101
label sang tên món ăn. Bộ này chỉ phù hợp làm `UNSUPPORTED`, không có lớp
động vật/phong cảnh/cây cối để tạo `NON_FOOD`.

Validation đã đủ 3/3 shard (25.250 record). Train mới có shard `00000` và
`00002` trên tổng 8 shard (18.938 record), còn thiếu `00001` và `00003–00007`.
Pipeline mặc định lấy 5.000 ảnh cân bằng từ validation hoàn chỉnh, vì vậy không
phụ thuộc vào các train shard còn thiếu.

Chuẩn bị dữ liệu negative:

```powershell
python scripts\extract_food101_parquet.py --dry-run
python scripts\extract_food101_parquet.py
python scripts\split_negative_dataset.py --threshold 0 --dry-run
python scripts\split_negative_dataset.py --threshold 0
```

Script đầu ghi nguyên bytes ảnh từ Parquet vào
`_staging/UNSUPPORTED/food101/<dish>/` và tạo
`metadata/food101_selected.csv`. Script thứ hai:

1. Đối chiếu SHA-256 của negative với toàn bộ `data_split`.
2. Loại file trùng với `SUPPORTED`, file trùng nhưng khác nhãn negative và exact
   duplicate dư thừa.
3. Gom pHash trong riêng từng nhãn negative.
4. Chia nguyên cluster theo 70/15/15 vào `data_negative/{train,valid,test}`.

Các file bị loại được ghi vào `reports/negative_excluded_files.csv`; audit
cluster nằm ở `reports/negative_cluster_summary.csv` và
`reports/negative_cluster_members.csv`. `--dry-run` không copy ảnh vào split,
nhưng vẫn cập nhật cache và các báo cáo audit.

### Kết quả negative hiện tại

Đã trích 5.000 ảnh Food-101 cho `UNSUPPORTED` và nhận 5.000 ảnh
`NON_FOOD`. Audit tại `threshold=0` không tìm thấy file lỗi, exact duplicate
hoặc xung đột SHA-256 với `data_split`. Cả 10.000 ảnh có pHash riêng biệt.

| Split | UNSUPPORTED | NON_FOOD | Tổng negative |
|---|---:|---:|---:|
| Train | 3.500 | 3.500 | 7.000 |
| Validation | 750 | 750 | 1.500 |
| Test | 750 | 750 | 1.500 |
| Tổng | 5.000 | 5.000 | 10.000 |

Manifest unified đã được tạo lại bằng:

```powershell
python scripts\build_unified_manifest.py --negative-root data_negative
```

Manifest có 104.449 dòng: 94.449 `SUPPORTED` và 10.000 negative. File này
được sinh tại `reports/unified_manifest.csv` và không commit vào Git.

## Báo cáo audit

- `reports/resplit_excluded_files.csv`: file bị loại và lý do.
- `reports/resplit_cluster_summary.csv`: thống kê từng cluster.
- `reports/resplit_cluster_members.csv`: thành viên cluster cần kiểm tra.
- `reports/resplit_phash_cache.pkl`: cache pHash và SHA-256.

Script không xóa hay di chuyển dữ liệu nguồn. Khi output đã tồn tại và không
rỗng, script từ chối ghi đè; hãy dùng một output mới hoặc lưu trữ split cũ trước.
