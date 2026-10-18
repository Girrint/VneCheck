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

## Báo cáo audit

- `reports/resplit_excluded_files.csv`: file bị loại và lý do.
- `reports/resplit_cluster_summary.csv`: thống kê từng cluster.
- `reports/resplit_cluster_members.csv`: thành viên cluster cần kiểm tra.
- `reports/resplit_phash_cache.pkl`: cache pHash và SHA-256.

Script không xóa hay di chuyển dữ liệu nguồn. Khi output đã tồn tại và không
rỗng, script từ chối ghi đè; hãy dùng một output mới hoặc lưu trữ split cũ trước.
