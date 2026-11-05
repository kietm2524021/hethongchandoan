# Hệ thống V2 – Triệu chứng, cá nhân hóa, luật, thống kê đơn thuốc lịch sử, tái khám

**Bản thử nghiệm nội bộ**, không được dùng để xác nhận chẩn đoán hoặc phát hành đơn thuốc thực tế. Bộ mã mới độc lập, không cần nhập bất kỳ mã Python V1 nào.

## 1. Cấu trúc đầu vào

Đọc đúng 12 file CSV, tháng 1–12, mặc định ở `E:\HeThongChanDoan\data`:

`normalized_top10_thang1.csv`, ..., `normalized_top10_thang12.csv` (chấp nhận tên dạng thang01).

Cột bắt buộc: `mabn`, `makh`, `maicd`, `motatrieuchung`.

Cột tùy chọn: `ngaysinh` (để tính tuổi), `gioitinh`, `ngayhd` (lấy làm **đại diện** ngày khám), `kqcdoan`, `tenhc`, `cachuong`, `nghenghiep`.

**Giới tính**: nếu `gioitinh` là chuỗi `Nam`/`Nữ`, chương trình đọc trực tiếp. Nếu là `0`/`1`/`2`, **không tự giả định** ý nghĩa các mã. Hãy xem `artifacts/audit.json` trường `gioitinh_values_top`, đối chiếu quy ước HIS rồi chạy `--sex-encoding 1nam0nu`, `0nam1nu` hoặc `1nam2nu` cho đúng. Mã số khi chưa xác nhận được tính là `Không rõ`.

**Nghề nghiệp**: 12 CSV đã mô tả chưa có trường này. Nếu HIS có bảng liên kết mã bệnh nhân, lưu tạm trên máy nội bộ `nghe_nghiep.csv` với đúng tiêu đề `mabn,nghenghiep` rồi thêm `--jobs-csv .\nghe_nghiep.csv`. Nếu không có bảng hợp pháp, chương trình mã hóa `Không rõ`, **không bịa dữ liệu nghề nghiệp**. Khi không có dữ liệu nghề nghiệp, đầu vào nghề nghiệp trên web cũng không có tác dụng đối với mô hình.

## 2. Cài thư viện / huấn luyện

```powershell
cd E:\HeThongChanDoan
.\.venv\Scripts\Activate.ps1

# Giải nén thư mục mới thành E:\HeThongChanDoan\HeThongChanDoan_V2
cd .\HeThongChanDoan_V2
python -m pip install -r requirements.txt

# Chỉ kiểm tra chất lượng và đếm Top 10, chưa huấn luyện:
python train.py audit --data-dir E:\HeThongChanDoan\data

# Huấn luyện từ đầu, 6 cấu hình truyền thống, luật triệu chứng và báo cáo:
python train.py train --data-dir E:\HeThongChanDoan\data

# Nếu có bảng nghề nghiệp hợp lệ:
python train.py train --data-dir E:\HeThongChanDoan\data --jobs-csv .\nghe_nghiep.csv
```

Lưu ý: huấn luyện cần 10 nhóm ICD hợp lệ và đủ bệnh nhân cho chia tập theo bệnh nhân. Nếu nhóm không đủ số lượng, chương trình sẽ báo lỗi cụ thể thay vì tự làm tròn điểm đánh giá. Các tệp CSV không được gửi lên máy chủ bên ngoài trong quá trình chạy.

## 3. Khởi chạy web local

```powershell
python -m streamlit run streamlit_app.py
```

Mở **http://127.0.0.1:8502**. Ứng dụng mặc định chỉ lắng nghe địa chỉ localhost. Không cần public hoặc mở port router.

Trên web: chọn thuật toán; nhập triệu chứng, giới tính, tuổi, nghề nghiệp; nhấn **Phân tích mã bệnh**; mở xem luật trùng khớp, thống kê hoạt chất/toa lịch sử; cuối cùng nhập mã bệnh nhân HIS ở phần kiểm tra tái khám (chỉ trên máy nội bộ).

## 4. Các bước thuật toán

1. Đọc 12 CSV, kiểm tra schema và **gom theo một lượt khám** bằng `(tháng, mabn, makh)`, không đếm nhiều dòng thuốc thành nhiều bệnh nhân.
2. Chuẩn hóa mã ICD thành nhóm 3 ký tự, xử lý văn bản triệu chứng tiếng Việt, chọn Top 10 nhóm theo **số lượt khám**, loại lượt xung đột ICD, tên bệnh/mã ICD có sẵn trong đầu vào và mô tả thiếu thông tin.
3. Tính tuổi từ `ngaysinh` và `ngayhd` (nếu hợp lệ); chuẩn hóa giới tính; mã hóa nghề nghiệp từ dữ liệu thực nếu có.
4. Chia train/validation/test theo **bệnh nhân** khoảng 64/16/20, không dùng cùng người ở các tập khác nhau.
5. Huấn luyện 3 thuật toán (Naive Bayes, Linear SVM, Logistic Regression) với hai bộ đặc trưng: (a) chỉ triệu chứng TF-IDF, (b) TF-IDF + nhóm tuổi + giới tính + nghề nghiệp. **Tổng cộng 6 cấu hình**.
6. Khai phá luật cụm triệu chứng 2–3 từ → ICD từ **train** theo support, confidence và lift. Có hai chế độ dự đoán: chỉ luật (có quyền không dự đoán) hoặc kết hợp luật với Logistic Regression cá nhân hóa.
7. Chọn mô hình tốt nhất theo **validation Macro F1**, sau đó báo cáo riêng test Accuracy, Macro F1, Balanced Accuracy, Cohen Kappa và ma trận nhầm lẫn. Không sử dụng test để chọn mô hình.
8. Đối với ICD được dự đoán, thống kê hoạt chất, mức xuất hiện trên các lượt khám có đơn, các **tổ hợp thuốc đồng xuất hiện trong những toa lịch sử**. Ngưỡng mặc định là >=5 lượt. Cách dùng chỉ là văn bản ghi nhận lịch sử, không xuất liều cá thể hóa.
9. Xác định tái khám: mã bệnh nhân có ngày khám trước với cùng **nhóm ICD**. Kiểm tra riêng có cùng hoạt chất được ghi nhận vào ít nhất hai ngày khám trước hay không. Hai điều kiện này chỉ là bằng chứng lịch sử và **không** đủ kết luận mắc bệnh mạn tính hoặc phải cấp thuốc mạn.

## 5. Kết quả được xuất

Trong `artifacts/`:
- `audit.json`, `tan_suat_icd.csv`, `phan_bo_10_benh.csv`, `chia_du_lieu.json`.
- `so_sanh_mo_hinh.csv`, `chon_mo_hinh.json`, `bao_cao_tung_benh_test.csv`, `ma_tran_nham_lan_test.csv`.
- `rules.json`, `drug_catalog.json`, `tai_kham_tong_hop.json`.
- `models/*.joblib` (6 mô hình độc lập); `metadata.json`.
- `patient_history_local.joblib` và `.hmac_key` (**nhạy cảm, chỉ lưu nội bộ**, tuyệt đối không đưa lên GitHub/public cloud).

## 6. Quy tắc an toàn và giới hạn

- Mã ICD 3 ký tự không phải chẩn đoán chi tiết; các bệnh như tăng huyết áp, đái tháo đường, HIV không thể xác nhận chỉ từ lời mô tả triệu chứng.
- **Không có cơ sở thống kê nào biến thuốc thường gặp thành thuốc "bắt buộc"**. Bản này đánh dấu thuốc có trong >=90% lượt có đơn, với tối thiểu 20 lượt, như một chỉ báo mô tả chứ không khuyến nghị kê đơn.
- Các tổ hợp thuốc là tần suất đồng xuất hiện của **toàn toa đã kê**, không phải đơn thuốc phát hành cho người nhập triệu chứng. Không tự kê liều, không xét chống chỉ định/tương tác/suy thận/suy gan/dị ứng.
- `ngayhd` có thể là ngày hóa đơn chứ không hoàn toàn là ngày thăm khám; cần đối chiếu nghiệp vụ HIS.
- Tái khám trong vòng 12 tháng quan sát là **không đầy đủ** nếu bệnh nhân khám trước khoảng quan sát; không tự nhận diện bệnh mạn tính hoặc cấp thuốc định kỳ.
- Các bản ghi triệu chứng và nội dung dùng thuốc có thể chứa thông tin nhận diện. Cần rà soát bảo mật/đồng ý xử lý dữ liệu, và **không public thư mục `artifacts/`**.
- Các mã mô hình trong tài liệu này không thay thế thẩm định lâm sàng hoặc đánh giá độ an toàn đơn thuốc. Cần đánh giá độ nhạy của từng bệnh và thử nghiệm có giám sát trước khi sử dụng trong môi trường thật.

## 7. Kiểm thử với CSV giả lập

```powershell
python -m unittest discover -s tests -v
```

Kiểm thử sử dụng file CSV sinh trong thư mục tạm, không dùng hồ sơ người bệnh thực tế.
