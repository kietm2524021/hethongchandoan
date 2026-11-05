HeThongChanDoan_V2/   (bộ mã HOÀN TOÀN MỚI, không dùng lại script V1)
  train.py             huấn luyện, đánh giá, audit 12 CSV
  data_core.py         đọc CSV; gom lượt; tính tuổi; chuẩn hóa ICD
  rules_engine.py      luật triệu chứng -> ICD (support, confidence, lift)
  drugs_engine.py      hoạt chất, tổ hợp đơn lịch sử, tần suất thường gặp
  revisit_engine.py    tái khám, dấu hiệu tái cấp cùng hoạt chất
  streamlit_app.py     giao diện local: chọn thuật toán, nhập tuổi/giới/nghề, xem thuốc, kiểm tái khám
  requirements.txt
  CHAY_NHANH.ps1       chạy toàn quy trình trên Windows PowerShell
  README.md           hướng dẫn chi tiết
  .streamlit/config.toml
  .gitignore           chặn thông tin nhạy cảm khỏi git
  tests/test_synthetic.py
  artifacts/           tự tạo sau huấn luyện; chứa dữ liệu suy ra từ HIS. TUYỆT ĐỐI KHÔNG PUBLIC.

Lưu cả thư mục này ở: E:\HeThongChanDoan\HeThongChanDoan_V2
12 CSV vẫn đặt ở:       E:\HeThongChanDoan\data

PowerShell:
  cd E:\HeThongChanDoan\HeThongChanDoan_V2
  ..\.venv\Scripts\Activate.ps1
  python -m pip install -r requirements.txt
  python train.py audit --data-dir ..\data
  python train.py train --data-dir ..\data
  python -m streamlit run streamlit_app.py
  # http://127.0.0.1:8502

Không có dữ liệu nghề nghiệp trong 12 CSV: trường này sẽ bị bỏ qua trong phân lớp.
Có bảng nghề nghiệp hợp pháp? thêm --jobs-csv .\nghe_nghiep.csv

Không sử dụng mô hình hoặc tổ hợp toa thống kê để chẩn đoán xác định/kê đơn thật.
