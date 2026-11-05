# Chạy trong PowerShell (máy nội bộ) từ thư mục HeThongChanDoan_V2.
# Mặc định .venv ở thư mục cha; 12 CSV ở ..\data.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
$data = Join-Path $PSScriptRoot '..\data'
if (-not (Test-Path $python)) { throw "Không tìm thấy Python venv: $python" }
if (-not (Test-Path $data)) { throw "Không tìm thấy thư mục data: $data" }
Write-Host "[1] Cài thư viện ..."
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Cài thư viện thất bại.' }
Write-Host "[2] Kiểm tra 12 CSV ..."
& $python train.py audit --data-dir $data
if ($LASTEXITCODE -ne 0) { throw 'Kiểm tra CSV không đạt. Đọc thông báo trước khi chạy tiếp.' }
Write-Host "[3] Huấn luyện và tạo luật, thống kê thuốc, tái khám ..."
& $python train.py train --data-dir $data
if ($LASTEXITCODE -ne 0) { throw 'Huấn luyện thất bại. Đọc log trên màn hình.' }
Write-Host "[4] Mở web local tại http://127.0.0.1:8502"
& $python -m streamlit run streamlit_app.py
