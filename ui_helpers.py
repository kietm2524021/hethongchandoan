# -*- coding: utf-8 -*-
"""Các hàm không phụ thuộc Streamlit: gợi ý hoạt chất, trình bày cách dùng,
biên soạn phiếu rà soát thuốc DỰ THẢO (không phải đơn thuốc hợp lệ)."""
from __future__ import annotations

from datetime import date
from html import escape
import re
import unicodedata


def fold_vi(value: object) -> str:
    """So khớp tên hoạt chất bất kể dấu và chữ hoa, kể cả đ/Đ."""
    s = unicodedata.normalize('NFD',str(value or '').casefold()).replace('đ','d')
    return ''.join(c for c in s if unicodedata.category(c)!='Mn').strip()


def ingredient_suggestions(term: str, names: list[str], limit: int=12) -> list[str]:
    """Khớp ở bất kỳ vị trí nào, ưu tiên tên bắt đầu bằng từ tìm kiếm."""
    q=fold_vi(term)
    if not q:
        return []
    hits=[n for n in names if q in fold_vi(n)]
    return sorted(hits,key=lambda n:(0 if fold_vi(n).startswith(q) else 1,fold_vi(n),n))[:limit]


def historical_ways(item: dict, limit: int=3) -> list[str]:
    """Chỉ đọc giá trị cachuong từ HIS, KHÔNG suy ra liều hay số lần dùng."""
    seen=set();out=[]
    for v in item.get('cach_dung_lich_su',[]):
        way=str(v.get('cachuong','')).strip()
        if way and fold_vi(way) not in seen:
            out.append(way);seen.add(fold_vi(way))
        if len(out)>=limit:break
    return out


def sanitize_print_field(value: object, max_chars: int=300) -> str:
    """Văn bản an toàn cho phiếu HTML."""
    value=' '.join(str(value or '').split())
    return escape(value[:max_chars],quote=True)


def make_draft_html(*, patient_id: str, encounter_date: date, visit_type: str,
                    icd: str, diagnosis: str, clinician: str,
                    medications: list[dict], notes: str='') -> str:
    """Chỉ tạo bản thảo sau nhập CHỦ ĐỘNG của cán bộ chuyên môn.

    Mỗi phần tử medications cần: name, instructions, quantity. Không sao chép
    hướng dẫn lịch sử vào instructions và không tự điền liều lượng.
    """
    if not clinician.strip():
        raise ValueError('Thiếu người lập phiếu.')
    if not medications:
        raise ValueError('Chưa chọn hoạt chất.')
    if len(medications)>30:
        raise ValueError('Vượt giới hạn 30 hoạt chất cho phiếu nháp.')
    rows=[]
    for i,med in enumerate(medications,1):
        name=str(med.get('name','')).strip()
        instructions=str(med.get('instructions','')).strip()
        quantity=str(med.get('quantity','')).strip()
        if not name or not instructions or not quantity:
            raise ValueError('Cần nhập cách sử dụng và số lượng cho từng hoạt chất.')
        rows.append('<tr><td>'+str(i)+'</td><td>'+sanitize_print_field(name,180)+
                    '</td><td>'+sanitize_print_field(instructions,300)+
                    '</td><td>'+sanitize_print_field(quantity,80)+'</td></tr>')
    safe_date=encounter_date.isoformat()
    return f'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Phiếu rà soát hoạt chất </title>
<style>
@page{{size:A4;margin:17mm}}
body{{font:14px/1.5 Arial,sans-serif;color:#182436;max-width:850px;margin:32px auto}}
header{{text-align:center;border-bottom:2px solid #344b70;margin-bottom:22px}}
h1{{font-size:21px;margin:8px 0}}
.notice{{background:#fff3f3;border:1px solid #ca3434;padding:10px;font-weight:700;color:#9d2020}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}
th,td{{border:1px solid #8090a6;padding:8px;vertical-align:top;text-align:left}}
th{{background:#eef2f6}}
.meta p{{margin:5px 0}}.foot{{margin-top:30px;display:flex;justify-content:flex-end}}
.sign{{text-align:center;min-width:270px;min-height:95px}}
.print{{background:#205bb1;color:white;border:0;padding:10px 16px;cursor:pointer}}
@media print{{.print{{display:none}}body{{margin:0}}}}
</style></head><body>
<button class="print" onclick="window.print()">In toa thuốc</button>
<header><h1>PHIẾU RÀ SOÁT HOẠT CHẤT </h1><p>Chưa ký, chưa có giá trị kê đơn hoặc cấp phát</p></header>
<div class="notice"></div>
<div class="meta">
<p><b>Mã bệnh nhân:</b> {sanitize_print_field(patient_id,80) or '(chưa nhập)'}</p>
<p><b>Ngày khám:</b> {safe_date} &nbsp; <b>Loại lượt khám (khai báo):</b> {sanitize_print_field(visit_type,80)}</p>
<p><b>Mã ICD tham khảo:</b> {sanitize_print_field(icd,30)} &nbsp; <b>Chẩn đoán tham khảo:</b> {sanitize_print_field(diagnosis,180)}</p>
</div>
<table><thead><tr><th>STT</th><th>Hoạt chất được lựa chọn</th><th>Hướng dẫn do cán bộ chuyên môn nhập</th><th>Số lượng</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p><b>Ghi chú:</b> {sanitize_print_field(notes,700) or '—'}</p>
<div class="foot"><div class="sign"><b>Người lập phiếu:</b> {sanitize_print_field(clinician,120)}<p>(Ký xác nhận theo quy trình chuyên môn)</p></div></div>
</body></html>'''
