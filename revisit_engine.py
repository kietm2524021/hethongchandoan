# -*- coding: utf-8 -*-
"""Tái khám và lặp hoạt chất trong lịch sử theo ngày.
- Cùng nhóm ICD ở ngày trước => có bằng chứng tái khám cùng nhóm bệnh.
- Hoạt chất lặp tại >=2 ngày => có bằng chứng lặp cấp phát, KHÔNG đủ kết luận bệnh mạn.
Không xuất mã bệnh nhân hoặc tên hoạt chất cấp bệnh nhân; dữ liệu vẫn nhạy cảm, chỉ nội bộ.
"""
from __future__ import annotations
import hashlib
import hmac
import secrets
from collections import defaultdict
from pathlib import Path


def get_or_create_key(path: Path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(secrets.token_bytes(32))
        try:path.chmod(0o600)
        except OSError:pass
    key=path.read_bytes()
    if len(key)<32:raise RuntimeError('Khóa HMAC không hợp lệ')
    return key


def token(patient, key):
    return hmac.new(key,patient.encode('utf-8'),hashlib.sha256).hexdigest()


def _med_token(name,key):
    return hmac.new(key,('med:'+name.casefold()).encode('utf-8'),hashlib.sha256).hexdigest()


def make_history(visits,key,drug_rows=None):
    med_by_visit=defaultdict(set)
    if drug_rows is not None:
        from data_core import norm
        for visit,med in drug_rows[['visit_key','tenhc']].itertuples(index=False,name=None):
            med=norm(med)
            if med and len(med)<=120:
                med_by_visit[visit].add(_med_token(med,key))
    history=defaultdict(dict)
    n_with_date=0
    valid=visits.loc[visits['valid_icd'] & visits['visit_date'].notna(),
                     ['visit_key','patient','icd','visit_date']]
    for visit,patient,icd,when in valid.itertuples(index=False,name=None):
        if not patient: continue
        day=when.date().isoformat()
        tag=token(patient,key)
        row=history[tag].setdefault((day,icd),set())
        row.update(med_by_visit.get(visit,()))
        n_with_date+=1
    result={tag:[{'ngay':day,'ma_icd':icd,'ma_hoat_chat':sorted(meds)}
                 for (day,icd),meds in sorted(events.items())]
            for tag,events in history.items()}
    return result,n_with_date


def _has_repeated_medicine(events):
    by_med=defaultdict(set)
    for row in events:
        for med in row.get('ma_hoat_chat',[]):
            by_med[med].add(row['ngay'])
    return any(len(days)>=2 for days in by_med.values())


def revisit_flags(history):
    total_followup=0; same_disease=0; repeat_med_patients=0
    for events in history.values():
        seen_days=set();seen_codes=defaultdict(set)
        for e in sorted(events,key=lambda x:x['ngay']):
            day=e['ngay'];code=e['ma_icd']
            if any(old<day for old in seen_days):total_followup+=1
            if any(old<day for old in seen_codes[code]):same_disease+=1
            seen_days.add(day);seen_codes[code].add(day)
        if _has_repeated_medicine(events):repeat_med_patients+=1
    return {'luot_tai_kham_cung_ma_icd':same_disease,
            'luot_co_kham_truoc_do':total_followup,
            'so_benh_nhan_lap_hoat_chat_lich_su':repeat_med_patients,
            'dinh_nghia':'Cùng mã ICD ở ngày trước là tái khám theo mã; cùng hoạt chất ở 2 ngày là lặp cấp phát, KHÔNG tự xác nhận bệnh mạn tính.'}


def patient_status(patient_id,target_icd,today_iso,history,key):
    if not patient_id:
        return {'status':'unknown','text':'Chưa nhập mã bệnh nhân nên không thể kiểm tra tái khám.'}
    match=history.get(token(patient_id,key),[])
    if not match:
        return {'status':'not_observed','text':'Không tìm thấy lượt khám trước trong 12 tháng dữ liệu; không có nghĩa bệnh nhân chưa từng khám.'}
    before=[e for e in match if e['ngay']<today_iso]
    if not before:
        return {'status':'not_observed','text':'Không có lượt khám nào trước ngày đang xét trong khoảng dữ liệu.'}
    same=[e for e in before if e['ma_icd']==target_icd]
    if same:
        repeat_drugs=_has_repeated_medicine(same)
        text=(f'Ghi nhận {len({e["ngay"] for e in same})} ngày khám trước cùng nhóm ICD {target_icd}.'
              + (' Đồng thời có hoạt chất được ghi nhận lặp ở ít nhất 2 ngày khám trước.'
                 if repeat_drugs else ' Chưa đủ bằng chứng về hoạt chất được cấp lặp trong các lần khám trước.')
              + ' Không thể kết luận bệnh mạn tính hoặc chỉ định cấp thuốc từ kết quả này.')
        return {'status':'followup_same_icd','previous_same_icd':len(same),
                'last_date':max(e['ngay'] for e in same),
                'repeat_medication_observed':repeat_drugs,'text':text}
    return {'status':'previous_other_icd','n_previous':len(before),
            'text':'Có khám trước, nhưng chưa ghi nhận cùng nhóm ICD trước ngày đang xét.'}
