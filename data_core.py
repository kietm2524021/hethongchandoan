# -*- coding: utf-8 -*-
"""Nạp 12 CSV HIS và xây dựng đơn vị phân tích là một lượt khám.
Không ghi thông tin nhận diện bệnh nhân ra các báo cáo thông thường.
"""
from __future__ import annotations
import re
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

EMPTY = {'', 'na', 'nan', 'none', 'null', 'n/a', '--', 'không rõ', 'khong ro'}
REQ = ('mabn', 'makh', 'maicd', 'motatrieuchung')
OPT = ('ngaysinh', 'gioitinh', 'ngayhd', 'nghenghiep', 'kqcdoan', 'tenhc', 'cachuong')
ICD_RE = re.compile(r'^([A-Z][0-9]{2})(?:\.?([A-Z0-9]{1,4}))?$')
ICD_IN_TEXT = re.compile(r'(?<!\w)[a-z]\d{2}(?:\.[a-z0-9]{1,4})?(?!\w)', re.I)
DIAGNOSIS_TERMS = ['đái tháo đường', 'tiểu đường', 'tăng huyết áp', 'cao huyết áp',
                   'viêm kết mạc', 'viêm phế quản', 'viêm họng', 'viêm xoang',
                   'viêm phổi', 'trào ngược dạ dày', 'hiv', 'aids',
                   'viêm khớp dạng thấp']
NON_SYMPTOM = {'khám', 'khám bệnh', 'tái khám', 'tái khám định kỳ', 'khám lại',
               'theo hẹn', 'không triệu chứng', 'không có triệu chứng'}


def norm(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    x = unicodedata.normalize('NFC', str(v))
    for c in ('\ufeff', '\u200b', '\u200c', '\u200d', '\xa0'):
        x = x.replace(c, ' ')
    x = ' '.join(x.split())
    return '' if x.casefold() in EMPTY else x


def clean_text(v):
    s = norm(v).casefold()
    s = re.sub(r'[^\w\s,.;:/%+\-()]', ' ', s, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', s).strip(' ,;.:/')


def icd_code(v):
    s = norm(v).upper().replace(' ', '')
    m = ICD_RE.fullmatch(s)
    return m.group(1) if m else ''  # Nhóm ICD 3 ký tự; KHÔNG phải mã chi tiết.


SEX_ENCODINGS = {
    'text': {},
    '1nam0nu': {'1':'Nam','0':'Nữ'},
    '0nam1nu': {'0':'Nam','1':'Nữ'},
    '1nam2nu': {'1':'Nam','2':'Nữ'},
}


def sex_code(v, encoding='text'):
    x = clean_text(v)
    if x in {'nam', 'male', 'm', 'trai'}: return 'Nam'
    if x in {'nữ', 'nu', 'female', 'f', 'gái'}: return 'Nữ'
    if x in {'khác', 'khac', 'other'}: return 'Khác'
    if encoding not in SEX_ENCODINGS:
        raise ValueError(f'Không hỗ trợ mã hóa giới tính: {encoding}')
    # 0/1/2 tùy quy ước HIS; không đoán nếu chưa được chủ dữ liệu xác nhận.
    return SEX_ENCODINGS[encoding].get(x,'Không rõ')


def age_band(age):
    if age is None or (isinstance(age, float) and not np.isfinite(age)):
        return 'Không rõ'
    try: a = int(float(age))
    except (ValueError, TypeError): return 'Không rõ'
    if not 0 <= a <= 110: return 'Không rõ'
    for cutoff, name in [(5,'00-04'), (15,'05-14'), (25,'15-24'), (45,'25-44'),
                         (65,'45-64'), (80,'65-79')]:
        if a < cutoff: return name
    return '80+'


def job_code(v):
    x = norm(v)
    if not x: return 'Không rõ'
    if len(x) > 80: return 'Không rõ'
    # Không cho văn bản nghề nghiệp tự do dài chứa thông tin cá nhân.
    if any(z in x for z in ('@', 'http://', 'https://')) or re.search(r'\d{5,}', x):
        return 'Không rõ'
    return x.casefold()


def month_path(folder: Path, month: int):
    paths = (folder / f'normalized_top10_thang{month}.csv',
             folder / f'normalized_top10_thang{month:02d}.csv')
    exists = list(dict.fromkeys(p for p in paths if p.is_file()))
    if len(exists) != 1:
        raise FileNotFoundError(f'Tháng {month}: phải có đúng 1 file normalized_top10_thang{month}[hoặc 0{month}].csv ở {folder}')
    return exists[0]


def parse_dates(series):
    x = series.map(norm).replace('', np.nan)
    # dayfirst=True: HIS thường dùng dd/mm/yyyy. Các giá trị không rõ được coi là thiếu.
    try:
        dt = pd.to_datetime(x, errors='coerce', dayfirst=True, format='mixed')
    except (TypeError, ValueError):
        dt = pd.to_datetime(x, errors='coerce', dayfirst=True)
    return dt


def _choose_text(values):
    xs = list(dict.fromkeys(x for x in values if x))
    return max(xs, key=len) if xs else ''


def _choose_mode(values, fallback=''):
    xs = [x for x in values if x]
    return Counter(xs).most_common(1)[0][0] if xs else fallback


def read_data(folder: Path, jobs_csv: Path | None = None, sex_encoding='text'):
    folder = Path(folder)
    frames = []
    monthly = []
    for m in range(1, 13):
        path = month_path(folder, m)
        columns = list(pd.read_csv(path, encoding='utf-8-sig', nrows=0).columns)
        aliases = {norm(c).casefold(): c for c in columns}
        missing = set(REQ) - set(aliases)
        if missing:
            raise ValueError(f'{path.name}: thiếu các cột bắt buộc {sorted(missing)}')
        # ĐỌC LẠI THEO TÊN CỘT: read_csv(usecols) không đảm bảo thứ tự đầu vào.
        cols = [k for k in REQ + OPT if k in aliases]
        piece = pd.read_csv(path, encoding='utf-8-sig', dtype=str,
                            keep_default_na=False, usecols=[aliases[k] for k in cols],
                            on_bad_lines='error', low_memory=False)
        piece = piece.rename(columns={aliases[k]: k for k in cols})
        piece = piece.reindex(columns=list(REQ + OPT), fill_value='')
        piece['month_file'] = m
        monthly.append({'thang':m, 'ten_file':path.name, 'so_dong':len(piece),
                        'co_nghenghiep':bool('nghenghiep' in aliases),
                        'co_ngayhd':bool('ngayhd' in aliases)})
        frames.append(piece)
        print(f'Đã đọc tháng {m:02}: {len(piece):,} dòng')
    raw = pd.concat(frames, ignore_index=True)
    raw['mabn'] = raw['mabn'].map(norm)
    raw['makh'] = raw['makh'].map(norm)
    raw['icd'] = raw['maicd'].map(icd_code)
    raw['symptoms'] = raw['motatrieuchung'].map(clean_text)
    raw['sex'] = raw['gioitinh'].map(lambda x: sex_code(x,sex_encoding))
    raw['occupation'] = raw['nghenghiep'].map(job_code)
    raw['visit_key'] = list(zip(raw['month_file'], raw['mabn'], raw['makh']))
    raw['has_keys'] = raw['mabn'].ne('') & raw['makh'].ne('')
    n_missing_keys = int((~raw['has_keys']).sum())
    work = raw.loc[raw['has_keys']].copy()

    if jobs_csv:
        ref = pd.read_csv(jobs_csv, encoding='utf-8-sig', dtype=str, keep_default_na=False)
        cmap = {norm(c).casefold(): c for c in ref.columns}
        if not {'mabn','nghenghiep'} <= set(cmap):
            raise ValueError('File nghề nghiệp phải có cột mabn và nghenghiep')
        ref = ref[[cmap['mabn'], cmap['nghenghiep']]].copy()
        ref.columns = ['mabn','job_external']
        ref['mabn'] = ref['mabn'].map(norm)
        ref['job_external'] = ref['job_external'].map(job_code)
        ref = ref.drop_duplicates('mabn', keep='last')
        work = work.merge(ref, on='mabn', how='left', validate='many_to_one')
        work['occupation'] = np.where(work['occupation'].eq('Không rõ'),
                                      work['job_external'].fillna('Không rõ'),
                                      work['occupation'])
        work = work.drop(columns='job_external')

    groups = work.groupby('visit_key', sort=False)
    visits = groups.agg(patient=('mabn', 'first'), month=('month_file', 'first'),
                        n_rows=('icd', 'size'),
                        icd_values=('icd', lambda x: sorted(set(x))),
                        symptoms=('symptoms', _choose_text),
                        sex=('sex', lambda x: _choose_mode((z for z in x if z != 'Không rõ'), 'Không rõ')),
                        occupation=('occupation', lambda x: _choose_mode((z for z in x if z != 'Không rõ'), 'Không rõ')),
                        dob_raw=('ngaysinh', lambda x: _choose_mode((norm(z) for z in x))),
                        date_raw=('ngayhd', lambda x: _choose_mode((norm(z) for z in x))),
                        disease_name=('kqcdoan', lambda x: _choose_mode((norm(z) for z in x)))).reset_index()
    visits['valid_icd'] = visits['icd_values'].map(lambda a: len(a)==1 and bool(a[0]))
    visits['icd'] = visits['icd_values'].map(lambda a: a[0] if len(a)==1 else '')
    visit_dt = parse_dates(visits['date_raw'])
    dob_dt = parse_dates(visits['dob_raw'])
    age = np.floor((visit_dt - dob_dt).dt.days / 365.2425)
    visits['age'] = age.where(age.between(0,110), np.nan)
    visits['age_band'] = visits['age'].map(age_band)
    visits['visit_date'] = visit_dt
    visits['has_leaked_icd'] = visits['symptoms'].map(lambda s: bool(ICD_IN_TEXT.search(s)))
    names = visits.loc[visits['valid_icd'] & visits['disease_name'].ne(''), ['icd','disease_name']]
    counts = (names.groupby(['icd','disease_name']).size().reset_index(name='count')
              .sort_values(['icd','count'], ascending=[True,False]).drop_duplicates('icd'))
    names_map = dict(zip(counts['icd'], counts['disease_name']))
    report = {'n_raw_rows':len(raw), 'n_rows_missing_patient_or_visit':n_missing_keys,
              'sex_encoding':sex_encoding,
              'gioitinh_values_top':raw['gioitinh'].map(norm).value_counts().head(12).to_dict(),
              'n_visits':len(visits), 'n_invalid_or_ambiguous_icd':int((~visits['valid_icd']).sum()),
              'n_visits_without_date':int(visits['visit_date'].isna().sum()),
              'n_visits_without_age':int(visits['age'].isna().sum()),
              'n_visits_without_sex':int(visits['sex'].eq('Không rõ').sum()),
              'n_visits_without_occupation':int(visits['occupation'].eq('Không rõ').sum()),
              'monthly':monthly}
    return work, visits, names_map, report


def select_training_visits(visits: pd.DataFrame, names: dict, n=10):
    valid = visits.loc[visits['valid_icd']].copy()
    valid = valid.loc[~valid['icd'].str[0].isin(list('RVWXYZ'))]
    freq = valid.groupby('icd').agg(n_visits=('patient','size'),
                                    n_patients=('patient','nunique')).reset_index()
    freq = freq.sort_values(['n_visits','icd'], ascending=[False,True]).reset_index(drop=True)
    freq['disease_name'] = freq['icd'].map(names).fillna('')
    if len(freq) < n:
        raise ValueError(f'Chỉ có {len(freq)} nhóm ICD hợp lệ; cần ít nhất {n} nhóm')
    codes = freq.head(n)['icd'].tolist()
    selected = valid.loc[valid['icd'].isin(codes)].copy()
    # Dữ liệu đầu vào chẩn đoán không được chứa nhãn bệnh hoặc tên chẩn đoán.
    terms = set(DIAGNOSIS_TERMS)
    for name in names.values():
        word = clean_text(name)
        if len(word)>=12 and len(word.split())>=3:
            terms.add(word)
    patt = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(t) for t in sorted(terms,key=len,reverse=True)) + r')(?!\w)', re.I)
    selected['leak_disease'] = selected['symptoms'].map(lambda s: bool(patt.search(s)))
    selected['valid_text'] = selected['symptoms'].map(lambda s: len(s)>=5 and s not in NON_SYMPTOM)
    usable = selected.loc[selected['valid_text'] & ~selected['has_leaked_icd'] & ~selected['leak_disease']].copy()
    usable = usable.sort_values(['month','patient'],kind='stable').reset_index(drop=True)
    if len(set(usable['icd'])) != n:
        raise ValueError('Sau khi lọc mô tả triệu chứng, không còn đủ 10 nhãn. Xem báo cáo Top10.')
    return usable, freq, codes, sorted(terms)
