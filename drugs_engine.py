# -*- coding: utf-8 -*-
"""Thống kê toa thuốc trong dữ liệu HIS theo ICD, KHÔNG sinh y lệnh mới.
'Rất phổ biến' không đồng nghĩa 'bắt buộc', và không có suy luận liều.
"""
from __future__ import annotations
import re
from collections import Counter, defaultdict
from data_core import norm


def safe_med_text(v, max_len=120):
    x = norm(v)
    if not x or len(x)>max_len: return ''
    if '@' in x or 'http://' in x or 'https://' in x or re.search(r'\d{7,}',x):
        return ''
    return x


def build_catalog(rows, visits, labels, min_ingredient_visits=5,
                  min_regimen_visits=5, min_way_visits=5,
                  recurring_ratio=0.9, min_recurring_visits=20):
    codes = set(labels)
    valid_visits = visits.loc[visits['valid_icd'] & visits['icd'].isin(codes), ['visit_key','icd']]
    eligible = dict(zip(valid_visits['visit_key'],valid_visits['icd']))
    # Một key = (tháng, mabn, makh). Không đếm 2 dòng thuốc thành 2 lượt khám.
    med_sets = defaultdict(set)
    med_display = Counter()
    way_sets = defaultdict(set)
    for rec in rows[['visit_key','tenhc','cachuong']].itertuples(index=False):
        code = eligible.get(rec.visit_key)
        if not code: continue
        ingredient = safe_med_text(rec.tenhc)
        if not ingredient: continue
        lower = ingredient.casefold()
        med_sets[rec.visit_key].add(lower)
        med_display[(lower,ingredient)] += 1
        way = safe_med_text(rec.cachuong,160)
        if way:
            way_sets[(code,lower,way)].add(rec.visit_key)
    canonical = {}
    for (low,real),count in med_display.items():
        if low not in canonical or count>canonical[low][1]:
            canonical[low] = (real,count)
    counts = {code:Counter() for code in labels}
    combo_counts = {code:Counter() for code in labels}
    prescription_visits = Counter()
    for visit, drugs in med_sets.items():
        code = eligible[visit]
        prescription_visits[code] += 1
        for med in drugs: counts[code][med] += 1
        # Giữ tổ hợp nguyên vẹn (theo dữ liệu lịch sử), không gọi là toa chuẩn.
        if 1 <= len(drugs) <= 12:
            combo_counts[code][tuple(sorted(drugs))] += 1
    catalog = {}
    for code in labels:
        denom = int(prescription_visits[code])
        ingredients = []
        for med,count in counts[code].most_common():
            if count < min_ingredient_visits: continue
            ways = [{'cachuong':way, 'so_luot':len(s)}
                    for (c,m,way),s in way_sets.items()
                    if c==code and m==med and len(s)>=min_way_visits]
            ways.sort(key=lambda a:(-a['so_luot'],a['cachuong']))
            proportion = count / denom if denom else 0
            ingredients.append({'tenhc':canonical[med][0], 'so_luot':int(count),
                                'tan_suat':round(proportion,4),
                                'xuat_hien_rat_thuong_xuyen':bool(count>=min_recurring_visits and proportion>=recurring_ratio),
                                'cach_dung_lich_su':ways[:3]})
        regimens = []
        for med_set,count in combo_counts[code].most_common(100):
            if count < min_regimen_visits: continue
            regimens.append({'hoat_chat':[canonical[m][0] for m in med_set],
                             'so_luot':int(count),'tan_suat':round(count/denom,4) if denom else 0})
            if len(regimens)>=10: break
        catalog[code] = {'so_luot_co_don':denom, 'hoat_chat':ingredients,
                         'to_hop_don_lich_su':regimens,
                         'quy_uoc_thuong_xuyen':f'Xuất hiện trong >= {100*recurring_ratio:.0f}% lượt có đơn và ít nhất {min_recurring_visits} lượt; không phải thuốc bắt buộc.'}
    return {'icd_catalog':catalog,
            'definition':'Dữ liệu tổng hợp toa lịch sử, không phải phác đồ, không dùng để kê đơn tự động.',
            'min_ingredient_visits':min_ingredient_visits,
            'min_way_visits':min_way_visits,
            'min_regimen_visits':min_regimen_visits}
