# -*- coding: utf-8 -*-
"""HUẤN LUYỆN MỚI HOÀN TOÀN - triệu chứng, đặc trưng cá nhân, luật, toa lịch sử, tái khám.

Ví dụ:
 python train.py audit --data-dir E:\\HeThongChanDoan\\data
 python train.py train --data-dir E:\\HeThongChanDoan\\data --out ./artifacts
 python train.py train --data-dir E:\\HeThongChanDoan\\data --jobs-csv ./nghe_nghiep.csv

Chỉ sử dụng bản ghi HIS được phép xử lý và máy tính được kiểm soát truy cập.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from data_core import read_data, select_training_visits
from rules_engine import mine_rules, score_rule_only, score_hybrid
from drugs_engine import build_catalog
from revisit_engine import get_or_create_key, make_history, revisit_flags

BASE = Path(__file__).resolve().parent
DEFAULT_DATA = BASE.parent / 'data'
DEFAULT_OUT = BASE / 'artifacts'
ABSTAIN = '__KHONG_CO_LUAT__'
FEATURES = ['symptoms','sex','age_band','occupation']


def save_json(data, file):
    path=Path(file);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def save_csv(df, file):
    path=Path(file);path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path,index=False,encoding='utf-8-sig')


def split_by_patient(frame, labels, seed):
    y=frame['icd'].to_numpy(); groups=frame['patient'].to_numpy()
    required=set(labels)
    best=None
    for s in (seed,seed+17,seed+61):
        outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=s)
        for a,te in outer.split(frame,y,groups):
            if set(y[te])!=required or set(y[a])!=required:continue
            inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=s+101)
            for r,v in inner.split(frame.iloc[a],y[a],groups[a]):
                tr,va=a[r],a[v]
                if any(set(y[idx])!=required for idx in (tr,va,te)): continue
                if any(min(Counter(y[idx]).values())<2 for idx in (tr,va,te)): continue
                err=sum(abs(len(idx)/len(y)-want) for idx,want in ((tr,.64),(va,.16),(te,.20)))
                for lab in labels:
                    count=(y==lab).sum()
                    err+=sum(abs((y[idx]==lab).sum()/count-want)/len(labels)
                             for idx,want in ((tr,.64),(va,.16),(te,.2)))
                if best is None or err<best[0]:best=(err,tr,va,te,s)
        if best is not None:break
    if best is None:
        raise RuntimeError('Không chia được train/validation/test đủ 10 nhóm theo bệnh nhân. Kiểm tra nhóm ít bệnh nhân hoặc giảm số nhóm chỉ sau khi thống nhất phương pháp.')
    _,tr,va,te,used=best
    assert not (set(groups[tr]) & set(groups[va]) | set(groups[tr]) & set(groups[te]) | set(groups[va]) & set(groups[te]))
    return tr,va,te,used


def features_maker(personalized):
    transformers=[('text',TfidfVectorizer(ngram_range=(1,2),lowercase=False,
                    token_pattern=r'(?u)\b\w+\b',max_features=70000,
                    min_df=2,max_df=.995,sublinear_tf=True),'symptoms')]
    if personalized:
        transformers += [('age',OneHotEncoder(handle_unknown='ignore'),['age_band']),
                         ('sex',OneHotEncoder(handle_unknown='ignore'),['sex']),
                         ('occupation',OneHotEncoder(handle_unknown='ignore'),['occupation'])]
    return ColumnTransformer(transformers, sparse_threshold=1.0)


def new_classifiers():
    return {'naive_bayes':MultinomialNB(alpha=1.0),
            'linear_svm':LinearSVC(class_weight='balanced',random_state=42),
            'logistic_regression':LogisticRegression(class_weight='balanced',max_iter=1600,random_state=42)}


def measurements(y_true,y_pred,labels):
    a=np.asarray(y_true,dtype=str);b=np.asarray(y_pred,dtype=str)
    return {'n':len(a),'accuracy':round(float(accuracy_score(a,b)),6),
            'balanced_accuracy':round(float(balanced_accuracy_score(a,b)),6),
            'macro_f1':round(float(f1_score(a,b,labels=labels,average='macro',zero_division=0)),6),
            'weighted_f1':round(float(f1_score(a,b,labels=labels,average='weighted',zero_division=0)),6),
            'cohen_kappa':round(float(cohen_kappa_score(a,b)),6)}


def audit(args):
    rows,visits,names,report=read_data(args.data_dir,args.jobs_csv,args.sex_encoding)
    usable,freq,labels,leak_terms=select_training_visits(visits,names,args.top)
    return rows,visits,names,report,usable,freq,labels,leak_terms


def run(args):
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    print('\n[1] Đọc 12 tháng và gom theo lượt khám')
    rows,visits,names,report,usable,freq,labels,leak_terms=audit(args)
    print('\nTop nhóm:',labels)
    report.update({'top_n':args.top,'top_icd':labels,
                   'n_visits_training':len(usable),
                   'n_training_no_job':int(usable['occupation'].eq('Không rõ').sum()),
                   'n_training_no_age':int(usable['age_band'].eq('Không rõ').sum()),
                   'n_training_no_sex':int(usable['sex'].eq('Không rõ').sum()),
                   'training_excludes_diagnostic_terms':True,
                   'label_unit':'ICD 3 ký tự; một bệnh nhân có thể có nhiều lượt khám'})
    save_json(report,out/'audit.json')
    save_csv(freq,out/'tan_suat_icd.csv')
    if args.action=='audit':
        print('Đã xuất:',out/'audit.json');return
    counts=usable.groupby('icd').agg(so_luot=('patient','size'),
                                    so_benh_nhan=('patient','nunique'))
    bad=counts.loc[(counts['so_luot']<20)|(counts['so_benh_nhan']<12)]
    if len(bad):
        raise RuntimeError(f'Chưa đủ số lượt/bệnh nhân cho kiểm thử phân nhóm, xem audit và tan_suat_icd.csv: {bad.to_dict("index")}')
    print('\n[2] Chia theo bệnh nhân, chống trùng người giữa các tập')
    tr,va,te,s=split_by_patient(usable,labels,args.seed)
    tr_df=usable.iloc[tr].copy(); va_df=usable.iloc[va].copy(); te_df=usable.iloc[te].copy()
    split_report={'train':len(tr),'validation':len(va),'test':len(te),
                  'patients_train':tr_df['patient'].nunique(),
                  'patients_validation':va_df['patient'].nunique(),
                  'patients_test':te_df['patient'].nunique(),
                  'seed':s,'same_patient_across_splits':False}
    save_json(split_report,out/'chia_du_lieu.json')
    stats=[]
    for lab in labels:
        stats.append({'ma_icd':lab,'ten_benh':names.get(lab,''),
                      'train':int((tr_df.icd==lab).sum()),
                      'validation':int((va_df.icd==lab).sum()),
                      'test':int((te_df.icd==lab).sum())})
    save_csv(pd.DataFrame(stats),out/'phan_bo_10_benh.csv')
    seen=set(tr_df['symptoms'])
    save_json({'validation_text_identical_to_train':int(va_df['symptoms'].isin(seen).sum()),
               'test_text_identical_to_train':int(te_df['symptoms'].isin(seen).sum())},
              out/'canh_bao_trung_van_ban.json')
    all_preds={}
    metrics=[]
    fitted={}
    (out/'models').mkdir(parents=True,exist_ok=True)
    print('\n[3] Huấn luyện 6 cấu hình: 3 thuật toán x 2 loại đặc trưng')
    for personalized in (False,True):
        profile='ca_nhan' if personalized else 'trieu_chung'
        features=features_maker(personalized)
        Xtr=features.fit_transform(tr_df[FEATURES])
        Xva=features.transform(va_df[FEATURES]); Xte=features.transform(te_df[FEATURES])
        for algo,clf in new_classifiers().items():
            name=f'{algo}_{profile}'
            print('  -',name)
            if algo=='naive_bayes':
                sw=compute_sample_weight(class_weight='balanced',y=tr_df['icd'])
                clf.fit(Xtr,tr_df['icd'],sample_weight=sw)
            else:
                clf.fit(Xtr,tr_df['icd'])
            # Features đã FIT riêng trên train. Tạo pipeline suy luận, KHÔNG fit lại.
            pipe=Pipeline([('features',features),('classifier',clf)])
            joblib.dump(pipe,out/'models'/f'{name}.joblib',compress=3)
            fitted[name]=pipe
            for split,frame,x in [('validation',va_df,Xva),('test',te_df,Xte)]:
                pred=clf.predict(x)
                all_preds[(name,split)]=pred
                metrics.append({'model':name,'split':split,
                                **measurements(frame['icd'],pred,labels)})
    print('\n[4] Khai phá luật TRIỆU CHỨNG -> ICD từ TRAIN, không dùng test')
    rules=mine_rules(tr_df['symptoms'].tolist(),tr_df['icd'].tolist(),labels,
                     min_support=args.rule_support,min_confidence=args.rule_confidence,
                     min_lift=args.rule_lift)
    save_json({'rules':rules,'min_support':args.rule_support,
               'min_confidence':args.rule_confidence,'min_lift':args.rule_lift},
              out/'rules.json')
    print('  Luật vượt ngưỡng:',len(rules))
    base=fitted['logistic_regression_ca_nhan']
    for split,frame in [('validation',va_df),('test',te_df)]:
        texts=frame['symptoms'].tolist()
        rules_only=[score_rule_only(t,rules,labels)[0] or ABSTAIN for t in texts]
        probs=base.predict_proba(frame[FEATURES])
        base_labels=[str(x) for x in base.named_steps['classifier'].classes_]
        order=[base_labels.index(l) for l in labels]
        hybrid=[score_hybrid(t,p[order],rules,labels)[0] for t,p in zip(texts,probs)]
        for name,pred in [('chi_luat',rules_only),('ket_hop_luat_va_mo_hinh',hybrid)]:
            all_preds[(name,split)]=pred
            metrics.append({'model':name,'split':split,
                            **measurements(frame['icd'],pred,labels),
                            'rule_coverage':round(sum(p!=ABSTAIN for p in pred)/len(pred),6) if name=='chi_luat' else 1})
    save_csv(pd.DataFrame(metrics),out/'so_sanh_mo_hinh.csv')
    val_rows=[r for r in metrics if r['split']=='validation']
    best=sorted(val_rows,key=lambda r:(-r['macro_f1'],-r['balanced_accuracy'],-r['accuracy']))[0]['model']
    save_json({'selected_by':'validation_macro_f1', 'best_model':best,
               'test_metrics':next(x for x in metrics if x['model']==best and x['split']=='test')},out/'chon_mo_hinh.json')
    print('  Tốt nhất theo validation Macro-F1:',best)
    for lab in ['validation','test']:
        pred=all_preds[(best,lab)];truth=(va_df if lab=='validation' else te_df)['icd']
        save_csv(pd.DataFrame(classification_report(truth,pred,labels=labels,
                                                    output_dict=True,zero_division=0)).T
                 .rename_axis('nhan').reset_index(),out/f'bao_cao_tung_benh_{lab}.csv')
        cm=confusion_matrix(truth,pred,labels=labels+[ABSTAIN])
        save_csv(pd.DataFrame(cm,index=labels+[ABSTAIN],columns=labels+[ABSTAIN])
                 .rename_axis('icd_thuc').reset_index(),out/f'ma_tran_nham_lan_{lab}.csv')
    print('\n[5] Thống kê hoạt chất/toa lịch sử theo ICD (không kê đơn tự động)')
    catalog=build_catalog(rows,visits,labels,
                          min_ingredient_visits=args.min_med_visits,
                          min_regimen_visits=args.min_regimen_visits,
                          min_way_visits=args.min_way_visits)
    save_json(catalog,out/'drug_catalog.json')
    print('\n[6] Tái khám cùng nhóm ICD theo ngày ghi nhận')
    key=get_or_create_key(out/'.hmac_key')
    history,n_date=make_history(visits,key,rows)
    joblib.dump(history,out/'patient_history_local.joblib',compress=3)
    summary=revisit_flags(history)
    summary['n_source_events_with_valid_date']=n_date
    save_json(summary,out/'tai_kham_tong_hop.json')
    personal_data_present=(usable['occupation']!='Không rõ').any()
    meta={'labels':labels,'names':{k:names.get(k,'') for k in labels},
          'best_model':best,'model_names':sorted(fitted),
          'has_occupation_data':bool(personal_data_present),
          'has_age_data':bool(usable['age'].notna().any()),
          'leak_terms':leak_terms,
          'limits':['ICD chỉ ở mức 3 ký tự','Không có bằng chứng xác nhận bệnh mạn tính từ tái khám đơn thuần',
                    'Thuốc trong toa lịch sử không phải thuốc bắt buộc','Hệ thống chưa có kiểm chứng lâm sàng độc lập'],
          'data_directory':str(args.data_dir),
          'n_patients_in_history':len(history)}
    save_json(meta,out/'metadata.json')
    print('\nHOÀN THÀNH. Báo cáo:',out/'so_sanh_mo_hinh.csv')
    print('Không công khai thư mục artifacts/ vì chứa mô hình, dữ liệu suy ra từ HIS và dữ liệu tái khám nhạy cảm.')


def main():
    p=argparse.ArgumentParser(description='Nghiên cứu phân loại ICD, luật triệu chứng, tra cứu thuốc và tái khám từ 12 CSV HIS')
    p.add_argument('action',choices=['audit','train'])
    p.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--jobs-csv',type=Path,default=None,help='Tùy chọn: CSV nội bộ mabn,nghenghiep')
    p.add_argument('--sex-encoding',choices=['text','1nam0nu','0nam1nu','1nam2nu'],default='text',
                   help='Mã 0/1/2 không mặc định quy đổi giới tính; phải xác nhận HIS trước khi chọn')
    p.add_argument('--top',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rule-support',type=int,default=15)
    p.add_argument('--rule-confidence',type=float,default=.62)
    p.add_argument('--rule-lift',type=float,default=1.18)
    p.add_argument('--min-med-visits',type=int,default=5)
    p.add_argument('--min-way-visits',type=int,default=5)
    p.add_argument('--min-regimen-visits',type=int,default=5)
    args=p.parse_args()
    if args.top!=10:p.error('Phiên bản nghiên cứu này cố định 10 nhóm ICD. --top phải bằng 10.')
    if args.rule_support<3 or not(0 < args.rule_confidence <=1) or args.rule_lift<1:
        p.error('Ngưỡng luật không hợp lệ')
    if min(args.min_med_visits,args.min_way_visits,args.min_regimen_visits)<3:
        p.error('Ngưỡng tổng hợp thuốc/toa phải >=3 lượt để giảm rủi ro công bố dữ liệu hiếm')
    run(args)

if __name__=='__main__':main()
