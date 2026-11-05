# -*- coding: utf-8 -*-
"""Ứng dụng NGHIÊN CỨU CHỈ CHẠY NỘI BỘ. Không dùng để tự chẩn đoán/kê đơn."""
from __future__ import annotations
import json
import re
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from data_core import clean_text, ICD_IN_TEXT, norm, sex_code, age_band, job_code
from rules_engine import rule_evidence, score_rule_only, score_hybrid
from revisit_engine import get_or_create_key, patient_status

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'artifacts'
DISPLAY = {
    'naive_bayes_trieu_chung':'Naive Bayes — chỉ triệu chứng',
    'linear_svm_trieu_chung':'Linear SVM — chỉ triệu chứng',
    'logistic_regression_trieu_chung':'Logistic Regression — chỉ triệu chứng',
    'naive_bayes_ca_nhan':'Naive Bayes — triệu chứng + cá nhân',
    'linear_svm_ca_nhan':'Linear SVM — triệu chứng + cá nhân',
    'logistic_regression_ca_nhan':'Logistic Regression — triệu chứng + cá nhân',
    'chi_luat':'Dự đoán chỉ dựa trên luật triệu chứng',
    'ket_hop_luat_va_mo_hinh':'Kết hợp luật và Logistic Regression cá nhân hóa',
}


@st.cache_data(show_spinner=False)
def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@st.cache_resource(show_spinner=False)
def load_model(path):
    # Chỉ nạp các file joblib do CHÍNH chương trình train.py sinh ra ở máy nội bộ.
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_history(path):
    return joblib.load(path)


def get_prediction(model_name, params, labels, rules):
    text = params['symptoms']
    # Các từ ngoài vocab không đủ để suy luận trong các mô hình TF-IDF.
    basis_name = ('logistic_regression_ca_nhan' if model_name in ('chi_luat','ket_hop_luat_va_mo_hinh')
                  else model_name)
    basis = load_model(str(OUT/'models'/f'{basis_name}.joblib'))
    transformer = basis.named_steps['features'].named_transformers_['text']
    if transformer.transform([text]).nnz == 0:
        return None, [], 'Không có từ/đặc trưng triệu chứng đã học; hệ thống không đưa ra dự đoán.'
    if model_name == 'chi_luat':
        result, matched = score_rule_only(text, rules, labels)
        if not result:
            return None, matched, 'Không có luật triệu chứng nào đạt ngưỡng với mô tả này.'
        return result, matched, ''
    if model_name == 'ket_hop_luat_va_mo_hinh':
        prob = basis.predict_proba(pd.DataFrame([params]))[0]
        classes = [str(v) for v in basis.named_steps['classifier'].classes_]
        ordered = [prob[classes.index(l)] for l in labels]
        pred, matched = score_hybrid(text,ordered,rules,labels)
        return pred,matched,''
    pred=str(basis.predict(pd.DataFrame([params]))[0])
    _score,matched=rule_evidence(text,rules,labels)
    return pred,matched,''


def show_drugs(catalog, code):
    info=catalog.get('icd_catalog',{}).get(code,{})
    n=int(info.get('so_luot_co_don',0))
    st.subheader('Thống kê hoạt chất theo mã bệnh')
    st.caption(f'Mã {code}: {n:,} lượt khám có đơn được ghi nhận trong dữ liệu; chỉ là thống kê lịch sử.')
    items=info.get('hoat_chat',[])
    if not items:
        st.info('Chưa có hoạt chất đủ số lượt ghi nhận cho mã này.')
    else:
        data=[{'Hoạt chất':row['tenhc'],'Số lượt':row['so_luot'],
               'Tỷ lệ trong lượt có đơn':f"{row['tan_suat']*100:.1f}%",
               'Xuất hiện ≥90% (chỉ báo)':('Có' if row['xuat_hien_rat_thuong_xuyen'] else '')}
              for row in items]
        query=st.text_input('Tìm hoạt chất',placeholder='Nhập tên hoạt chất...',key=f'find_{code}')
        if query:
            import unicodedata
            def fold(x):
                return ''.join(c for c in unicodedata.normalize('NFD',x.casefold())
                               if unicodedata.category(c)!='Mn').replace('đ','d')
            data=[x for x in data if fold(query) in fold(x['Hoạt chất'])]
        st.dataframe(pd.DataFrame(data),use_container_width=True,hide_index=True)
        st.caption('Xuất hiện ≥90% KHÔNG có nghĩa là thuốc bắt buộc hoặc phù hợp với người bệnh hiện tại.')
        with st.expander('Xem cách dùng từng ghi nhận trong hồ sơ (không phải chỉ định hiện tại)'):
            for row in items:
                for way in row.get('cach_dung_lich_su',[]):
                    st.write(f"**{row['tenhc']}**: {way['cachuong']} — {way['so_luot']} lượt")
    st.subheader('Các tổ hợp hoạt chất trong toa đã kê')
    combinations=info.get('to_hop_don_lich_su',[])
    if not combinations:
        st.info('Không có tổ hợp toa lịch sử đủ ngưỡng xuất hiện.')
    else:
        for i,reg in enumerate(combinations,1):
            with st.expander(f"Mẫu lịch sử {i}: {reg['so_luot']} lượt ({reg['tan_suat']*100:.1f}%)"):
                st.write('; '.join(reg['hoat_chat']))
                st.caption('Đây là thuốc đã cùng xuất hiện trong các toa cũ, không phải toa được kê cho người đang nhập triệu chứng.')


def main():
    st.set_page_config(page_title='Nghiên cứu phân loại bệnh theo triệu chứng',layout='wide',page_icon='🩺')
    st.title('Hệ thống phân loại bệnh theo triệu chứng lâm sàng')
    st.caption('Bản thử nghiệm nội bộ: phân loại nhóm ICD, phân tích luật, thống kê toa lịch sử và tra cứu tái khám.')
    meta_path=OUT/'metadata.json'
    if not meta_path.exists():
        st.error('Chưa huấn luyện V2. Hãy chạy: python train.py train --data-dir ..\\data')
        st.stop()
    try:
        meta=load_json(str(meta_path))
        rules=load_json(str(OUT/'rules.json')).get('rules',[])
        catalog=load_json(str(OUT/'drug_catalog.json'))
    except (OSError,ValueError) as ex:
        st.error(f'Thiếu tệp kết quả: {ex}'); st.stop()
    labels=meta['labels']
    options={k:DISPLAY[k] for k in DISPLAY
             if (k in ('chi_luat','ket_hop_luat_va_mo_hinh') and
                 (k!='chi_luat' or bool(rules)) and
                 (OUT/'models'/'logistic_regression_ca_nhan.joblib').is_file())
             or (OUT/'models'/f'{k}.joblib').is_file()}
    if not options:
        st.error('Chưa có mô hình trong artifacts/models.');st.stop()
    default = meta.get('best_model') if meta.get('best_model') in options else next(iter(options))
    with st.sidebar:
        st.header('Tùy chỉnh mô hình')
        chosen=st.selectbox('Thuật toán',options=list(options),index=list(options).index(default),
                            format_func=lambda k:options[k])
        st.caption('Các mô hình được huấn luyện riêng trên dữ liệu TRAIN; lựa chọn tốt nhất dựa trên Validation Macro F1.')
        st.write(f"**Mô hình dẫn đầu validation:** {DISPLAY.get(meta.get('best_model'),meta.get('best_model'))}")
        st.divider()
        st.write('**Đặc trưng được huấn luyện**')
        st.write('Nghề nghiệp:', 'Có dữ liệu' if meta.get('has_occupation_data') else 'Chưa có dữ liệu HIS')
        st.write('Tuổi:', 'Có' if meta.get('has_age_data') else 'Không có/không hợp lệ')
        st.write('Phân lớp ICD:', ', '.join(labels))
    if st.session_state.get('last_chosen_model') != chosen:
        st.session_state.pop('last_result',None)
        st.session_state['last_chosen_model'] = chosen
    with st.form('predict_form'):
        text=st.text_area('Triệu chứng lâm sàng',height=125,max_chars=1500,
                          placeholder='Nhập các biểu hiện, thời điểm và dấu hiệu liên quan, không nhập tên bệnh hay mã ICD.')
        a,b,c=st.columns(3)
        with a: sex=st.selectbox('Giới tính',options=['Không rõ','Nam','Nữ','Khác'])
        with b: age_text=st.text_input('Tuổi (nếu biết)',placeholder='Ví dụ: 45')
        with c: occupation=st.text_input('Nghề nghiệp (nếu biết)',placeholder='Ví dụ: công nhân')
        st.caption('Nghề nghiệp chỉ có ảnh hưởng đến dự đoán nếu dữ liệu huấn luyện có trường nghề nghiệp hoặc bảng bổ sung hợp lệ.')
        submit=st.form_submit_button('Phân tích mã bệnh',type='primary',use_container_width=True)
    if submit:
        cleaned=clean_text(text)
        if len(cleaned)<5:
            st.warning('Vui lòng mô tả triệu chứng rõ hơn.');st.stop()
        if len(cleaned)>1500:
            st.warning('Văn bản quá dài.');st.stop()
        if ICD_IN_TEXT.search(cleaned) or any(re.search(r'(?<!\w)'+re.escape(term)+r'(?!\w)',cleaned)
                                                  for term in meta.get('leak_terms',[])):
            st.warning('Đầu vào chứa mã hoặc tên chẩn đoán rõ ràng. Vui lòng nhập biểu hiện triệu chứng.');st.stop()
        age=None
        if age_text.strip():
            try:
                age=int(age_text)
                if not 0<=age<=110:raise ValueError
            except ValueError:
                st.warning('Tuổi cần là số nguyên từ 0 đến 110.');st.stop()
        if occupation.strip() and not meta.get('has_occupation_data'):
            st.info('Chưa có nghề nghiệp trong dữ liệu huấn luyện. Trường nghề nghiệp bạn vừa nhập sẽ không ảnh hưởng đến dự đoán.')
        params={'symptoms':cleaned,'sex':sex_code(sex),
                'age_band':age_band(age),
                'occupation':job_code(occupation) if meta.get('has_occupation_data') else 'Không rõ'}
        try: pred,matched,msg=get_prediction(chosen,params,labels,rules)
        except Exception as exc:
            st.error(f'Không thể chạy mô hình: {type(exc).__name__}: {exc}');st.stop()
        if msg:st.warning(msg)
        if pred:
            st.session_state['last_result']={'code':pred,'matched':matched,
                                             'model':chosen,'name':meta['names'].get(pred,'')}
        else:
            st.session_state.pop('last_result',None)
    if st.session_state.get('last_result'):
        res=st.session_state['last_result']
        st.subheader('Kết quả phân lớp (tham khảo nghiên cứu)')
        st.success(f"Nhóm ICD dự đoán: {res['code']} — {res['name']}")
        st.caption('Kết quả là NHÓM ICD 3 ký tự. Các dấu hiệu đầu vào không đủ để xác nhận bệnh, nhất là những bệnh cần xét nghiệm hoặc đo lường.')
        if res['matched']:
            with st.expander('Các luật triệu chứng trùng khớp'):
                st.dataframe(pd.DataFrame([{'Cụm triệu chứng':r['phrase'], 'ICD':r['icd'],
                                            'Số lượt chứa cụm':r['support'],
                                            'Độ tin cậy trên train':round(r['confidence']*100,1),
                                            'Lift':r['lift']} for r in res['matched']]),
                             use_container_width=True,hide_index=True)
        with st.expander('Xem thống kê hoạt chất và tổ hợp toa lịch sử',expanded=False):
            show_drugs(catalog,res['code'])
        st.subheader('Kiểm tra tái khám theo mã bệnh')
        st.caption('Bắt buộc cần mã bệnh nhân trên máy nội bộ; thông tin nhập không được lưu trong mô hình hay báo cáo. KHÔNG dùng trên bản public.')
        with st.form('revisit_form'):
            patient_id=st.text_input('Mã bệnh nhân trong HIS',type='password')
            day=st.date_input('Ngày khám hiện tại / ngày cần đối chiếu',value=date.today())
            check=st.form_submit_button('Kiểm tra lịch sử tái khám')
        if check:
            if not patient_id.strip():
                st.warning('Cần mã bệnh nhân để tra cứu lịch sử.')
            else:
                history_path=OUT/'patient_history_local.joblib'
                keypath=OUT/'.hmac_key'
                if not history_path.exists() or not keypath.exists():
                    st.error('Chưa có dữ liệu tái khám được tạo từ 12 tháng.')
                else:
                    # KHÔNG tạo khoá mới nếu bị mất: khoá mới sẽ khiến token không khớp.
                    history=load_history(str(history_path));key=keypath.read_bytes()
                    status=patient_status(norm(patient_id),res['code'],day.isoformat(),history,key)
                    if status['status']=='followup_same_icd':st.info(status['text'])
                    else:st.write(status['text'])
    st.divider()
    st.caption('Chỉ để nghiên cứu/thử nghiệm trên localhost; không phải thiết bị y tế, không xác lập đơn thuốc hay phác đồ điều trị.')

if __name__=='__main__':main()
