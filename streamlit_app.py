# -*- coding: utf-8 -*-
"""Giao diện V2: phân loại ICD, tra cứu hoạt chất và phiếu rà soát.

Chạy: python -m streamlit run streamlit_app.py
"""
from __future__ import annotations

from datetime import date
from hashlib import sha1
import json
from pathlib import Path
import re

import joblib
import pandas as pd
import streamlit as st
from streamlit_searchbox import st_searchbox

from data_core import clean_text, ICD_IN_TEXT, norm, sex_code, age_band, job_code
from revisit_engine import patient_status
from rules_engine import rule_evidence, score_rule_only, score_hybrid
from ui_helpers import fold_vi, ingredient_suggestions, historical_ways, make_draft_html

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts'
PER_PAGE=10
DISPLAY={
    'naive_bayes_trieu_chung':'Naive Bayes · Triệu chứng',
    'linear_svm_trieu_chung':'Linear SVM · Triệu chứng',
    'logistic_regression_trieu_chung':'Logistic Regression · Triệu chứng',
    'naive_bayes_ca_nhan':'Naive Bayes · Cá nhân hóa',
    'linear_svm_ca_nhan':'Linear SVM · Cá nhân hóa',
    'logistic_regression_ca_nhan':'Logistic Regression · Cá nhân hóa',
    'chi_luat':'Chỉ luật triệu chứng',
    'ket_hop_luat_va_mo_hinh':'Luật + Logistic Regression',
}


@st.cache_data(show_spinner=False)
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    # Chỉ sử dụng joblib do chính train.py V2 huấn luyện và lưu tại máy nội bộ.
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_history(path: str):
    return joblib.load(path)


def get_prediction(model_name, params, labels, rules):
    symptoms=params['symptoms']
    basis_name=('logistic_regression_ca_nhan' if model_name in
                ('chi_luat','ket_hop_luat_va_mo_hinh') else model_name)
    basis=load_model(str(OUT/'models'/f'{basis_name}.joblib'))
    text_transformer=basis.named_steps['features'].named_transformers_['text']
    if text_transformer.transform([symptoms]).nnz==0:
        return None, [], 'Không có đặc trưng triệu chứng đã học. Vui lòng mô tả rõ hơn.'
    if model_name=='chi_luat':
        predicted,matched=score_rule_only(symptoms,rules,labels)
        return predicted,matched,('' if predicted else 'Không tìm thấy luật triệu chứng đạt ngưỡng.')
    if model_name=='ket_hop_luat_va_mo_hinh':
        probability=basis.predict_proba(pd.DataFrame([params]))[0]
        classes=list(map(str,basis.named_steps['classifier'].classes_))
        ordered=[probability[classes.index(label)] for label in labels]
        predicted,matched=score_hybrid(symptoms,ordered,rules,labels)
        return predicted,matched,''
    predicted=str(basis.predict(pd.DataFrame([params]))[0])
    _,matched=rule_evidence(symptoms,rules,labels)
    return predicted,matched,''


def css():
    st.markdown('''<style>
    .block-container{max-width:1120px;padding-top:1.35rem;padding-bottom:2.8rem}
    h1{letter-spacing:-.03em;font-size:1.9rem!important}
    h2{font-size:1.22rem!important}
    h3{font-size:1.03rem!important}
    [data-testid="stSidebar"]{background:#f7f9fc}
    div[data-testid="stMetric"]{border:1px solid #e2e8f0;padding:12px 18px;border-radius:10px}
    div[data-testid="stVerticalBlockBorderWrapper"]{border-color:#e2e8f0}
    </style>''',unsafe_allow_html=True)


def reset_encounter():
    for key in ('last_result','selected_meds','drug_page','selected_search','visit_sig'):
        st.session_state.pop(key,None)
    for key in ('patient_id','symptoms','age_text','occupation','clinician','draft_notes'):
        st.session_state[key]=''
    st.session_state['declared_visit']='Chưa xác định'
    st.session_state['search_epoch']=st.session_state.get('search_epoch',0)+1
    st.session_state['med_epoch']=st.session_state.get('med_epoch',0)+1
    st.session_state['encounter_epoch']=st.session_state.get('encounter_epoch',0)+1


def on_model_changed():
    for key in ('last_result','selected_meds','drug_page','selected_search','visit_sig'):
        st.session_state.pop(key,None)
    st.session_state['search_epoch']=st.session_state.get('search_epoch',0)+1
    st.session_state['med_epoch']=st.session_state.get('med_epoch',0)+1
    st.session_state['encounter_epoch']=st.session_state.get('encounter_epoch',0)+1


def select_med(name: str, cb_key: str):
    selected=list(st.session_state.get('selected_meds',[]))
    if st.session_state.get(cb_key,False):
        if name not in selected:
            selected.append(name)
    else:
        selected=[n for n in selected if n!=name]
    st.session_state['selected_meds']=selected


def remove_med(name: str):
    st.session_state['selected_meds']=[n for n in st.session_state.get('selected_meds',[]) if n!=name]
    # Không được sửa session_state của checkbox đã render trong cùng lượt Streamlit.
    # Đổi thế hệ key để checkbox được tạo lại với giá trị đúng sau rerun.
    st.session_state['med_epoch']=st.session_state.get('med_epoch',0)+1


def med_widget_key(name: str) -> str:
    return 'select_med_'+str(st.session_state.get('med_epoch',0))+'_'+sha1(name.encode('utf-8')).hexdigest()[:14]


def show_visit_status(patient_id, target_icd, when, declared):
    st.markdown('##### Tình trạng lượt khám')
    if not patient_id.strip():
        st.info('Chưa nhập mã bệnh nhân. Không thể đối chiếu lịch sử tái khám.')
        return
    path=OUT/'patient_history_local.joblib';keypath=OUT/'.hmac_key'
    if not path.is_file() or not keypath.is_file():
        st.warning('Chưa có dữ liệu lịch sử hoặc khóa bảo vệ; chỉ ghi nhận loại lượt khám theo khai báo.')
        return
    try:
        status=patient_status(norm(patient_id),target_icd,when.isoformat(),
                              load_history(str(path)),keypath.read_bytes())
    except (OSError,KeyError,ValueError,TypeError) as ex:
        st.warning(f'Không đối chiếu được lịch sử: {type(ex).__name__}.')
        return
    if status['status']=='followup_same_icd':
        st.info('Có hồ sơ khám trước cùng nhóm ICD trong dữ liệu quan sát. '+status['text'])
        if declared=='Khám lần đầu':
            st.warning('Khai báo “Khám lần đầu” khác với lịch sử cùng nhóm ICD; cần kiểm tra với HIS.')
    elif status['status']=='previous_other_icd':
        st.info('Đã có lượt khám trước, nhưng chưa ghi nhận cùng nhóm ICD trong khoảng dữ liệu. '+status['text'])
    else:
        st.info('Chưa ghi nhận lượt khám trước trong dữ liệu 12 tháng. Không đủ căn cứ xác nhận bệnh nhân khám lần đầu trên thực tế.')
    if declared=='Tái khám' and status['status']!='followup_same_icd':
        st.caption('“Tái khám” là thông tin được khai báo, chưa được lịch sử cùng nhóm ICD xác nhận.')


def render_med_list(items, code, names):
    from math import ceil
    def search(term: str):
        return ingredient_suggestions(term,names,limit=12)
    selected_search=st_searchbox(
        search,
        key=f'autocomplete_{code}_{st.session_state.get("search_epoch",0)}',
        label='Tìm kiếm hoạt chất (gợi ý ngay khi nhập, không phân biệt dấu)',
        placeholder='Nhập từ bất kỳ trong tên hoạt chất…',
        debounce=150,default_options=names[:12],
    )
    if selected_search and selected_search in names:
        if st.session_state.get('selected_search')!=selected_search:
            st.session_state['drug_page']=1
        st.session_state['selected_search']=selected_search
    filt=st.session_state.get('selected_search')
    filtered=[x for x in items if x['tenhc']==filt] if filt else items
    col_info,col_clear=st.columns([4,1])
    with col_info:
        st.caption(f'Hiển thị {len(filtered)} / {len(items)} hoạt chất · A–Z')
    with col_clear:
        if filt and st.button('Hiện tất cả',key='clear_drug_filter',use_container_width=True):
            st.session_state.pop('selected_search',None)
            st.session_state['search_epoch']=st.session_state.get('search_epoch',0)+1
            st.session_state['drug_page']=1
            st.rerun()
    pages=max(1,ceil(len(filtered)/PER_PAGE))
    current=max(1,min(st.session_state.get('drug_page',1),pages))
    st.session_state['drug_page']=current
    shown=filtered[(current-1)*PER_PAGE:current*PER_PAGE]
    if not shown:
        st.info('Không có hoạt chất phù hợp với bộ lọc.')
    for record in shown:
        name=record['tenhc']
        key=med_widget_key(name)
        checked=name in st.session_state.get('selected_meds',[])
        a,b=st.columns([0.6,7.5],vertical_alignment='top')
        with a:
            # Widget có thể bị hủy khi đổi trang. Trạng thái lựa chọn luôn
            # được duy trì riêng trong selected_meds.
            if key not in st.session_state:
                st.session_state[key]=checked
            st.checkbox('Chọn',key=key,label_visibility='collapsed',
                        on_change=select_med,args=(name,key))
        with b:
            st.write(f'**{name}**')
            ways=historical_ways(record)
            if ways:
                for way in ways:
                    st.caption('Cách uống / cách dùng đã ghi nhận: '+way)
            else:
                st.caption('Chưa có nội dung cách dùng đạt ngưỡng từ dữ liệu gốc.')
        st.divider()
    if pages>1:
        prev,indicator,nxt=st.columns([1,2,1],vertical_alignment='center')
        with prev:
            if st.button('‹ Trước',disabled=current==1,use_container_width=True):
                st.session_state['drug_page']=current-1;st.rerun()
        with indicator: st.markdown(f'<p style="text-align:center">Trang {current} / {pages}</p>',unsafe_allow_html=True)
        with nxt:
            if st.button('Sau ›',disabled=current>=pages,use_container_width=True):
                st.session_state['drug_page']=current+1;st.rerun()


def render_draft(code,diagnosis,patient_id,when,declared,items):
    by_name={r['tenhc']:r for r in items}
    chosen=[name for name in st.session_state.get('selected_meds',[]) if name in by_name]
    st.markdown('##### Phiếu rà soát hoạt chất')
    if not chosen:
        st.info('Chọn ít nhất một hoạt chất ở danh sách bên trên.')
        return
    st.write(f'Đã chọn **{len(chosen)} hoạt chất**:')
    medications=[]
    for name in chosen:
        sig=str(st.session_state.get('encounter_epoch',0))+'_'+sha1(name.encode('utf-8')).hexdigest()[:14]
        with st.container(border=True):
            col_name,col_remove=st.columns([5,1])
            col_name.markdown('**'+name+'**')
            if col_remove.button('Bỏ',key='remove_'+sig,use_container_width=True):
                remove_med(name);st.rerun()
            x,y=st.columns([3,1.4])
            with x:
                instructions=st.text_input('Cách uống / cách sử dụng (do bác sĩ nhập)',
                                           key='rx_instruction_'+sig,placeholder='Nhập sau khi đánh giá người bệnh')
            with y:
                quantity=st.text_input('Số lượng / đơn vị',key='rx_qty_'+sig,placeholder='VD: 10 viên')
            medications.append({'name':name,'instructions':instructions,'quantity':quantity})
    clinician=st.text_input('Người lập phiếu (bác sĩ / cán bộ chuyên môn)',key='clinician')
    notes=st.text_area('Ghi chú của người lập phiếu (không bắt buộc)',key='draft_notes',height=80)
    review=st.checkbox('Đã kiểm tra chỉ định, dị ứng, tương tác thuốc và liều dùng',key='rx_review_'+str(st.session_state.get('encounter_epoch',0)))
    valid=(bool(patient_id.strip()) and bool(clinician.strip()) and bool(review) and all(m['instructions'].strip() and m['quantity'].strip() for m in medications))
    if not valid:
        st.caption('Để tạo bản in: nhập mã bệnh nhân ở đầu trang, người lập phiếu, cách dùng và số lượng cho từng hoạt chất, rồi xác nhận đã rà soát.')
        st.button('Tạo phiếu in',disabled=True,use_container_width=True)
        return
    try:
        html=make_draft_html(patient_id=patient_id,encounter_date=when,visit_type=declared,
                             icd=code,diagnosis=diagnosis,clinician=clinician,
                             medications=medications,notes=notes)
    except ValueError as ex:
        st.error(str(ex));return
    st.download_button('Tải phiếu HTML',data=html.encode('utf-8'),
                       file_name=f'phieu_ra_soat_{code}_{when.isoformat()}.html',
                       mime='text/html',type='primary',use_container_width=True)
    st.caption('Phiếu in ở trạng thái chờ phê duyệt.')


def show_drugs(catalog,code,diagnosis,patient_id,when,declared):
    info=catalog.get('icd_catalog',{}).get(code,{})
    items=info.get('hoat_chat',[]) or []
    st.subheader('Hoạt chất theo nhóm ICD '+code)
    if not items:
        st.warning('Chưa có hoạt chất đủ dữ liệu cho nhóm ICD này. Kiểm tra artifacts/drug_catalog.json và các ngưỡng thống kê khi huấn luyện.')
        return
    items=sorted(items,key=lambda x:(fold_vi(x.get('tenhc','')),x.get('tenhc','')))
    names=[i['tenhc'] for i in items if i.get('tenhc')]
    # Loại những lựa chọn thuộc nhóm ICD cũ.
    st.session_state['selected_meds']=[n for n in st.session_state.get('selected_meds',[]) if n in names]
    tab_drugs,tab_draft,tab_stats=st.tabs(['Danh sách hoạt chất','Phiếu in','Thống kê'])
    with tab_drugs:
        render_med_list(items,code,names)
    with tab_draft:
        render_draft(code,diagnosis,patient_id,when,declared,items)
    with tab_stats:
        st.metric('Lượt khám có đơn (nhóm ICD)',f"{int(info.get('so_luot_co_don',0)):,}")
        rows=[{'Hoạt chất':i['tenhc'], 'Lượt ghi nhận':i.get('so_luot',0),
               'Tỷ lệ (%)':round(100*i.get('tan_suat',0),1)} for i in items]
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        with st.expander('Tổ hợp hoạt chất thường đồng xuất hiện'):
            combos=info.get('to_hop_don_lich_su',[])
            if combos:
                for row in combos:
                    st.write('; '.join(row['hoat_chat']))
                    st.caption(f"Ghi nhận {row['so_luot']} lượt; tỷ lệ {100*row['tan_suat']:.1f}%")
            else:st.caption('Không có tổ hợp đủ ngưỡng thống kê.')


def main():
    st.set_page_config(page_title='Hệ thống chẩn đoán · V2',page_icon='🩺',layout='wide')
    css()
    st.title('Hệ thống chẩn đoán bệnh')
    meta_path=OUT/'metadata.json'
    if not meta_path.is_file():
        st.error('Chưa có mô hình V2. Chạy: python train.py train --data-dir E:\\HeThongChanDoan\\data')
        st.stop()
    try:
        meta=load_json(str(meta_path))
        rules=load_json(str(OUT/'rules.json')).get('rules',[])
        catalog=load_json(str(OUT/'drug_catalog.json'))
    except (OSError,ValueError,KeyError) as exc:
        st.error(f'Không tải được dữ liệu mô hình hoặc danh mục: {exc}')
        st.stop()
    labels=meta['labels']
    options={name:label for name,label in DISPLAY.items()
             if ((name in ('chi_luat','ket_hop_luat_va_mo_hinh') and
                 (name!='chi_luat' or bool(rules)) and
                 (OUT/'models'/'logistic_regression_ca_nhan.joblib').is_file())
                 or (OUT/'models'/f'{name}.joblib').is_file())}
    if not options:
        st.error('Không có mô hình hợp lệ trong artifacts/models/.');st.stop()
    default=meta.get('best_model') if meta.get('best_model') in options else next(iter(options))
    with st.sidebar:
        st.markdown('### Thuật toán')
        chosen=st.selectbox('Mô hình suy luận',options=list(options),
                            index=list(options).index(default),format_func=lambda x:options[x],
                            key='model_choice',on_change=on_model_changed)
        st.caption('Mặc định: mô hình có Macro F1 cao nhất trên tập validation.')
        st.divider()
        st.markdown('**Dữ liệu đặc trưng**')
        st.caption('Tuổi: '+('Có' if meta.get('has_age_data') else 'Chưa đủ dữ liệu'))
        st.caption('Nghề nghiệp: '+('Có' if meta.get('has_occupation_data') else 'Chưa có dữ liệu HIS'))
        st.divider()
        if st.button('Làm mới lượt khám',use_container_width=True,on_click=reset_encounter):
            st.rerun()

    st.markdown('### 1. Tiếp nhận người bệnh')
    with st.form('encounter_form'):
        c1,c2,c3=st.columns([2,1.2,1.3])
        with c1:
            patient_id=st.text_input('Mã bệnh nhân HIS',key='patient_id',
                                     placeholder='Nhập mã bệnh nhân để đối chiếu lịch sử')
        with c2:
            declared=st.selectbox('Loại lượt khám (khai báo)',
                                  ['Chưa xác định','Khám lần đầu','Tái khám'],key='declared_visit')
        with c3:
            when=st.date_input('Ngày khám cần đối chiếu',value=date.today(),key='visit_day')
        text=st.text_area('Triệu chứng lâm sàng',height=120,max_chars=1500,key='symptoms',
                          placeholder='Mô tả biểu hiện, thời gian và diễn tiến. Không nhập tên bệnh hoặc mã ICD.')
        d1,d2,d3=st.columns([1,1,2])
        with d1:sex=st.selectbox('Giới tính',['Không rõ','Nam','Nữ','Khác'],key='sex_input')
        with d2:age_text=st.text_input('Tuổi',key='age_text',placeholder='Ví dụ: 45')
        with d3:occupation=st.text_input('Nghề nghiệp (nếu có)',key='occupation')
        submitted=st.form_submit_button('Phân tích mã bệnh',type='primary',use_container_width=True)
    if submitted:
        cleaned=clean_text(text)
        warning=''
        if len(cleaned)<5:
            warning='Vui lòng mô tả triệu chứng rõ hơn.'
        elif len(cleaned)>1500:
            warning='Mô tả vượt quá 1.500 ký tự.'
        elif ICD_IN_TEXT.search(cleaned) or any(
            re.search(r'(?<!\w)'+re.escape(term)+r'(?!\w)',cleaned)
            for term in meta.get('leak_terms',[]) if term):
            warning='Vui lòng chỉ mô tả triệu chứng, không nhập tên bệnh hay mã ICD.'
        age=None
        if age_text.strip() and not warning:
            try:
                age=int(age_text)
                if not 0<=age<=110:raise ValueError
            except ValueError:
                warning='Tuổi phải là số nguyên từ 0 đến 110.'
        if warning:
            st.warning(warning)
            st.session_state.pop('last_result',None)
        else:
            params={'symptoms':cleaned,'sex':sex_code(sex),'age_band':age_band(age),
                    'occupation':job_code(occupation) if meta.get('has_occupation_data') else 'Không rõ'}
            try:
                pred,matched,msg=get_prediction(chosen,params,labels,rules)
            except Exception as exc:
                st.error(f'Lỗi mô hình: {type(exc).__name__}: {exc}')
                pred=None;matched=[];msg=''
            if msg:st.warning(msg)
            if pred:
                # Gắn kết kết quả với một lượt nhập để không giữ nhầm hoạt chất.
                st.session_state['last_result']={'code':pred,'matched':matched,'model':chosen,
                                                  'name':meta.get('names',{}).get(pred,'')}
                st.session_state['selected_meds']=[]
                st.session_state['drug_page']=1
                st.session_state.pop('selected_search',None)
                st.session_state['search_epoch']=st.session_state.get('search_epoch',0)+1
                st.session_state['med_epoch']=st.session_state.get('med_epoch',0)+1
                st.session_state['encounter_epoch']=st.session_state.get('encounter_epoch',0)+1
            else:
                st.session_state.pop('last_result',None)
    res=st.session_state.get('last_result')
    if not res:
        st.info('Nhập thông tin ở trên và nhấn “Phân tích mã bệnh” để xem kết quả.')
        return
    st.markdown('### 2. Kết quả phân loại')
    x,y=st.columns([1,3],vertical_alignment='center')
    with x: st.metric('Nhóm ICD',res['code'])
    with y:
        st.markdown('**'+res['name']+'**')
    show_visit_status(patient_id,res['code'],when,declared)
    if res['matched']:
        with st.expander('Xem luật triệu chứng khớp với dữ liệu huấn luyện'):
            st.dataframe(pd.DataFrame([{'Cụm triệu chứng':r['phrase'],'ICD':r['icd'],
                                        'Support':r['support'],'Confidence (%)':round(r['confidence']*100,1),
                                        'Lift':r['lift']} for r in res['matched']]),
                         hide_index=True,use_container_width=True)
    st.markdown('### 3. Hoạt chất và phiếu in')
    show_drugs(catalog,res['code'],res['name'],patient_id,when,declared)


if __name__=='__main__':
    main()
