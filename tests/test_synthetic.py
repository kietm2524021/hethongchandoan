import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
LABELS=['B20','J02','K21','M54','H81','I10','M15','J06','E11','M47']
SYMPTOMS=['mệt mỏi kéo dài khó chịu trong người',
          'đau họng rát cổ nuốt khó nhiều',
          'ợ chua đau vùng thượng vị sau ăn',
          'đau lưng khi xoay người cúi xuống',
          'chóng mặt quay cuồng mất thăng bằng',
          'đau đầu hoa mắt choáng váng thường xuyên',
          'đau khớp gối cứng khớp khi đi lại',
          'ho sổ mũi nghẹt mũi hắt hơi',
          'khát nước nhiều tiểu đêm gầy sút',
          'đau vùng thắt lưng tê lan chân']


def make_fixture(folder:Path):
    folder.mkdir(parents=True,exist_ok=True)
    for month in range(1,13):
        rows=[]
        for i in range(60):
            li=i%10; j=i//10
            patient=f'BN-{li}-{(month*6+j)%20}'
            code=LABELS[li]
            symptoms=SYMPTOMS[li]+ (' về sáng' if j%2 else ' lúc chiều')
            for med in ('Hoạt chất A'+str(li),'Hoạt chất B'+str(li)):
                rows.append({'mabn':patient,'makh':f'{month}_{i}',
                             'maicd':code+'.9' if code in ('E11','K21') else code,
                             'motatrieuchung':symptoms, 'gioitinh':'Nam' if i%2 else 'Nữ',
                             'ngaysinh':'03/09/1981','ngayhd':f'{month:02d}/01/2024',
                             # deliberately tenhc BEFORE maicd in source, to prevent reordering bugs
                             'tenhc':med,'cachuong':'Uống theo chỉ định',
                             'kqcdoan':'Tên bệnh giả lập '+str(li)})
        pd.DataFrame(rows)[['mabn','makh','tenhc','cachuong','maicd',
                            'motatrieuchung','gioitinh','ngaysinh','ngayhd','kqcdoan']].to_csv(
                                folder/f'normalized_top10_thang{month}.csv',index=False,encoding='utf-8-sig')


class TestPipeline(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data';out=root/'artifacts'
            make_fixture(data)
            command=[sys.executable,str(BASE/'train.py'),'train','--data-dir',str(data),'--out',str(out),
                     '--rule-support','5','--min-med-visits','3','--min-regimen-visits','3','--min-way-visits','3']
            proc=subprocess.run(command,capture_output=True,text=True,timeout=180,cwd=str(BASE))
            self.assertEqual(proc.returncode,0,msg=proc.stdout[-3000:]+'\n'+proc.stderr[-3000:])
            info=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(len(info['labels']),10)
            self.assertFalse(info['has_occupation_data'])
            self.assertEqual(len(list((out/'models').glob('*.joblib'))),6)
            med=json.loads((out/'drug_catalog.json').read_text(encoding='utf-8'))
            self.assertGreater(len(med['icd_catalog']['E11']['hoat_chat']),0)
            self.assertGreater(len(med['icd_catalog']['E11']['to_hop_don_lich_su']),0)
            self.assertTrue(med['icd_catalog']['E11']['hoat_chat'][0]['cach_dung_lich_su'])
            stats=pd.read_csv(out/'so_sanh_mo_hinh.csv')
            self.assertEqual(stats['model'].nunique(),8)
            self.assertFalse(json.loads((out/'chia_du_lieu.json').read_text(encoding='utf-8'))['same_patient_across_splits'])
            from revisit_engine import patient_status
            import joblib
            key=(out/'.hmac_key').read_bytes()
            history=joblib.load(out/'patient_history_local.joblib')
            status=patient_status('BN-0-7','B20','2025-01-02',history,key)
            self.assertEqual(status['status'],'followup_same_icd')
            self.assertTrue(status['repeat_medication_observed'])


if __name__=='__main__':unittest.main()
