"""Produce comparison tables without copying artifacts off the server."""
import argparse,csv,json
from pathlib import Path
from datetime import datetime,timezone
from paths import ROOT,RESULTS,DATA
p=argparse.ArgumentParser(); p.add_argument('--directory',default=str(RESULTS/'full')); a=p.parse_args()
folder=Path(a.directory).resolve()
if not folder.is_relative_to(DATA.resolve()): raise ValueError('Keep sampling results inside /data/RBF-Solver')
rows=[]; configs=[]
for nfe in [5,6,8,10]:
 for method in ['dpmpp','unipc']:
  path=folder/f'{method}_nfe{nfe}/result.json'
  row=dict(method=method,nfe=nfe,status='pending',n='',fid='',seconds='',images_per_second='')
  if path.exists():
   d=json.loads(path.read_text()); configs.append(d['config'])
   row.update(status='complete',n=d['config']['n'],fid=d['fid'],seconds=d['total_seconds'],images_per_second=d['images_per_second'])
  rows.append(row)
with (folder/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
lines=['# SiT-S/2 샘플러 비교','',f'갱신: {datetime.now(timezone.utc).isoformat()}','',
       'ImageNet 256×256, 조건당 50,000장, order 3, CFG 1.5, FP32, 동일 시드 20261007.',
       '원 논문의 NFE 구간을 SiT에 적용한 비교이며 Guided Diffusion의 논문 수치 재현은 아닙니다.','',
       '| NFE | DPM-Solver++ FID | UniPC FID | DPM-Solver++ 장/초 | UniPC 장/초 |','| --- | ---: | ---: | ---: | ---: |']
for nfe in [5,6,8,10]:
 pair=[next(r for r in rows if r['nfe']==nfe and r['method']==m) for m in ['dpmpp','unipc']]
 fid=[f"{r['fid']:.4f}" if r['status']=='complete' else '진행 중' for r in pair]
 rate=[f"{r['images_per_second']:.2f}" if r['status']=='complete' else '—' for r in pair]
 lines.append(f'| {nfe} | {fid[0]} | {fid[1]} | {rate[0]} | {rate[1]} |')
lines+=['','FID는 낮을수록 좋습니다. 완료된 조건만 수치를 표시합니다.',
        '처리량에는 샘플링·VAE·GPU Inception·주기적 체크포인트가 포함되며 최초 모델 로딩과 최종 고유값 계산은 제외됩니다.',
        '각 샘플러가 다른 GPU에서 실행되어 처리량에는 GPU 상태의 차이도 포함됩니다.','',
        '- 가중치: /data/RBF-Solver/weights/SiT-diffusers/SiT-S-2-256',
        '- 출처: https://huggingface.co/BiliSakura/SiT-diffusers (외부 변환 공개본; 원래 학습 설정 미기재)',
        '- GPU FID: torch-fidelity Inception FP32, 누적 통계·공분산 계산 FP64',
        '- ADM 특징 호환성: 32장 기준 최대 절대 오차 1.62e-5, 상대 L2 오차 2.83e-6',
        '- 대시보드: http://localhost:8765/','',
        '각 조건 디렉터리의 result.json에는 설정·가중치 해시·참조 통계 해시·실제 NFE가 기록됩니다.']
(folder/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(rows,indent=2))
