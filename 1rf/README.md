# 1-RF 무조건부 FM 실험

**다른 환경으로 이전:** [공통 설치·자산 복원·실행 인수인계](../sit/HANDOFF.md).
현재 11 NFE × 3방법 = 33조건, 조건당 50,000장 평가가 완료됐다.


| 요구 기능 | 기존 실행 경로 | 재사용 및 필수 변경 | 검증 |
|---|---|---|---|
| DPM++·UniPC 비교 | launch → benchmark.Worker | 공용 큐·저장·재개·최대 배치 탐색 유지, 모델/입력/조건 연결 변경 | 동일 노이즈·실측 NFE·재개 식별자 |
| Target 생성 | rbf_pipeline.StageWorker → TrainingWorker.latent | GPU별 ID 분할과 병합 유지, 3채널 RF 및 BH1 연결 | 정확히 128개 ID·중복/누락·해시 |
| 계수 학습 | StageWorker → fit_condition → FMShapeTrainer → 원본 sample_with_optim | 전체 128쌍 1회로 설정, CIFAR 원본 루프 유지, FM 적분·시간 연결만 적용 | 부모 함수 동일성·저장/복원·실제 NFE |
| RBF 샘플링 | SamplingWorker → Worker.run_job | 학습 계수 필수 로딩·GPU FID·동일 입력 유지 | 계수·입력·모델·설정 식별자 |
| 대시보드 | dashboard API → 기존 HTML/CSS/JS | 비교·3단계·GPU·설정·계수 화면 보존, 무조건부/CIFAR 표기와 조건 변경 | 세 방법 모두 표시·11 NFE·단계별 실제 상태 |
| 평가 | fid_gpu → Worker.postprocess | 기존 GPU 특징·FP64 통계 재사용, CIFAR 참조와 공식 픽셀 변환 연결 | TF 기준 특징 대조·샘플 수·공분산 |

코드는 이 디렉터리 안에서만 변경한다. 가중치는 /data/RBF-Solver/weights/1rf,
산출물은 /data/RBF-Solver/results/1rf 아래에 저장한다.
공식 모델 소스는 vendor/RectifiedFlow에 고정한다. 기존 SiT 코드와 결과는 수정하지 않는다.
사용자가 시간 설정까지 프로젝트 CIFAR-10 방식으로 변경하도록 승인했다. 아래 조건을 적용하고 연결 검증 후 실행한다.

## 승인된 시간·좌표 조건

- 사용자 최종 지시: 시간 설정까지 프로젝트 CIFAR-10 방식으로 변경.
- 원본 score_sde 실행 경로: linear VP(beta_0=0.1, beta_1=20), t_start=1, log-SNR 균일 간격, NFE≤10은 t_end=1e-3, 그 외와 UniPC-200 target은 1e-4. UniPC BH1·3차·multistep·마지막 저차·추가 denoise 없음.
- 시간 그리드는 원본 DPM get_time_steps가 생성한 CPU FP32 배열을 사용하며 모든 샘플러에 공유한다.
- RF 연결: K=alpha+sigma, s=alpha/K, Y_RF=X_VP/K. 모델 입력은 Y_RF와 999*s이며 data prediction은 Y_RF+(1-s)*v이다.
- 모든 방법의 초기값은 같은 VP Gaussian Z다. DPM++·UniPC는 원본 VP 솔버를 그대로 사용한다. FM-RBF는 Y_start=Z/K_start에서 물리 RF 시간 격자를 따라 적분하고 최종값에 K_end를 곱해 VP 출력 단위로 맞춘다.
- RBF target은 UniPC-200의 VP 출력이다. 원본 target matching을 RF 좌표로 옮기면 s*target_VP+(1-s)*Z다. 상속한 루프 입력이 Z/K_start이므로 target용 noise prefactor에 K_start를 곱한다. 최적화 루프는 원본 그대로 유지한다.
- 위 RF 시간 수치는 좌표 변환의 계산 결과이며 논문에 명시된 RF 범위가 아니다. SiT의 s=0→0.999 설정 및 임의의 추가 절단값은 적용하지 않는다.

## 준비 검증 상태

공식 EMA의 CPU 로딩과 GPU 유한 출력, GPU 특징과 TF 기준 특징의 일치,
참조 5만 pool_3 특징의 GPU 통계 준비, 기존 대시보드 API의 33조건 포함을 확인했다.
승인된 VP/RF 좌표 연결의 36개 합성 조건, 원본 최적화·저장·복원·재생, GPU 0 DPM++(NFE 5·40), GPU 1 UniPC BH1(NFE 5·200) 검증을 통과했다. 다른 장치로 이전할 때는 전체 작업 조건에서 용량을 별도로 검증한다. GPU 번호만 같은 과거 배치를 재사용하지 않는다. 검증 산출물은 실험 결과에 집계하지 않는다.

## 실행 순서

사용자 지시에 따라 NFE 오름차순으로 RBF·DPM-Solver++·UniPC를 함께 배정한다. 완료 결과를 제외한 조건을 NFE 오름차순으로 두 GPU에 배정한다. 낮은 NFE가 실행 중이면 다음 미배정 조건을 다른 GPU에 배정하며, 특정 방법을 우선하지 않는다. 기존 launch의 큐·GPU 작업자·잠금·저장·재개·평가를 재사용하고 배정 범위만 변경한다. 수치 설정과 입력·계수·완료 결과는 유지한다. 전환 전에 이미 배정된 작업은 종료하지 않고 완료시킨다. 검증은 큐 순서·완료 결과 재사용·실제 작업자와 진행 기록으로 수행한다.

배치 탐색은 NFE마다 반복하지 않는다. 동일 실행 조건의 기존 실측 배치를 재사용하고 첫 실제 배치부터 결과에 집계한다. 모델·솔버·이미지 변환·GPU FID·학습 계산은 변경하지 않는다.
