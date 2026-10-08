# SiT-S/2 FM 실험

**화면 사용법:** [스크린샷으로 보는 대시보드 역할·기능 안내](DASHBOARD_GUIDE.md).

**다른 환경으로 이전:** [설치·자산 복원·실행·검증 인수인계](HANDOFF.md).
현재 완료: 양의 CFG 48조건 + CFG 0.0 33조건. 아래 준비 기록은 구현의 배경이며 현재 완료 상태와 구분한다.


기준: RBF-Solver 논문 arXiv:2603.13330 및 저장소의 `guided-diffusion` ImageNet256 실행 경로. 논문의 Guided Diffusion FID 재현이 아니라 사용자가 지정한 SiT-S/2 추가 실험이다.

## CFG 0.0 무조건부 추가 실험

unconditional_sampling.json과 unconditional_training.json은 기존 조건을 유지하면서 CFG 0.0을 추가한다. 세 샘플러(DPM-Solver++, UniPC, RBF), NFE 5/6/8/10, 2차, 조건당 10,000장, seed 1234다. 조건부·무조건부를 함께 학습한 SiT 체크포인트의 null-label branch를 평가하는 추가 실험이며, unconditional-only로 별도 학습한 모델이라는 뜻은 아니다.

CFG 0.0은 label 1000을 넣어 모델을 한 번 호출하고 모든 4개 latent 채널을 사용한다. 기존 3채널 forward_with_cfg에 숫자 0만 넣으면 남은 채널은 조건부이므로 그 경로를 사용하지 않는다. 기존 CFG 1.5/3.5/5.5/7.5의 계산은 보존한다. 결과에는 conditioning=unconditional_null_label, cfg_channels=0(guidance 적용 채널 수), velocity_channels=4, 실제 NFE와 모델 처리 이미지 수를 기록한다.

기존 rbf_followup.py, StageWorker, FollowupSamplingWorker, benchmark.Worker를 재사용한다. 128 target 쌍을 생성한 뒤 낮은 NFE부터 20회 계수 최적화·평균·계수 그래프·세 방법 샘플링을 수행한다. 해당 NFE의 비교가 완료되면 다음 NFE로 넘어간다. 시드별 기존 초기 노이즈를 재사용하고 저장된 클래스 라벨은 CFG 0.0에서 무시한다.

샘플링 배치는 같은 GPU·모델·정밀도·차수·입력·FID 기준으로 실제 완료한 SiT 결과에서 재사용한다. 첫 실제 배치부터 결과에 포함하며, NFE별 탐색을 반복하지 않는다. 실제 OOM이 발생한 단계에서만 이전에 성공한 더 작은 배치로 조정한다. VAE/FID 청크도 성공 기록을 재사용한다.

최초 CFG 0.0 4개 NFE 실행: `SIT_SAMPLING_PLAN=unconditional_sampling.json SIT_TRAINING_PLAN=unconditional_training.json python rbf_followup.py --run`.
`run_unconditional.sh`는 추가 7개 NFE 실행용 별칭이다.
저장: /data/RBF-Solver/results/sit_unconditional/ 아래 training/, sampling/, validation.json, launcher.log.
기존 결과·manifest와 추가 실험의 식별자는 분리하고 대시보드에서 합쳐 표시한다. 순수 null 출력·클래스 라벨 불변성·기존 양수 CFG 회귀·실제 체크포인트 로딩을 검증했다. 검증은 FID 결과를 대신하지 않는다.

## 기존 CFG 실험의 설정 기록

실제 설정의 단일 기준은 `experiment.json`이다. CFG 1.5/3.5/5.5/7.5, DPM-Solver++/UniPC, NFE 5/6/8/10, 2차, 조건당 10,000장, 총 32조건·320,000장. 원본의 multistep/time_uniform/lower_order_final=True/denoise_to_zero=False와 seed 1234를 유지한다. UniPC는 원본 기본 bh2를 사용한다. 이후 RBF target 생성·계수학습·샘플링도 완료되어 기존 네 CFG의 세 방법 48개 조건은 모두 평가됐다. 아래는 최초 DPM-Solver++·UniPC 실행 설정과 준비 절차의 기록이다.

최초 조건부 실행은 조건별 배치 탐색을 사용했던 과거 실행이다. 현재 운용에서 이를 반복하지 않는다. 새 장치는 명시적 1회 용량 검증 후 측정값을 재사용한다. 측정 중 모델·VAE·Inception·참조 통계·FID 누적기가 모두 GPU에 상주한다. VAE와 FID는 메모리에 맞게 GPU 청크로 처리한다. 튜닝 이미지와 실제 10,000장 평가는 분리한다.

두 GPU·서로 다른 배치에서도 같은 샘플 ID에는 같은 입력을 쓴다. seed 1234의 CPU torch.Generator로 생성한 10,000개 초기 잠재 노이즈와 1,000개 클래스 중 무작위 라벨을 `inputs.pt`에 한 번 저장하고 모든 조건에서 재사용한다. 입력 파일 해시를 각 결과에 기록한다. 부동소수점 연산의 배치별 반올림 차이까지 비트 단위 동일성을 주장하지 않는다.

## 시간 구간과 SiT 연결

원본 샘플러의 시작점 `schedule.T=1` 및 종료점 `1/total_N=0.001`을 사용한다. `dpm_solver.py:1245-1248`의 연속 시간 모델·낮은 NFE 권고도 종료점 1e-3이다. SiT 모델 시간은 `s=1-t`이므로 모델 경로는 0→0.999이다. 시작점을 0.999로 자르는 기존 설정은 사용하지 않는다.

SiT `transport/path.py`의 ICPlan은 `x=s*data+(1-s)*noise`, `velocity=data-noise`다. 따라서 data prediction은 `x0=x+(1-s)*velocity=x+t*velocity`다. 이는 원본의 noise→data 변환과 대수적으로 같고 시작점에서 0/0을 만들지 않는다.

시작점에서 half-log-SNR은 -∞다. DPM-Solver++의 첫 업데이트와 다음 업데이트는 해당 극한에서 직접 계산 가능하다. UniPC bh2의 두 번째 업데이트에는 과거 비율 `r=(lambda_old-lambda_prev)/h→-∞`가 들어간다. 원본의 `R*rho=b`에서 `rho_old=(b1-b0)/(r-1)→0`, `rho_current→b0`이고 `(m_old-m_prev)/r→0`이다. `sampling.EndpointUniPC`는 이 경우만 정확한 극한식으로 계산한다. 다른 단계는 복사한 원본을 그대로 호출한다. 차수, 격자, 모델 호출 수 또는 추가 epsilon을 바꾸지 않는다.

`validate_endpoint.py`는 CPU 상수 벡터장 해, affine 벡터장의 유한성, 요청 NFE, 내부 구간에서 원본 noise adapter와의 일치, UniPC 계수 극한으로의 수렴, vendor 원본 해시 일치를 확인한다. 결과는 `/data/RBF-Solver/results/endpoint_validation.json`이다. 이는 모델 품질 또는 최적 종료점 검증을 뜻하지 않는다. 종전 `validate_sampler_adapter.py`의 외부 Diffusers 기본값 검사는 현재 구간의 근거로 사용하지 않는다.

## GPU FID

ADM 참조 통계 `/data/checkpoints/VIRTUAL_imagenet256_labeled.npz`를 읽고, 검증된 torch-fidelity Inception-v3-compat의 CUDA FP32 특징을 사용한다. 평균·공분산 누적과 FID 계산은 CUDA FP64다. 기존 ADM GraphDef 대조 검증은 `/data/RBF-Solver/results/fid_parity.json`에 있다. VAE 배율은 저장된 VAE 설정에서 읽는다. 모델은 FP32이며 TF32를 사용하지 않는다.

## 실행·관찰·재개

대시보드 주소: http://localhost:8765/

서버에서 `./run_comparison.sh`를 실행한다. 이 스크립트는 `launch.py`를 실행하며, 대시보드가 정확한 설정을 표시하는지 확인한 다음 GPU 작업자 두 개를 시작한다. 단일 launcher 잠금으로 중복 실행을 막는다. 각 배치마다 통계 체크포인트를 저장하며, 같은 계획·입력·모델·참조 통계에서만 재개한다.

이전 단일 CFG용 `summarize.py`와 `verify_results.py`는 현재 실행 경로에서 사용하지 않는다. 현재 launcher가 모든 조건 완료 후 `comparison.json`을 작성하고 입력 해시·샘플 수·차수·실제 NFE를 확인한다.

## 위치와 산출물

- 코드: `./`
- 모델·가중치 캐시: `/data/RBF-Solver/weights/`
- 현재 실행: `/data/RBF-Solver/results/cfg_sweep/`
- 조건별: `status.json`, `batch_gpu*.json`, `progress.pt`, `preview.png`, `statistics.npz`, `result.json`
- 전체: `inputs.pt`, `manifest.json`, `queue.json`, `gpu*.json`, `gpu*.log`, `run_state.json`, 완료 후 `comparison.json`

미리보기와 FID 통계·재개 상태를 저장하며 전체 생성 이미지는 GPU FID로 처리 후 메모리에서 해제한다. 튜닝 시간은 조건별 생성/평가 처리 시간에 포함하지 않는다. 결과와 모델은 코드 디렉터리나 로컬 컴퓨터에 저장하지 않는다.

가중치는 `BiliSakura/SiT-diffusers`의 외부 변환본이며 원 학습 절차는 문서화되지 않았다. 가중치 SHA256: `3b0754c57c2b6e2e4e74b181d1730a8bd824a30eafeaae9eaf8bc4015e8e4f39`. RBF-Solver 기준 커밋: `af306c69f016ed3ef5a72c96bbb08af285c79315`. 세 vendor sampler 파일은 원본과 바이트 단위로 동일하며, 실행 manifest에 SHA256을 기록한다.

## RBF FM 계수 학습 준비

설정은 `./rbf_training.json`이다. CFG 1.5/3.5/5.5/7.5, NFE 5/6/8/10, 이력 2, CFG당 target 128쌍, UniPC-200 teacher, 16쌍씩 복원추출하여 20회 최적화하고 log shape를 평균한다. SiT 시간은 0→0.999, 시드는 1234다. SiT 가중치는 고정한다.

`rbf_training_run.py`는 `FMShapeTrainer`를 통해 원본 `RBFSolver.sample_with_optim()`을 그대로 호출한다. FM 계수·상태 갱신·target 경로와 후보 손실의 Adams 분기 일치는 연결부에서 처리한다. teacher는 기존 `sampling.sample()`, 최대 배치 탐색과 공용 대기열은 기존 `benchmark.Worker`의 절차를 재사용한다. 반복 결과 전체의 설정·입력·코드·파일 해시를 확인한 뒤 평균하며, 계수 파일 누락이나 손상은 기본값으로 대체하지 않는다. `load_trained_solver()`가 저장된 log shape를 FM 샘플러에 연결한다.

준비 검사는 `python rbf_training_run.py --check`로 수행한다. 인자 없이 실행해도 검사만 한다. 실제 target 생성과 학습은 명시적인 `--run`에서만 시작한다. 대시보드에 같은 설정이 표시되는지 확인하고, 기존 DPM·UniPC 실행 또는 다른 GPU 계산이 진행 중이면 시작하지 않는다. GPU 0·1이 CFG 작업을 나누며, teacher 샘플링 배치는 요청한 128쌍 범위에서 실제 최대값을 측정한다. 최적화 배치 16은 논문 조건이다.

준비 검증 기록: `/data/RBF-Solver/results/rbf_preparation_validation.json`. CPU 테스트 22개와 실제 대시보드 API 연결을 확인했다. 원본 전체 후보 격자의 합성 입력 검증과 실행 연결·저장·복원 검증을 포함하며, 실제 SiT target 생성이나 GPU 학습을 수행했다는 뜻은 아니다. 결과 경로는 `/data/RBF-Solver/results/rbf_training/`이고, 이는 최초 준비 당시 기록이다. 현재 target·학습·샘플링은 완료됐으며, 정확한 상태는 HANDOFF.md와 결과 목록을 따른다.


### CFG 0.0 NFE extension

The user extended CFG 0.0 to NFE 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40 for DPM-Solver++, UniPC and RBF. The first four NFEs and their original manifests remain unchanged. The seven additional NFEs use unconditional_extended_sampling.json and unconditional_extended_training.json with the same existing rbf_followup.py execution path. All other numerical settings remain the approved SiT settings: order 2, 10,000 samples per condition, seed 1234, uniform physical time 0 to 0.999.

prepare_unconditional_extension.py validates and imports the existing 128 target pairs without teacher sampling; its receipt preserves original identities and hashes. The full 11-NFE seeded draw bank retains the exact original first four NFE draws and uses the remaining slice for new fits. Outputs remain under /data/RBF-Solver/results/sit_unconditional/nfe12_40. Start or resume with run_unconditional.sh (or run_unconditional_extended.sh); it completes fitting and all three samplers at each NFE before advancing, and reuses successful batches without per-NFE search. The dashboard combines the two CFG 0.0 runs as 33 unique conditions and preserves all positive-CFG results.


### CFG 0.0 RBF NFE 6: new-target retraining

The user requested a new 128-pair target set and retraining of NFE 6 only. The new run uses unconditional_retrain_training.json and unconditional_retrain_sampling.json through the existing run_unconditional_retrain.sh / rbf_followup.py pipeline. prepare_unconditional_retrain.py replays and verifies the original seed-1234 noise and label draws, then consumes the next block for new targets. The original NFE6 replacement indices and original 10,000-sample evaluation input bank are unchanged. Original results, targets and coefficient files remain intact; provenance and new outputs live under /data/RBF-Solver/results/sit_unconditional/nfe6_retrain.

The dashboard shows the requested latest NFE6 run with the previous FID alongside it and retains the old result in the detailed previous field. The original preview remains available with variant=original. This is one repeated run of the same comparison condition, not an additional sampler or an independent NFE. The latest result is displayed regardless of whether its FID improves.
