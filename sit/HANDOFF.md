# FM 실험 재현·이전·인수인계

이 저장소는 RBF-Solver의 기존 구현에 SiT-S/2와 비증류 1-Rectified Flow 연결부를 추가한 실험이다.
[논문](https://arxiv.org/abs/2603.13330)의 Guided Diffusion/score-SDE 수치를 그대로 재현했다는 뜻은 아니다.
**처음 인수한 사람은 2–5절의 복원·검증부터 수행한다. 대시보드 시작은 샘플링 시작이 아니다.**

화면별 역할·지표 해석·조작은 [실제 스크린샷을 포함한 대시보드 가이드](DASHBOARD_GUIDE.md)를 참고한다.

## 1. 인수인계 상태와 기준

2026-10-08 인수인계 검토에서 다음 완료 결과를 확인했다. 계산 작업은 종료됐으며 결과 열람 서비스만 실행 중이었다.
커밋 식별자는 `git rev-parse HEAD`로 기록한다. 이 문서와 코드, 아래 자산 목록을 같은 Git 커밋에서 가져온다.

| 실험 | CFG | NFE | 방법 | 조건당 표본 | 완료 |
|---|---|---|---|---:|---:|
| SiT-S/2, ImageNet 256 | 1.5·3.5·5.5·7.5 | 5·6·8·10 | DPM++·UniPC·RBF | 10,000 | 48/48 |
| SiT-S/2, null-label | 0.0 | 5·6·8·10·12·15·20·25·30·35·40 | 동일 | 10,000 | 33/33 |
| 1-RF, CIFAR-10 | 무조건부 | 동일한 11개 NFE | 동일 | 50,000 | 33/33 |

SiT 고유 조건은 81개다. CFG 0.0·RBF·NFE 6을 새 target으로 반복한 10,000장은 별도의 반복 실행이며 고유 조건 수에 더하지 않는다.
평가 규모는 SiT 정규 비교 810,000장 + 반복 10,000장, 1-RF 1,650,000장이다.
전체 수치는 [결과 CSV](handoff/results.csv), [결과 JSON](handoff/results.json)에 있다. 두 파일은 인수인계용 수치 요약이다.
원본의 입력·target·계수·통계·실행 기록은 외부 자산이다.

## 2. 무엇을 어디서 받는가

| 전달물 | 위치/방법 | 필요한 이유 |
|---|---|---|
| 실행 코드·설정·대시보드·테스트 | 현재 Git 저장소 | 전체 실행 구조 |
| SiT 공식 코드 | Git submodule, `cbde832a40b153ccc79603412409da9c9b0c568c` | 모델 구조·원전 |
| RectifiedFlow 공식 코드 | Git submodule, `5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6` | 비증류 1-RF·EMA 복원 |
| Python 의존성 | [requirements.lock](handoff/requirements.lock), [software.json](handoff/software.json) | 실제 관측 버전과 설치 목록 |
| 가중치·입력·target·계수·FID 통계·미리보기·기록 | 기존 운영자로부터 외부 데이터 디렉터리 이전 | 기존 실험의 정확한 자산 복원 |
| ImageNet/CIFAR FID 기준 파일 | 별도 reference 디렉터리 | 동일 평가 기준 |
| 파일별 크기·SHA-256 | [assets.json](handoff/assets.json) | 빠짐·손상·다른 파일 대체 탐지 |

원본 결과 전체는 Git에 들어 있지 않다. **Git clone만으로 기존 결과와 학습 자산까지 복원되지는 않는다.**
외부 자산 전달 권한과 위치는 운영자에게서 받아야 한다. 공개 체크포인트만 재다운로드해도 학습 target·계수·진행 기록은 복구되지 않는다.
토큰, SSH 키, 개인 서버 주소, 개인 계정 경로, 장치 기종은 공개 설치 조건에 포함하지 않는다.
자산 목록은 `data`/`reference` 기준 상대 경로만 가진다.

```bash
git clone --recurse-submodules https://github.com/scpark20/RBF-Solver.git
cd RBF-Solver
git submodule update --init --recursive
git submodule status
```

외부 데이터는 **새 대상 장비로 직접** 전송한다. 아래 변수는 수신 환경에서 지정하며 실제 서버 정보를 문서에 고정하지 않는다.

```bash
# SOURCE는 접근 권한이 있는 원본 호스트의 SSH 별칭이다.
# SOURCE_DATA, SOURCE_REFERENCE, DATA_HOST, REFERENCE_HOST는 각 환경의 절대 경로다.
rsync -aL --partial --exclude='results/handoff/' --exclude='.cache/' \
  "$SOURCE:$SOURCE_DATA/" "$DATA_HOST/"
rsync -aL --partial \
  "$SOURCE:$SOURCE_REFERENCE/VIRTUAL_imagenet256_labeled.npz" "$REFERENCE_HOST/"
rsync -aL --partial \
  "$SOURCE:$SOURCE_REFERENCE/cifar10_stats.npz" "$REFERENCE_HOST/"
python3 sit/handoff/assets.py verify \
  --data-root "$DATA_HOST" --reference-root "$REFERENCE_HOST" \
  --manifest sit/handoff/assets.json
```

검사 대상은 2,363파일, 13,519,404,157바이트다. 전송 시 대상과 여유 공간을 확인한다.
모든 생성 이미지를 저장한 것은 아니다. 이미지들은 GPU FID에 반영 후 해제되고 미리보기·통계·재개 상태가 저장됐다.
가중치 캐시도 포함하며, symlink를 전송할 때는 대상 파일까지 포함한다. 기존 결과를 재생성하는 명령으로 전송 누락을 메우지 않는다.

## 3. 환경 구성 — 호스트 경로와 내부 실행 경로

권장 이전 방식은 [Dockerfile](handoff/Dockerfile)과 [Compose](handoff/compose.yaml)다.
호스트의 코드·저장소 위치는 자유롭게 지정하되 내부 경로를 고정해 과거 JSON의 절대 경로 의미를 보존한다.

| 용도 | 호스트 | 컨테이너 내부 |
|---|---|---|
| Git checkout | `RBF_REPO_HOST` | `/opt/RBF-Solver` |
| 외부 가중치·실험 자산 | `RBF_DATA_HOST` | `/data/RBF-Solver` |
| 원본 FID 기준 파일 | `RBF_REFERENCE_HOST` | `/data/checkpoints`, 읽기 전용 |

요구 조건은 Linux, 지원되는 NVIDIA 드라이버와 NVIDIA Container Toolkit이다.
기록된 실행 환경은 Python 3.12.13, PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130이며 전체 의존성을 고정했다.
Docker의 Python 3.12 기본 이미지는 patch/digest를 고정하지 않았다. 빌드한 이미지의 digest와 실제 Python 버전도 새 실행 기록에 남긴다.
특정 GPU 제품명이나 VRAM 크기를 요구하지 않는다. 필요한 메모리와 처리량은 실제 환경에서 측정한다.

```bash
cp sit/handoff/environment.example sit/handoff/.env
# sit/handoff/.env의 세 경로, 호스트 GPU 목록과 로컬 포트를 대상 환경에 맞게 수정한다.
docker compose --env-file sit/handoff/.env -f sit/handoff/compose.yaml build
docker compose --env-file sit/handoff/.env -f sit/handoff/compose.yaml up -d
docker compose --env-file sit/handoff/.env -f sit/handoff/compose.yaml exec -d lab \
  python 1rf/dashboard.py --host 0.0.0.0 --port 8766
```

기본 Compose는 두 호스트 GPU를 노출하고 기존 설정은 보이는 GPU `[0,1]`을 사용한다.
다른 개수의 GPU를 사용하면 Compose의 `device_ids`와 신규 실행의 `gpus`를 함께 맞춘다.
호스트 GPU 번호와 컨테이너 내부에서 보이는 번호를 같다고 가정하지 않는다.
다음 명령으로 실제 장치 수와 번호를 확인한 뒤 배정한다.

```bash
docker compose --env-file sit/handoff/.env -f sit/handoff/compose.yaml exec lab \
  python -c "import torch; print(torch.__version__, torch.version.cuda); print([(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])"
```

동일한 내부 데이터 경로를 마련한 Linux의 native 실행도 가능하다. 가상환경을 활성화한 뒤 잠금 파일을 설치한다.
`run*.sh`는 `PYTHON_BIN` 또는 활성화된 `python`을 쓰며, 하위 Python 프로세스는 부모의 `sys.executable`을 사용한다.
서로 다른 GPU·드라이버·라이브러리에서 비트 단위 동일 결과를 보장하지 않는다.
동일한 모델·입력·수치 조건과 추적 가능한 평가 절차를 유지하는 것이 이전의 기준이다.

## 4. 복원 직후 확인

기본 서비스는 SiT 대시보드만 시작하며 학습·샘플링은 시작하지 않는다.
브라우저는 필요할 때 `http://localhost:8765/`와 `http://localhost:8766/`로 연다. 공개 인터넷 노출은 기본 설정에 없다.

```bash
docker compose --env-file sit/handoff/.env -f sit/handoff/compose.yaml exec lab \
  python sit/handoff/assets.py verify \
  --data-root /data/RBF-Solver --reference-root /data/checkpoints \
  --manifest sit/handoff/assets.json

docker compose --env-file sit/handoff/.env -f sit/handoff/compose.yaml exec -w /opt/RBF-Solver/sit lab \
  python -m unittest -q test_sampling test_rbf_solver_fm test_rbf_training test_unconditional test_capacity
docker compose --env-file sit/handoff/.env -f sit/handoff/compose.yaml exec -w /opt/RBF-Solver/1rf lab \
  python -m unittest -q test_capacity
```

대시보드 `/api/comparison`에서 SiT 81조건, 1-RF 33조건과 각 완료 상태를 확인한다.
SiT에서는 CFG 0.0·NFE 6의 최신 RBF 결과와 `previous` 원본을 둘 다 확인한다.
`/api/rbf-training`은 target·학습·샘플링을 분리해 보여주며, `/api/rbf-coefficients?cfg=0`은 실제 계수 자료다.
GPU 계산 프로세스가 없는 상태와 완료 상태는 양립한다. 서비스가 살아 있다는 이유로 샘플링 중이라고 보고하지 않는다.

## 5. 보존·재개·새 실행의 구분

원본 결과의 `source_sha256`, `protocol_sha256`, 입력·계수 SHA-256은 생성 당시 식별자다.
이번 공개 준비에서 Python 경로와 배치 재사용을 수정했으므로 현재 코드 hash는 과거와 다르다.
**과거 manifest를 현재 hash로 덮어쓰거나 검증을 비활성화해서 재개하지 않는다.**

- 완료 결과 열람: 복원한 원본과 현재 대시보드를 사용한다. 전체 실험 재실행 불필요.
- 중단된 과거 실행의 재개: 그 실행 manifest와 일치하는 코드·설정·자산을 먼저 확보한다.
- 현재 코드로 신규 실행: 별도 checkout과 새 결과 namespace를 사용한다. 원본과 섞지 않는다.

운영 서버의 `results/handoff/source_before_portability.tar.gz`에는 공개 준비 직전 코드와 과거 history가 별도 보존돼 있다.
이는 개인 경로를 포함할 수 있는 **비공개 보존 자료**이며 Git과 공개 자산 목록에서 제외했다.
파일 식별자는 [source_snapshot.json](handoff/source_snapshot.json)에 기록했다.
필요하면 권한이 있는 운영자로부터 별도 전달받고, 파일별 hash를 해당 실행 manifest와 대조한다.
이 snapshot 하나가 모든 과거 실행 버전과 일치한다고 가정하지 않는다. 현재 전달 대상 비교는 모두 완료 상태다.

신규 실행용 구성기는 수치 조건을 유지하고 결과 참조와 GPU 배정만 변경한다. 기존 결과에는 쓰지 않는다.
아래 예시는 새 checkout에서 수행하며 먼저 변경 내용을 출력한다.

```bash
python sit/handoff/new_run.py --project sit \
  --results /data/RBF-Solver/results/runs/reproduction-01 --gpus 0 1
# 검토한 같은 명령에 --apply를 붙이면 해당 checkout의 JSON 설정을 수정한다.
# 1-RF는 --project 1rf, --results /data/RBF-Solver/results/1rf/runs/reproduction-01
```

설정 수정 후 해당 checkout을 사용하는 대시보드를 재시작해 표시와 실제 계획을 일치시킨다.
검증 기준 파일과 원본 기록은 유지한다. 신규 target·학습 계수는 새 source/plan 식별자로 생성한다. 학습만 준비할 때는 선택한 training plan으로 `python rbf_training_run.py --run`을 사용한다. 이 명령은 target과 계수를 만들며 FID 샘플링은 시작하지 않는다.

## 6. 배치 용량과 실행 경계

과거 배치 값은 다른 하드웨어에서 유효하다는 증거가 아니다.
현재 샘플링은 장치·메모리·CUDA/PyTorch·allocator·plan에 연결된 명시적 capacity 기록을 요구한다.
이전 GPU 번호만 같은 기록, 환경 식별자가 없는 과거 cache는 거절한다.
용량 측정은 명시적으로 수행하며 **NFE가 바뀐다는 이유로 반복하지 않는다.**
실제 OOM은 저장된 더 작은 성공 후보로만 복구한다. 후보가 없으면 진행분을 보존하고 명시적 재측정을 요구한다.

`calibrate_capacity.py`는 기존 Worker의 전체 샘플링 경로와 VAE/FID 후처리를 사용한다.
방법·CFG·NFE는 승인된 plan 안에서 직접 지정한다. CFG 0과 양수 CFG는 서로 다른 workload다.
샘플링 batch와 논문의 최적화 표본 수를 혼동하지 않는다. SiT 학습 16쌍, 1-RF 학습 128쌍은 임의로 축소하지 않는다.
각 사용하는 GPU·계획·방법에 대해 측정하며, 초기 용량 검증 이후에는 모든 NFE가 같은 기록을 재사용한다.

```bash
# 프로젝트 디렉터리에서, 새 실행의 baseline 입력 은행 준비와 명시적 용량 측정:
python calibrate_capacity.py --kind baseline --gpu 0 --method dpmpp --cfg 1.5 --nfe 10
# 1-RF baseline은 --cfg 1, --nfe 40 등 해당 plan의 조건을 사용한다.
# --kind rbf는 rbf_sampling.json, SiT --kind followup은 SIT_*_PLAN 선택을 사용한다.
```

용량 검증에는 실모델·FID 자산·검증 기록이 필요하다. RBF 용량 검증은 먼저 학습된 해당 NFE 계수를 필요로 한다.
RBF Worker는 비교 입력 검증을 위해 기존 baseline NFE 5 결과도 요구한다.
새 실행에서 그 기준 결과가 아직 없으면, baseline NFE 5 명령에 **`--run-baseline`을 명시**해 한 조건을 실제 평가한다.
이 옵션은 준비만 하는 옵션이 아니며 조건당 전체 표본을 생성·평가한다. 이미 완료된 조건은 덮어쓰지 않는다.

## 7. 단계별 진입점과 의존성

명령은 각 프로젝트 디렉터리에서 활성화된 Python으로 실행한다. 먼저 같은 계획의 대시보드가 실행 중이어야 한다.
아래는 코드 진입점이며, 완료된 원본 디렉터리에 무조건 실행하는 명령 목록이 아니다.

| 목적 | SiT | 1-RF |
|---|---|---|
| 준비 확인, 계산 시작 없음 | `python rbf_training_run.py --check` | 동일 |
| target 생성 | `python rbf_pipeline.py --cfg 1.5 --stage target` | `--cfg 1 --stage target` |
| 계수 학습 | `python rbf_pipeline.py --cfg 1.5 --stage fit` | `--cfg 1 --stage fit` |
| RBF 샘플링 | `python rbf_pipeline.py --cfg 1.5 --stage sample` | `--cfg 1 --stage sample` |
| 조건부 baseline 비교 | `python launch.py` | 세 방법 통합도 아래 `launch.py` |
| 후속 CFG 3.5·5.5·7.5 | `python rbf_followup.py --run` | 해당 없음 |
| CFG 0.0 최초 4 NFE | 아래 환경 변수 선택 후 `rbf_followup.py --run` | 해당 없음 |
| CFG 0.0 추가 7 NFE | `prepare_unconditional_extension.py` 후 `run_unconditional_extended.sh` | 해당 없음 |
| CFG 0.0 NFE 6 새 target 반복 | `prepare_unconditional_retrain.py` 후 `run_unconditional_retrain.sh` | 해당 없음 |
| RBF 계수 그림 | `python plot_rbf_coefficients.py --cfg <cfg>` | `--cfg 1` |
| 1-RF 세 방법 비교 배정 | 해당 없음 | 학습·capacity 준비 후 `python launch.py` |

```bash
SIT_SAMPLING_PLAN=unconditional_sampling.json \
SIT_TRAINING_PLAN=unconditional_training.json \
python rbf_followup.py --run
```

SiT extension은 최초 4 NFE의 검증된 128 target을 가져오며 teacher를 다시 생성하지 않는다.
최초/추가 NFE의 난수 draw 순서를 보존한다. NFE 6 반복은 새 초기 noise의 다음 block으로 target을 다시 생성한다.
SiT 후속 실행은 해당 NFE의 학습·세 방법 비교를 마친 뒤 다음 NFE로 진행한다.
1-RF 통합 배정은 **미배정 조건을 NFE 오름차순**으로 빈 작업자에 넣는다. 실행 중인 낮은 NFE와 다음 NFE가 겹칠 수 있다.
방법별 우선순위를 임의로 추가하지 않는다. 1-RF `run_comparison.sh`는 target→fit→그림→통합 launch의 편의 진입점이다.
초기화가 없는 새 환경에서는 단계별 진입점으로 target·계수·baseline 기준 결과·capacity를 갖춘 후 통합 launch를 실행한다.

## 8. 수치 조건과 코드 대응

| 항목 | SiT | 1-RF |
|---|---|---|
| 모델 | SiT-S/2, ImageNet256 latent | 공식 CIFAR-10 1-RF EMA, 비증류 |
| 차수 / UniPC | 2 / BH2 | 3 / BH1 |
| Seed / precision | 1234 / FP32, TF32 off | 42 / FP32, TF32 off |
| 마지막 단계 / extra denoise | 저차 / 없음 | 동일 |
| 격자 | time_uniform | 원본 VP CPU FP32 log-SNR uniform |
| target / teacher | 128 / UniPC-200 | 128 / UniPC-200 |
| 학습 | 16쌍 복원추출 ×20, log γ 평균 | 128쌍 ×1 |
| log γ 후보 | [-2,2]의 33점, 원본 Adams 분기 | 동일한 원본 후보 절차 |
| 평가 | FP32 Inception, CUDA FP64 moments/FID | 동일 |

SiT `s=1-t`, 모델 시간 `s:0→0.999`, solver 시간 `t:1→0.001`.
data prediction은 `x+t*v`다. BH2의 무한 log-SNR 과거 비율은 `EndpointUniPC`에서 정확한 극한으로 처리한다.
CFG 0은 null label 1000을 모델에 직접 넣어 latent 네 채널 모두 사용한다.
양수 CFG는 기존 공식 3채널 guidance 경로다. CFG 0은 null branch 생성이며 unconditional-only로 학습한 체크포인트라는 뜻은 아니다.

1-RF는 linear VP β₀=0.1, β₁=20, t_start=1, NFE≤10의 t_end=0.001, 더 높은 NFE와 teacher의 t_end=0.0001이다.
K=α+σ, s=α/K, Y_RF=X_VP/K, 모델 입력 시간은 999s, data prediction은 Y_RF+(1−s)v.
RBF 초기값은 Z/K_start, 마지막 값은 K_end를 곱해 VP 출력 단위로 맞춘다.
target matching은 s·target_VP+(1−s)·Z이며 상속 루프의 noise prefactor는 K_start다.
이 변환은 RF 추가 실험의 연결이며 논문에 직접 적힌 RF 시간 범위가 아니다.
`1rf` 설정의 CFG 1은 무조건부 실행 식별값으로 SiT guidance scale과 의미가 다르다.

`rbf_shape_training.FMShapeTrainer`는 원본 `RBFSolver.sample_with_optim`을 상속해 호출한다.
변경은 FM 적분·상태 갱신·target 연결부에 한정한다. 후보 평가와 실제 갱신의 Adams 분기를 일치시킨다.
vendor sampler는 프로젝트의 ImageNet256/score_sde 원본 출처를 보존한다.

## 9. 자산 출처와 결과 위치

SiT checkpoint는 `BiliSakura/SiT-diffusers`의 revision `2f02393a3079dfa6210c10302cc927f59d2cb142` 변환본이다.
원 학습 절차는 문서화되지 않았다. 공식 학습 recipe 재현으로 설명하지 않는다.
`prepare_weights.py`는 이 revision과 model SHA-256을 고정한다.

- SiT checkpoint SHA-256: `3b0754c57c2b6e2e4e74b181d1730a8bd824a30eafeaae9eaf8bc4015e8e4f39`
- 1-RF checkpoint SHA-256: `27d1463f573556765d380b1983664d24e00a194853e0778f5af18fcdf34500de`
- FID: torch-fidelity 0.4.0 inception-v3-compat. ADM/TF 특징 대조는 `fid_parity.json`에 기록.
- CIFAR reference: 원본 50,000×2048 pool_3 특징을 `prepare_reference.py`에서 CUDA FP64 통계로 변환.

아래는 `/data/RBF-Solver/results`에 대한 상대 경로다.

| 자산 | 상대 경로 |
|---|---|
| SiT 조건부 baseline/입력 은행 | `cfg_sweep/` |
| SiT RBF CFG 1.5 / 나머지 CFG | `rbf_cfg1.5_sampling/` / `rbf_cfgs_sampling/` |
| SiT 양수 CFG 학습 | `rbf_training/` |
| SiT CFG 0 최초 4 NFE | `sit_unconditional/{training,sampling}/` |
| SiT CFG 0 추가 7 NFE | `sit_unconditional/nfe12_40/{training,sampling}/` |
| SiT CFG 0 NFE 6 새 target 반복 | `sit_unconditional/nfe6_retrain/{training,sampling}/` |
| 1-RF baseline / 학습 / RBF | `1rf/comparison/` / `1rf/rbf_training/` / `1rf/rbf_sampling/` |

조건별 `result.json`, `statistics.npz`, `progress.pt`, `preview.png`와 입력 은행·manifest를 함께 보존한다.
`queue.json`, `run_state.json`, 작업자 log/상태 파일은 운영 기록이다. 과거 PID를 새 호스트의 프로세스로 해석하지 않는다.
코드·가상환경은 Git checkout에, 가중치·캐시 자산·샘플링 결과는 외부 데이터 루트에 둔다.

## 10. NFE 6 재학습 결과의 해석

| SiT CFG 0.0, RBF NFE 6 | FID |
|---|---:|
| 원본 | 103.48957989114041 |
| 새 target 128개로 재학습·재평가 | 103.48928057712658 |
| 차이 | −0.0002993140138300987 |

seed 1234의 원본 noise·label draw를 재생·대조한 뒤 다음 block으로 새 target을 만들었다.
20×16 복원추출 인덱스와 10,000장 평가 입력은 원본 그대로다.
20회 모두 손실 기록은 달라졌지만 선택된 log γ 배열은 원본과 정확히 같았다.
작은 FID 차이를 계수 개선으로 해석하지 않는다. 실행 장치·batch가 달랐으며 차이의 원인을 하나로 입증한 것은 아니다.
`comparison_report.json`, `training/input_provenance.json`, `training/retraining_comparison.json`에 근거가 있다.
대시보드는 최신 요청 결과를 표시하고 원본을 `previous`로 보존한다. 더 좋은 값만 선택하는 규칙이 아니다.
1-RF NFE 6은 별도 실험이며 RBF 13.119507381, DPM++ 12.595664925, UniPC 10.427916168이다.

## 11. 오류·중단·검증 범위

| 증상 | 확인할 곳과 조치 |
|---|---|
| submodule 모듈을 찾지 못함 | `git submodule update --init --recursive`, 고정 SHA 확인 |
| 가중치/기준 파일 누락 | assets 검사와 mount 확인. 다른 checkpoint로 자동 대체하지 않음 |
| source/plan/shape hash 불일치 | 기존 실행 버전과 새 실행 구분. 기록 수정으로 우회하지 않음 |
| capacity 미존재/환경 불일치 | 해당 환경에서 명시적 calibration. 매 NFE 자동 탐색 금지 |
| 계수 파일 없음 | target→fit 완료와 해당 NFE 경로 확인. 기본 계수 fallback 없음 |
| GPU 메모리 부족 | 진척 보존, 검증된 작은 batch 사용. 학습 표본 수는 유지 |
| 대시보드는 있으나 계산 없음 | queue·run_state·작업자 실제 프로세스 확인. 완료/대기/중단 구분 |

중단은 해당 실행의 launcher·worker·재시작 경로를 대상으로 한다. 무관한 작업을 종료하지 않는다.
강제 중단 뒤에는 마지막 원자적 progress 저장까지 복원할 수 있다. 이후 재개는 식별자 일치를 먼저 확인한다.

이번 공개 준비에서 확인한 범위는 코드/쉘/YAML 구문, CPU 계산·회귀 테스트, 장치 capacity의 실패 분기,
실제 완료 결과와 두 대시보드의 비교 조건, 외부 자산 해시, Git 게시 파일의 개인정보/대용량 제외다.
**현재 서버에 Docker가 없어 이미지 빌드 및 다른 장비에서의 GPU 실행은 아직 검증하지 않았다.**
새 Git 복제본에서 두 submodule 복원과 SiT CPU 테스트 30개, 별도 1-RF capacity 테스트 4개를 확인했다. [검증 범위](handoff/validation.json)를 함께 보존한다.
CPU 테스트 통과를 새 GPU의 FID 재현 성공으로 확대하지 않는다. 실제 이전 환경에서 3–6절을 완료한 기록이 필요하다.
