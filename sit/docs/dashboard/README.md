# 대시보드 문서용 스크린샷

사용 설명은 [대시보드 가이드](../../DASHBOARD_GUIDE.md)에 있다.
이 디렉터리는 그 문서에 사용한 실제 화면, 캡처 조건, 재캡처 도구를 보관한다.
문서와 PNG를 읽는 데 추가 설치는 필요하지 않다.

## 파일과 기준

| 파일 | 역할 |
|---|---|
| [capture.py](capture.py) | 기존 SiT·1-RF 대시보드를 별도 headless 브라우저로 조회 |
| [screenshots.json](screenshots.json) | 이미지별 시각(UTC)·선택 조건·공개용 가림·SHA-256 |
| [screenshots/](screenshots/) | 가이드에 삽입한 11개 PNG |
| [requirements.txt](requirements.txt) | 캡처 전용 Playwright 버전. 실험 환경 의존성과 별개 |

최초 캡처는 코드 `b1a22204c5cb4a756dc3aa3e2daa5870ace72ad8`과 실제 완료 기록을 사용했다.
Playwright 1.63.0·Chromium Headless Shell 153.0.8010.12(build v1243),
1600×1100 viewport·배율 1·한국어 locale·UTC 시간대를 사용했다.
패널 캡처는 패널 크기에 따라 PNG 치수가 다르다.

이미지는 당시의 기록이다. 다른 시점·자료·폰트로 재캡처하면 숫자와 배치, 파일 해시가 달라질 수 있다.
원래의 가이드 숫자와 동일한 화면을 재현하려면 [인수인계 자산](../../HANDOFF.md)을 복원해야 한다.

## 재캡처

실험 데이터가 있는 **대상 서버의 저장소 최상위**에서 수행한다.
가중치·결과를 문서 작성용 로컬 장비로 복사할 필요는 없다.
두 대시보드는 인수인계 문서에 따라 먼저 실행한다. 이 도구는 조회 서버도, 학습·샘플링도 시작하지 않는다.

다음 설치는 캡처 의존성과 브라우저를 Git 제외 경로 `sit/.cache/dashboard_docs`에 둔다.
`python3`는 대상 환경에서 사용할 Python이다.
실험용 의존성 잠금 파일을 변경하거나 Playwright를 실험의 필수 의존성으로 추가하지 않는다.

```bash
mkdir -p sit/.cache/dashboard_docs/python sit/.cache/dashboard_docs/browsers sit/.cache/tmp
export PIP_CACHE_DIR="$PWD/sit/.cache/dashboard_docs/pip"
export TMPDIR="$PWD/sit/.cache/tmp"
export PYTHONPATH="$PWD/sit/.cache/dashboard_docs/python"
export PLAYWRIGHT_BROWSERS_PATH="$PWD/sit/.cache/dashboard_docs/browsers"
export PYTHONDONTWRITEBYTECODE=1

python3 -m pip install --target "$PYTHONPATH" -r sit/docs/dashboard/requirements.txt
python3 -m playwright install chromium --only-shell
python3 sit/docs/dashboard/capture.py \
  --sit-url http://127.0.0.1:8765 \
  --rf-url http://127.0.0.1:8766
```

포트와 주소는 대상 환경에 맞게 지정한다.
Chromium의 운영체제 라이브러리와 한글 폰트도 필요하다. 최초 캡처에서는 Noto Sans CJK를 사용했다.
운영체제 패키지는 이 스크립트가 자동으로 설치하지 않는다.

스크립트는 본 저장소의 화면 구조와 현재 조건(CFG 0, NFE 6 등)을 대상으로 한다.
다른 대시보드에 그대로 적용하는 범용 캡처기가 아니다.
해당 조건의 API 자료·미리보기·계수 그림이 있어야 가이드와 같은 화면을 얻는다.

## 무엇을 변경하는가

- 기존 사용자 창·탭 대신 별도 headless 브라우저를 사용한다.
- 화면이 데이터를 받은 뒤 **자신의 브라우저 안에서만** 자동 조회 간격을 멈춰 캡처 중 화면 변화를 줄인다. 서버와 사용자의 대시보드는 계속 동작한다.
- CFG·NFE·메뉴를 선택하고, 실제 이미지 확대창과 상세 표를 연다.
- GPU 캡처에서 장치명·VRAM 수치·측정 용량 범위만 해당 브라우저 DOM에서 가린다. API 원본·실험 결과·대시보드 소스는 수정하지 않는다.
- `screenshots/*.png`와 `screenshots.json`을 덮어쓴다. 검토용 JPEG는 Git 제외 경로 `sit/.cache/dashboard_docs`에 저장한다.
- 브라우저 JavaScript 오류가 발생하면 실패로 종료한다. 이미지 로딩 실패까지 성공으로 보장하는 검증기는 아니므로 결과는 반드시 눈으로 확인한다.

기존 문서용 PNG를 보존해야 한다면 재실행 전에 Git 상태와 변경분을 확인한다.
실제 실행 데이터·가중치·학습 계수·샘플링 산출물을 이 디렉터리에 저장하지 않는다.
현재 도구는 신뢰하는 대시보드를 대상으로 Chromium sandbox를 끄고 실행하므로, 관련 없는 외부 페이지 주소를 넣지 않는다.

## 게시 전 확인

1. 모든 PNG에서 한글·범례·선택 조건·확대창·표를 확인한다.
2. 숫자는 같은 시점의 저장 기록과 대조한다. 완료·진행·오류 상태를 연출해 실제 캡처처럼 제시하지 않는다.
3. 주소·개인 경로·장치 정보가 새 화면에 노출되는지 확인한다. 현재 가림은 GPU 영역의 지정 항목만 다룬다.
4. 이미지 선택 조건·시각·가림을 `screenshots.json`과 가이드에 반영한다.
5. 가이드 링크와 이미지 해시를 확인한 뒤 문서·PNG·메타데이터만 함께 커밋한다.

이 캡처는 완료 화면과 조회 조작을 검토하기 위한 것이다.
학습·샘플링·FID 재평가·진행 중 장애 시험을 대신하지 않는다.
