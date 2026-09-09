# `sweep_config.py` 설명

- **작성일** : 2026-07-17
- **목적** : 실제 백엔드·프런트를 낀 E2E 측정용. `max_wait_ms` / `dim_reduction.enabled` 조합마다 `config.yaml`을 자동으로 바꾸고 진짜 `main.py`(uvicorn) 서버를 재시작해주는 러너 — `sweep_batch.py`(로컬 시뮬레이션)와 달리 실제 RabbitMQ·백엔드·프런트가 전부 개입하는 측정에 씀.

---

## 왜 필요한가

- `max_wait_ms`/`dim_reduction.enabled`는 CLI 인자가 아니라 `config.yaml` 값이고, `main.py`는 `lifespan()`에서 **기동 시점에 한 번만** 이 값을 읽음 — 즉 값을 바꾸려면 파일을 고치고 **서버를 재시작**해야 함.
- 조합이 여러 개(기본 4×2=8개)라 수동으로 매번 `config.yaml`을 열어 고치고, 서버 껐다 켜고, 타이밍 맞춰 프런트 재생하는 걸 반복하면 실수(설정 안 바꾸고 측정, 재시작 깜빡함 등)가 나오기 쉬움 — 이 스크립트가 그 반복을 대신함.

## 무엇을 하는가

1. `config.yaml`을 백업(`config.yaml.bak`)
2. 각 조합마다:
   - `config.yaml`의 `enabled:`(dim_reduction), `max_wait_ms:` 두 줄만 정규식이 아니라 **라인 단위로 안전하게** 치환 (주석·나머지 설정은 그대로 유지 — YAML 파서로 통째로 dump하면 주석이 날아가서 일부러 텍스트 치환 방식 사용)
   - `uvicorn main:app`을 서브프로세스로 기동, ai-server 콘솔 출력은 `research/results/config_sweep_logs/comboNN_dim{on|off}_waitMS.log`로 저장
   - `/health` 엔드포인트를 폴링해서 서버가 실제로 뜰 때까지 대기 (모델 로딩 시간이 매번 다를 수 있어 고정 sleep 대신 헬스체크로 확인)
   - 서버가 준비되면 `--duration_s`(기본 60초) 동안 대기하면서 콘솔에 남은 시간을 출력 — **이 구간 동안 사용자가 직접 프런트에서 영상을 재생**해야 함
   - 구간이 끝나면 서버를 정상 종료(SIGTERM, 15초 내 미종료 시 강제 kill)
3. 모든 조합이 끝나거나 중간에 Ctrl+C로 중단되어도 `finally`에서 **`config.yaml`을 원래 상태로 복원**

## 사람이 직접 해야 하는 부분 (자동화 범위 밖)

- **프런트에서 영상 재생**: 스크립트가 "지금부터 N초간 재생하세요"라고 콘솔에 안내하는 시점에 맞춰 직접 재생을 시작해야 함. 조합마다 **같은 영상(같은 파일/같은 구간)**을 트는 게 공정한 비교에 중요.
- **백엔드 E2E 로그 수집**: `T2-T1`~`T5-T4` 로그는 백엔드 콘솔에만 찍힘 (백엔드 코드를 못 건드려서 자동 수집 불가). 각 조합 구간 동안 나온 로그를 조합 이름(`combo01_dimon_wait10` 등, 스크립트가 출력하는 label과 동일하게)별로 따로 복사해두면, 이후 로그 파일을 공유해줄 시 파싱/집계 스크립트를 만들어드릴 수 있음.

## 실행 방법

```bash
cd /home/lookie14/Intelligent-CCTV-ReID/ai-server
python research/deploy_sweep/sweep_config.py --max_wait_ms 10 30 50 100 --dim_reduction both --duration_s 60
```

- `--duration_s`: 조합당 측정 구간 길이(초). 프런트 영상 재생 시간에 맞춰 조정.
- `--port`: 기본 8000, 이미 다른 프로세스가 쓰고 있으면 변경.
- `--python`: 기본 `/home/lookie14/Intelligent-CCTV-ReID/venv/bin/python` (torch/ultralytics 설치된 venv).

## 주의할 점

- **이 스크립트는 `config.yaml`의 `rabbitmq` 섹션(큐 이름 등)을 건드리지 않음** — 지금 설정된 큐가 프로덕션 큐라면, 조합마다 서버를 껐다 켜는 동안 실제 트래픽/DB에 영향을 줄 수 있음. 실행 전에 지금 이 큐가 실제로 다른 곳에서 쓰이고 있지 않은지 확인 필요. (필요하면 테스트 큐 분리 설정도 별도로 도와드릴 수 있음.)
- 조합 사이 서버 재시작 시 RabbitMQ 컨슈머 스레드가 daemon 스레드라 완전히 graceful하게 끊기지 않을 수 있음 — 재시작마다 3초 대기를 넣어뒀지만, CloudAMQP 쪽에 죽은 연결이 잠깐 남아있을 수 있음(heartbeat로 곧 정리됨).
- 스크립트 실행 중 강제로 터미널을 닫는 등 `finally`가 실행되지 못하는 상황이 생기면 `config.yaml.bak`이 남아있을 수 있음 — 이 경우 `config.yaml.bak` 내용으로 `config.yaml`을 수동 복원.
