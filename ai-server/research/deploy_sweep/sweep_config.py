import argparse
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

AI_SERVER_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = AI_SERVER_ROOT / "config.yaml"
BACKUP_PATH = CONFIG_PATH.with_suffix(".yaml.bak")
LOG_DIR = AI_SERVER_ROOT / "research" / "results" / "config_sweep_logs"

ENABLED_PATTERN = ("enabled:", ("true", "false"))
WAIT_PATTERN = "max_wait_ms:"


def set_config(dim_reduction: bool, max_wait_ms: int):
    lines = CONFIG_PATH.read_text(encoding="utf-8").splitlines()
    hit_enabled = hit_wait = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("enabled:") and not hit_enabled:
            prefix = line[: len(line) - len(line.lstrip())]
            comment = ""
            if "#" in line:
                comment = "  " + line[line.index("#"):]
            lines[i] = f"{prefix}enabled: {'true' if dim_reduction else 'false'}{comment}"
            hit_enabled = True
        elif stripped.startswith("max_wait_ms:") and not hit_wait:
            prefix = line[: len(line) - len(line.lstrip())]
            comment = ""
            if "#" in line:
                comment = "  " + line[line.index("#"):]
            lines[i] = f"{prefix}max_wait_ms: {max_wait_ms}{comment}"
            hit_wait = True

    if not (hit_enabled and hit_wait):
        raise RuntimeError(
            f"config.yaml에서 enabled/max_wait_ms 라인을 못 찾음 (enabled={hit_enabled}, max_wait_ms={hit_wait}) "
            "— config.yaml 구조가 바뀌었을 수 있음, 수동 확인 필요"
        )
    CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wait_for_ready(port: int, timeout_s: float = 60.0) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_combo(python_exe, port, dim_reduction, max_wait_ms, duration_s, idx, total):
    set_config(dim_reduction, max_wait_ms)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    label = f"combo{idx:02d}_dim{'on' if dim_reduction else 'off'}_wait{max_wait_ms}"
    log_path = LOG_DIR / f"{label}.log"

    print(f"\n{'=' * 70}")
    print(f"[{idx}/{total}] dim_reduction={'on' if dim_reduction else 'off'}  max_wait_ms={max_wait_ms}")
    print(f"서버 기동 중... (ai-server 로그: {log_path})")

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(CONFIG_PATH.parent),
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    try:
        if not wait_for_ready(port):
            print("[경고] /health 응답 없음 — 서버 기동 실패 가능성, 로그 확인 필요. 이 조합은 건너뜀.")
            return

        print(f"서버 준비 완료. 지금부터 {duration_s}초간 프런트에서 영상을 재생하고, "
              f"백엔드 콘솔의 T2-T1~T5-T4 로그를 '{label}' 이름으로 따로 저장해두세요.")
        remaining = duration_s
        while remaining > 0:
            step = min(10, remaining)
            time.sleep(step)
            remaining -= step
            print(f"  남은 시간: {remaining}s")

        print("측정 구간 종료 — 서버 종료 중...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()

    time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_wait_ms", type=int, nargs="+", default=[10, 30, 50, 100])
    parser.add_argument("--dim_reduction", choices=["on", "off", "both"], default="both")
    parser.add_argument("--duration_s", type=int, default=60)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--python", type=str, default="/home/lookie14/Intelligent-CCTV-ReID/venv/bin/python")
    args = parser.parse_args()

    dim_reduction_options = {"on": [True], "off": [False], "both": [True, False]}[args.dim_reduction]
    combos = [(d, w) for d in dim_reduction_options for w in args.max_wait_ms]

    print(f"[경고] 이 스크립트는 config.yaml의 rabbitmq 설정을 그대로 사용합니다 — "
          f"현재 프로덕션 큐를 바라보고 있다면 실제 트래픽에 영향을 줍니다.")
    shutil.copy(CONFIG_PATH, BACKUP_PATH)
    print(f"[백업] {CONFIG_PATH} → {BACKUP_PATH}")

    try:
        for i, (dim_reduction, max_wait_ms) in enumerate(combos, 1):
            run_combo(args.python, args.port, dim_reduction, max_wait_ms, args.duration_s, i, len(combos))
    except KeyboardInterrupt:
        print("\n[중단] 사용자 중단 — config.yaml 복원 후 종료합니다.")
    finally:
        shutil.copy(BACKUP_PATH, CONFIG_PATH)
        BACKUP_PATH.unlink(missing_ok=True)
        print(f"\n[복원] config.yaml을 원래 상태로 되돌림")

    print("\n모든 조합 실행 완료. 조합별 백엔드 로그를 모아 T4-T3 등 구간별 통계를 정리하세요.")


if __name__ == "__main__":
    main()
