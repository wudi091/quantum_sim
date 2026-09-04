#!/usr/bin/env bash

set -Eeuo pipefail

# 单文件固定任务：check | start | status | log | stop
# 只训练论文对齐的 TELGEN/IPM 连续轨迹模型，不启动旧自回归模型。

readonly PYTHON_BIN="/data/anaconda3/envs/reliq/bin/python"
readonly BASH_BIN="/bin/bash"
readonly FLOCK_BIN="/usr/bin/flock"
readonly NOHUP_BIN="/usr/bin/nohup"
readonly SETSID_BIN="/usr/bin/setsid"
readonly TAIL_BIN="/usr/bin/tail"
readonly PS_BIN="/bin/ps"
readonly GIT_BIN="/usr/bin/git"
readonly NVIDIA_SMI_BIN="/usr/bin/nvidia-smi"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SELF="${SCRIPT_DIR}/server_training_job.sh"
readonly RESULT_BASE="${SCRIPT_DIR}/results/telgen_ipm_delay_v1"
readonly STATE_DIR="${SCRIPT_DIR}/results/telgen_ipm_delay_state_v1"
readonly PID_FILE="${STATE_DIR}/training.pid"
readonly JOB_LOG="${STATE_DIR}/training.log"
readonly PHASE_FILE="${STATE_DIR}/phase.txt"
readonly STARTED_AT_FILE="${STATE_DIR}/started_at.txt"
readonly FINISHED_AT_FILE="${STATE_DIR}/finished_at.txt"
readonly EXIT_CODE_FILE="${STATE_DIR}/exit_code.txt"
readonly RUN_ROOT_FILE="${STATE_DIR}/run_root.txt"
readonly MANAGER_LOCK_FILE="${STATE_DIR}/manager.lock"
readonly TRAIN_LOCK_FILE="${SCRIPT_DIR}/results/telgen_ipm_delay_v1.lock"

readonly TRAIN_SAMPLES=400
readonly VALIDATION_SAMPLES=80
readonly TEST_SAMPLES=80
readonly CROSS_SAMPLES=80
readonly EPOCHS=150
readonly PATIENCE=40
readonly REQUEST_COUNT=8
readonly HORIZON=12
readonly PATH_COUNT=4
readonly CONSTRUCTION_PLANS=5
readonly IPM_STEPS=16

readonly TRAINING_SEED=20260826
declare -ar TRAIN_NODES=(64 96 128)
declare -ar TEST_NODES=(160 192)
declare -ar CROSS_NODES=(128 192)

usage() {
    echo "用法：$0 {check|start|status|log|stop}" >&2
}

ensure_directories() {
    mkdir -p "${STATE_DIR}" "${RESULT_BASE}"
}

read_pid() {
    local pid
    [[ -f "${PID_FILE}" ]] || return 1
    IFS= read -r pid < "${PID_FILE}" || return 1
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
    printf '%s\n' "${pid}"
}

process_exists() {
    kill -0 "$1" 2>/dev/null
}

process_group_exists() {
    kill -0 -- "-$1" 2>/dev/null
}

process_matches_worker() {
    local pid="$1"
    local token
    local -a argv=()

    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    while IFS= read -r -d '' token; do
        argv+=("${token}")
    done < "/proc/${pid}/cmdline"

    [[ ${#argv[@]} -ge 3 ]] || return 1
    [[ "${argv[0]##*/}" == "bash" ]] || return 1
    [[ "${argv[1]}" == "${SELF}" ]] || return 1
    [[ "${argv[2]}" == "_worker" ]]
}

remove_pid_if_owned() {
    local expected_pid="$1"
    local current_pid
    current_pid="$(read_pid 2>/dev/null || true)"
    if [[ "${current_pid}" == "${expected_pid}" ]]; then
        rm -f -- "${PID_FILE}"
    fi
}

set_phase() {
    printf '%s\n' "$1" > "${PHASE_FILE}"
    echo "$1"
}

validated_resume_run_root() {
    local candidate=""
    [[ -f "${RUN_ROOT_FILE}" ]] || return 1
    IFS= read -r candidate < "${RUN_ROOT_FILE}" || return 1
    [[ -n "${candidate}" && -d "${candidate}" ]] || return 1

    local resolved
    resolved="$(cd -- "${candidate}" && pwd -P)"
    [[ "${resolved}" == "${RESULT_BASE}/"* ]] || return 1
    [[ ! -f "${resolved}/COMPLETED" ]] || return 1
    printf '%s\n' "${resolved}"
}

check_environment() {
    ensure_directories
    cd "${SCRIPT_DIR}"

    local required
    for required in \
        "${PYTHON_BIN}" \
        "${BASH_BIN}" \
        "${FLOCK_BIN}" \
        "${NOHUP_BIN}" \
        "${SETSID_BIN}" \
        "${TAIL_BIN}" \
        "${PS_BIN}" \
        "${GIT_BIN}" \
        "${NVIDIA_SMI_BIN}"; do
        if [[ ! -x "${required}" ]]; then
            echo "缺少可执行文件：${required}" >&2
            return 1
        fi
    done

    "${BASH_BIN}" -n "${SELF}"
    "${PYTHON_BIN}" -c '
import scipy
import sequence
import torch
from algorithms.telgen.ipm_trajectory_pilot import TELGENPaperGNN, build_parser

if not torch.cuda.is_available():
    raise SystemExit("未检测到 CUDA")
TELGENPaperGNN()
build_parser()
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"scipy={scipy.__version__}")
print("SeQUeNCe、SciPy IPM 和 TELGEN 训练入口：OK")
'
    "${NVIDIA_SMI_BIN}" --query-gpu=name,memory.total,memory.used \
        --format=csv,noheader
}

validate_report() {
    local report="$1"
    local checkpoint="$2"
    local seed="$3"
    "${PYTHON_BIN}" - "${report}" "${checkpoint}" "${seed}" <<'PY'
import json
import math
import pathlib
import sys
import torch

report_path = pathlib.Path(sys.argv[1])
checkpoint_path = pathlib.Path(sys.argv[2])
seed = int(sys.argv[3])
if not report_path.is_file() or not checkpoint_path.is_file():
    raise SystemExit(1)
payload = json.loads(report_path.read_text(encoding="utf-8"))
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)
decoder = payload.get("paper_alignment", {}).get("decoder", {})
valid = (
    payload.get("method") == "TELGEN IPM-trajectory GNN for the single-stage expected-delay LP"
    and payload.get("seed") == seed
    and payload.get("data_seed") == 20260826
    and payload.get("device") == "cuda"
    and payload.get("epochs") == 150
    and payload.get("data_protocol", {}).get("train", {}).get("samples") == 400
    and payload.get("data_protocol", {}).get("validation", {}).get("samples") == 80
    and payload.get("best_epoch", 0) > 0
    and math.isfinite(float(payload.get("best_validation_loss", float("nan"))))
    and payload.get("learning_rate") == 0.0002
    and payload.get("quantum_adaptation", {}).get("objective") == "single-stage expected censored completion latency"
    and payload.get("paper_alignment", {}).get("constraint_supervision") == "final readout only; intermediate IPM iterates may be temporarily infeasible"
    and checkpoint.get("schema_version") == 5
    and checkpoint.get("model_class") == "TELGENPaperGNN"
    and checkpoint.get("objective") == "expected_censored_completion_latency"
    and decoder.get("name") == "shared_capacity_safe_rounding"
    and decoder.get("admission_mass") == "unscaled request mass"
    and decoder.get("teacher_and_gnn_share_decoder") is True
)
raise SystemExit(0 if valid else 1)
PY
}

run_training_seed() {
    local run_root="$1"
    local seed="$2"
    local model_dir="${run_root}/seed_${seed}"
    local report="${model_dir}/telgen_ipm_report.json"
    local checkpoint="${model_dir}/telgen_ipm_model.pt"
    local log_file="${run_root}/logs/seed_${seed}.log"

    mkdir -p "${model_dir}" "${run_root}/logs"
    if validate_report "${report}" "${checkpoint}" "${seed}"; then
        echo "训练 seed=${seed} 已完成，跳过。"
        return 0
    fi

    "${PYTHON_BIN}" -m algorithms.telgen.ipm_trajectory_pilot \
        --output "${report}" \
        --checkpoint "${checkpoint}" \
        --train-samples "${TRAIN_SAMPLES}" \
        --validation-samples "${VALIDATION_SAMPLES}" \
        --test-samples "${TEST_SAMPLES}" \
        --cross-samples "${CROSS_SAMPLES}" \
        --epochs "${EPOCHS}" \
        --patience "${PATIENCE}" \
        --seed "${seed}" \
        --data-seed 20260826 \
        --dataset-cache "${run_root}/dataset/telgen_ipm_dataset.pkl" \
        --hidden-dim 180 \
        --inner-layers 2 \
        --message-mlp-layers 4 \
        --prediction-layers 4 \
        --objective-weight 3.43 \
        --constraint-weight 5.8 \
        --request-mass-weight 2.0 \
        --candidate-distribution-weight 0.5 \
        --learning-rate 0.0002 \
        --weight-decay 0 \
        --train-topology waxman \
        --train-nodes "${TRAIN_NODES[@]}" \
        --test-topology waxman \
        --test-nodes "${TEST_NODES[@]}" \
        --cross-topology barabasi_albert \
        --cross-nodes "${CROSS_NODES[@]}" \
        --request-count "${REQUEST_COUNT}" \
        --horizon "${HORIZON}" \
        --path-count "${PATH_COUNT}" \
        --construction-plans "${CONSTRUCTION_PLANS}" \
        --ipm-steps "${IPM_STEPS}" \
        --device cuda \
        2>&1 | tee "${log_file}"

    validate_report "${report}" "${checkpoint}" "${seed}"
}

run_pipeline() {
    local run_root
    local run_id
    run_root="$(validated_resume_run_root 2>/dev/null || true)"
    if [[ -z "${run_root}" ]]; then
        run_id="$(date +%Y%m%d_%H%M%S)_pid$$"
        run_root="${RESULT_BASE}/${run_id}"
        mkdir -p "${run_root}/logs"
    fi
    printf '%s\n' "${run_root}" > "${RUN_ROOT_FILE}"
    cd "${SCRIPT_DIR}"

    export PYTHONUNBUFFERED=1
    export PYTHONNOUSERSITE=1
    export OMP_NUM_THREADS=16
    export MKL_NUM_THREADS=16
    export OPENBLAS_NUM_THREADS=16

    echo "运行目录：${run_root}"
    set_phase "阶段 0/3：记录环境与代码版本"
    {
        "${GIT_BIN}" -C "${SCRIPT_DIR}" rev-parse HEAD
        "${PYTHON_BIN}" -c '
import scipy
import torch
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"scipy={scipy.__version__}")
'
    } 2>&1 | tee "${run_root}/logs/environment.log"

    set_phase "阶段 1/3：同配置 GPU 冒烟训练"
    local sanity_report="${run_root}/sanity/telgen_ipm_report.json"
    local sanity_checkpoint="${run_root}/sanity/telgen_ipm_model.pt"
    if [[ ! -f "${sanity_report}" || ! -f "${sanity_checkpoint}" ]]; then
        mkdir -p "${run_root}/sanity"
        "${PYTHON_BIN}" -m algorithms.telgen.ipm_trajectory_pilot \
            --output "${sanity_report}" \
            --checkpoint "${sanity_checkpoint}" \
            --train-samples 4 \
            --validation-samples 2 \
            --test-samples 2 \
            --cross-samples 2 \
            --epochs 1 \
            --patience 1 \
            --seed 20260825 \
            --train-nodes 32 48 \
            --test-nodes 64 \
            --cross-nodes 64 \
            --request-count 4 \
            --horizon 8 \
            --path-count 3 \
            --construction-plans 3 \
            --ipm-steps 4 \
            --hidden-dim 32 \
            --inner-layers 1 \
            --message-mlp-layers 2 \
            --prediction-layers 2 \
            --device cuda \
            2>&1 | tee "${run_root}/logs/sanity.log"
    fi
    test -f "${sanity_report}"
    test -f "${sanity_checkpoint}"

    set_phase "阶段 2/3：正式训练（单种子），seed=${TRAINING_SEED}"
    run_training_seed "${run_root}" "${TRAINING_SEED}"

    touch "${run_root}/COMPLETED"
    set_phase "全部完成"
    echo "结果目录：${run_root}"
}

start_job() {
    ensure_directories
    local old_pid
    old_pid="$(read_pid 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && process_exists "${old_pid}"; then
        if process_matches_worker "${old_pid}"; then
            echo "训练已经在运行，PID=${old_pid}"
            return 0
        fi
        echo "PID 文件指向其他进程，拒绝启动：${old_pid}" >&2
        return 1
    fi
    rm -f -- "${PID_FILE}"

    local probe_lock_fd
    exec {probe_lock_fd}>"${TRAIN_LOCK_FILE}"
    if ! "${FLOCK_BIN}" -n "${probe_lock_fd}"; then
        exec {probe_lock_fd}>&-
        echo "训练锁已被占用，已有任务正在运行。"
        return 0
    fi
    "${FLOCK_BIN}" -u "${probe_lock_fd}"
    exec {probe_lock_fd}>&-

    check_environment
    rm -f -- "${FINISHED_AT_FILE}" "${EXIT_CODE_FILE}" "${PHASE_FILE}"
    {
        echo
        echo "===== 启动 $(date --iso-8601=seconds) ====="
    } >> "${JOB_LOG}"

    "${NOHUP_BIN}" "${SETSID_BIN}" "${BASH_BIN}" "${SELF}" _worker \
        </dev/null >> "${JOB_LOG}" 2>&1 &

    local attempt
    local worker_pid=""
    for attempt in {1..30}; do
        worker_pid="$(read_pid 2>/dev/null || true)"
        if [[ -n "${worker_pid}" ]] && process_exists "${worker_pid}"; then
            break
        fi
        sleep 0.1
    done
    if [[ -z "${worker_pid}" ]] || ! process_exists "${worker_pid}"; then
        echo "后台训练启动失败，请查看日志。" >&2
        "${TAIL_BIN}" -n 80 "${JOB_LOG}" >&2 || true
        return 1
    fi
    if ! process_matches_worker "${worker_pid}"; then
        echo "后台进程身份校验失败：${worker_pid}" >&2
        return 1
    fi
    echo "训练已在后台启动，PID=${worker_pid}"
    echo "日志：${JOB_LOG}"
}

worker() {
    ensure_directories
    exec 9>"${TRAIN_LOCK_FILE}"
    if ! "${FLOCK_BIN}" -n 9; then
        echo "已有训练正在运行，worker 退出。" >&2
        exit 3
    fi

    cleanup() {
        local exit_code=$?
        printf '%s\n' "${exit_code}" > "${EXIT_CODE_FILE}"
        date --iso-8601=seconds > "${FINISHED_AT_FILE}"
        remove_pid_if_owned "$$"
        echo "===== 结束 exit=${exit_code} $(date --iso-8601=seconds) ====="
    }
    trap cleanup EXIT
    trap 'exit 143' TERM INT

    printf '%s\n' "$$" > "${PID_FILE}"
    date --iso-8601=seconds > "${STARTED_AT_FILE}"
    run_pipeline
}

show_status() {
    ensure_directories
    local pid
    pid="$(read_pid 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && process_exists "${pid}" && process_matches_worker "${pid}"; then
        echo "状态：RUNNING"
        echo "PID：${pid}"
        [[ -f "${STARTED_AT_FILE}" ]] && echo "开始时间：$(<"${STARTED_AT_FILE}")"
        "${PS_BIN}" -o pid,ppid,pgid,etime,%cpu,%mem,stat,cmd -p "${pid}"
        "${NVIDIA_SMI_BIN}" --query-gpu=name,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader || true
    elif [[ -f "${EXIT_CODE_FILE}" ]]; then
        echo "状态：FINISHED"
        echo "退出码：$(<"${EXIT_CODE_FILE}")"
        [[ -f "${FINISHED_AT_FILE}" ]] && echo "结束时间：$(<"${FINISHED_AT_FILE}")"
    else
        echo "状态：NOT_RUNNING"
    fi
    [[ -f "${PHASE_FILE}" ]] && echo "阶段：$(<"${PHASE_FILE}")"
    [[ -f "${RUN_ROOT_FILE}" ]] && echo "运行目录：$(<"${RUN_ROOT_FILE}")"
    if [[ -f "${JOB_LOG}" ]]; then
        echo "最近日志："
        "${TAIL_BIN}" -n 35 "${JOB_LOG}"
    fi
}

show_log() {
    ensure_directories
    if [[ -f "${JOB_LOG}" ]]; then
        "${TAIL_BIN}" -n 200 "${JOB_LOG}"
    else
        echo "尚无训练日志。"
    fi
}

stop_job() {
    ensure_directories
    local pid
    local pgid
    pid="$(read_pid 2>/dev/null || true)"
    if [[ -z "${pid}" ]] || ! process_exists "${pid}"; then
        echo "当前没有正在运行的训练任务。"
        rm -f -- "${PID_FILE}"
        return 0
    fi
    if ! process_matches_worker "${pid}"; then
        echo "PID=${pid} 不属于训练 worker，拒绝发送信号。" >&2
        return 1
    fi
    pgid="$("${PS_BIN}" -o pgid= -p "${pid}" | tr -d ' ')"
    if [[ "${pgid}" != "${pid}" ]]; then
        echo "训练 worker 不是独立进程组，拒绝发送信号。" >&2
        return 1
    fi
    echo "向训练进程组 ${pgid} 发送 TERM。"
    kill -TERM -- "-${pgid}"

    local attempt
    for attempt in {1..20}; do
        if ! process_group_exists "${pgid}"; then
            echo "训练任务已停止。"
            return 0
        fi
        sleep 1
    done
    echo "任务仍在退出中；没有发送 KILL，请稍后查看状态。" >&2
    return 1
}

main() {
    if [[ $# -ne 1 ]]; then
        usage
        return 2
    fi
    case "$1" in
        check) check_environment ;;
        start)
            ensure_directories
            exec "${FLOCK_BIN}" -x -w 5 -o "${MANAGER_LOCK_FILE}" \
                "${BASH_BIN}" "${SELF}" _start
            ;;
        status) show_status ;;
        log) show_log ;;
        stop)
            ensure_directories
            exec "${FLOCK_BIN}" -x -w 5 -o "${MANAGER_LOCK_FILE}" \
                "${BASH_BIN}" "${SELF}" _stop
            ;;
        _start) start_job ;;
        _worker) worker ;;
        _stop) stop_job ;;
        *) usage; return 2 ;;
    esac
}

main "$@"
