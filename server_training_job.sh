#!/usr/bin/env bash

set -Eeuo pipefail

# 单文件固定训练任务：check | start | status | log | stop

readonly PYTHON_BIN="/data/anaconda3/envs/reliq/bin/python"
readonly BASH_BIN="/bin/bash"
readonly FLOCK_BIN="/usr/bin/flock"
readonly NOHUP_BIN="/usr/bin/nohup"
readonly SETSID_BIN="/usr/bin/setsid"
readonly TAIL_BIN="/usr/bin/tail"
readonly PS_BIN="/bin/ps"
readonly GIT_BIN="/usr/bin/git"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SELF="${SCRIPT_DIR}/server_training_job.sh"
readonly RESULT_BASE="${SCRIPT_DIR}/results/server_generalization_v2"
readonly STATE_DIR="${SCRIPT_DIR}/results/server_training_state"
readonly PID_FILE="${STATE_DIR}/training.pid"
readonly JOB_LOG="${STATE_DIR}/training.log"
readonly PHASE_FILE="${STATE_DIR}/phase.txt"
readonly STARTED_AT_FILE="${STATE_DIR}/started_at.txt"
readonly FINISHED_AT_FILE="${STATE_DIR}/finished_at.txt"
readonly EXIT_CODE_FILE="${STATE_DIR}/exit_code.txt"
readonly RUN_ROOT_FILE="${STATE_DIR}/run_root.txt"
readonly MANAGER_LOCK_FILE="${STATE_DIR}/manager.lock"
readonly TRAIN_LOCK_FILE="${SCRIPT_DIR}/results/server_training.lock"

readonly MILP_TIME_LIMIT_SECONDS=300
readonly TRAIN_EPISODES_PER_GROUP=150
readonly VALIDATION_EPISODES=25
readonly TEST_EPISODES_PER_GROUP=25
readonly ONLINE_TEST_EPISODES=100

declare -ar TRAINING_SEEDS=(20260821 20260822 20260823)
declare -ar VALIDATION_SEEDS=(
    220000 220001 220002 220003 220004
    220005 220006 220007 220008 220009
    220010 220011 220012 220013 220014
    220015 220016 220017 220018 220019
    220020 220021 220022 220023 220024
)
declare -ar TEST_SEEDS=(
    230000 230001 230002 230003 230004
    230005 230006 230007 230008 230009
    230010 230011 230012 230013 230014
    230015 230016 230017 230018 230019
    230020 230021 230022 230023 230024
    231000 231001 231002 231003 231004
    231005 231006 231007 231008 231009
    231010 231011 231012 231013 231014
    231015 231016 231017 231018 231019
    231020 231021 231022 231023 231024
)

declare -ar LABEL_SCENARIO_ARGS=(
    --requests 40
    --requests-per-batch 10
    --decision-interval 4
    --ttl 16
    --min-hops 4
    --max-hops 4
    --paths 4
    --construction-plans 5
    --generation-probability 0.8
    --swap-probability 0.9
    --memory-capacity 2
    --quantum-distance-m 1000
    --slot-duration-ps 50000000
)

declare -ar ONLINE_SCENARIO_ARGS=(
    --requests 100
    --requests-per-batch 10
    --decision-interval 4
    --ttl 16
    --min-hops 4
    --max-hops 4
    --paths 4
    --construction-plans 5
    --generation-probability 0.8
    --swap-probability 0.9
    --memory-capacity 2
    --quantum-distance-m 1000
    --slot-duration-ps 50000000
)

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
        "${GIT_BIN}"; do
        if [[ ! -x "${required}" ]]; then
            echo "缺少可执行文件：${required}" >&2
            return 1
        fi
    done

    "${BASH_BIN}" -n "${SELF}"
    "${PYTHON_BIN}" -c '
import scipy
import torch
import sequence
from scipy.optimize import milp
from algorithms.telgen.analyze_online_gnn import main as analyze_main
from algorithms.telgen.combine_online_milp_datasets import main as combine_main
from algorithms.telgen.generate_online_milp_data import main as generate_main
from algorithms.telgen.train_online_milp_gnn import main as train_main
from algorithms.telgen.compare_online_gnn import main as compare_main

if not torch.cuda.is_available():
    raise SystemExit("未检测到 CUDA")

print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"scipy={scipy.__version__}")
print("SeQUeNCe、MILP、训练和评价入口：OK")
'
    local df_bin
    df_bin="$(command -v df || true)"
    if [[ -z "${df_bin}" || ! -x "${df_bin}" ]]; then
        echo "缺少可执行文件：df" >&2
        return 1
    fi
    "${df_bin}" -h "${SCRIPT_DIR}"
}

validated_resume_run_root() {
    local candidate=""
    local line
    if [[ -f "${RUN_ROOT_FILE}" ]]; then
        IFS= read -r candidate < "${RUN_ROOT_FILE}" || candidate=""
    elif [[ -f "${JOB_LOG}" ]]; then
        while IFS= read -r line; do
            if [[ "${line}" == "运行目录："* ]]; then
                candidate="${line#运行目录：}"
            fi
        done < "${JOB_LOG}"
    fi
    [[ -n "${candidate}" && -d "${candidate}" ]] || return 1

    local resolved
    resolved="$(cd -- "${candidate}" && pwd -P)"
    [[ "${resolved}" == "${RESULT_BASE}/"* ]] || return 1
    [[ -d "${resolved}/collections" ]] || return 1
    [[ ! -f "${resolved}/COMPLETED" ]] || return 1
    printf '%s\n' "${resolved}"
}

set_phase() {
    printf '%s\n' "$1" > "${PHASE_FILE}"
    echo "$1"
}

generate_collection() {
    local collections_dir="$1"
    local log_dir="$2"
    local name="$3"
    local episodes="$4"
    local seed_start="$5"
    local nodes="$6"
    local topology_mode="$7"
    shift 7

    local output="${collections_dir}/${name}"
    "${PYTHON_BIN}" -m algorithms.telgen.generate_online_milp_data \
        --output "${output}" \
        --episodes "${episodes}" \
        --seed-start "${seed_start}" \
        --nodes "${nodes}" \
        --topology-mode "${topology_mode}" \
        --time-limit-seconds "${MILP_TIME_LIMIT_SECONDS}" \
        --resume \
        "${LABEL_SCENARIO_ARGS[@]}" \
        "$@" \
        2>&1 | tee -a "${log_dir}/01_${name}.log"
    test -f "${output}/online_milp_dataset.json"
}

train_model() {
    local suite_manifest="$1"
    local models_dir="$2"
    local log_dir="$3"
    local training_seed="$4"
    local model_dir="${models_dir}/seed_${training_seed}"

    if [[ -f "${model_dir}/online_milp_gnn.pt" \
        && -f "${model_dir}/online_milp_gnn.json" ]]; then
        echo "训练 seed=${training_seed} 已完成，跳过。"
        return 0
    fi

    "${PYTHON_BIN}" -m algorithms.telgen.train_online_milp_gnn \
        --dataset "${suite_manifest}" \
        --output "${model_dir}" \
        --epochs 25 \
        --patience 10 \
        --learning-rate 0.001 \
        --weight-decay 0.00001 \
        --hidden-dim 32 \
        --layers 2 \
        --training-seed "${training_seed}" \
        --validation-seeds "${VALIDATION_SEEDS[@]}" \
        --test-seeds "${TEST_SEEDS[@]}" \
        --batch-size 2 \
        --evaluation-batch-size 2 \
        --random-baseline-trials 32 \
        --target-mode set \
        --device cuda \
        2>&1 | tee "${log_dir}/03_train_${training_seed}.log"
    test -f "${model_dir}/online_milp_gnn.pt"
    test -f "${model_dir}/online_milp_gnn.json"
}

run_online_case() {
    local checkpoint="$1"
    local output="$2"
    local log_file="$3"
    local seed_start="$4"
    local nodes="$5"
    local topology_mode="$6"
    shift 6

    if [[ -f "${output}/online_gnn_comparison.json" ]]; then
        echo "在线评价 ${output} 已完成，跳过。"
        return 0
    fi

    "${PYTHON_BIN}" -m algorithms.telgen.compare_online_gnn \
        --checkpoint "${checkpoint}" \
        --output "${output}" \
        --seeds "${ONLINE_TEST_EPISODES}" \
        --seed-start "${seed_start}" \
        --nodes "${nodes}" \
        --topology-mode "${topology_mode}" \
        --gnn-device cuda \
        --skip-milp \
        "${ONLINE_SCENARIO_ARGS[@]}" \
        "$@" \
        2>&1 | tee "${log_file}"
    test -f "${output}/online_gnn_comparison.json"
}

run_pipeline() {
    local run_id
    local run_root
    local collections_dir
    local suite_dir
    local models_dir
    local online_dir
    local analysis_dir
    local log_dir
    local training_seed

    run_root="$(validated_resume_run_root 2>/dev/null || true)"
    if [[ -z "${run_root}" ]]; then
        run_id="$(date +%Y%m%d_%H%M%S)_pid$$"
        run_root="${RESULT_BASE}/${run_id}"
        mkdir "${run_root}"
    fi
    collections_dir="${run_root}/collections"
    suite_dir="${run_root}/suite"
    models_dir="${run_root}/models"
    online_dir="${run_root}/online_ood"
    analysis_dir="${run_root}/analysis"
    log_dir="${run_root}/logs"

    mkdir -p \
        "${collections_dir}" \
        "${suite_dir}" \
        "${models_dir}" \
        "${online_dir}" \
        "${analysis_dir}" \
        "${log_dir}"
    printf '%s\n' "${run_root}" > "${RUN_ROOT_FILE}"
    cd "${SCRIPT_DIR}"

    export PYTHONUNBUFFERED=1
    export PYTHONNOUSERSITE=1
    export OMP_NUM_THREADS="$(nproc)"
    export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

    echo "运行目录：${run_root}"
    set_phase "阶段 0/5：记录环境与代码版本"
    {
        "${GIT_BIN}" -C "${SCRIPT_DIR}" rev-parse HEAD
        "${PYTHON_BIN}" -c '
import scipy
import torch
import sequence
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"scipy={scipy.__version__}")
print("SeQUeNCe：OK")
'
    } 2>&1 | tee "${log_dir}/00_environment.log"

    set_phase "阶段 1/5：生成多拓扑精确 MILP 标签"
    generate_collection \
        "${collections_dir}" "${log_dir}" train_waxman_64 \
        "${TRAIN_EPISODES_PER_GROUP}" 210000 64 waxman \
        --waxman-alpha 0.15 --waxman-beta 0.45
    generate_collection \
        "${collections_dir}" "${log_dir}" train_waxman_96 \
        "${TRAIN_EPISODES_PER_GROUP}" 211000 96 waxman \
        --waxman-alpha 0.20 --waxman-beta 0.40
    generate_collection \
        "${collections_dir}" "${log_dir}" train_waxman_128 \
        "${TRAIN_EPISODES_PER_GROUP}" 212000 128 waxman \
        --waxman-alpha 0.10 --waxman-beta 0.55
    generate_collection \
        "${collections_dir}" "${log_dir}" validation_waxman_160 \
        "${VALIDATION_EPISODES}" 220000 160 waxman \
        --waxman-alpha 0.18 --waxman-beta 0.50
    generate_collection \
        "${collections_dir}" "${log_dir}" test_waxman_192 \
        "${TEST_EPISODES_PER_GROUP}" 230000 192 waxman \
        --waxman-alpha 0.12 --waxman-beta 0.60
    generate_collection \
        "${collections_dir}" "${log_dir}" test_barabasi_128 \
        "${TEST_EPISODES_PER_GROUP}" 231000 128 barabasi_albert \
        --barabasi-attachment 2

    set_phase "阶段 2/5：合并固定训练、验证与测试划分"
    "${PYTHON_BIN}" -m algorithms.telgen.combine_online_milp_datasets \
        --output "${suite_dir}" \
        --profile generalization_v2 \
        --input train_waxman_64 train \
            "${collections_dir}/train_waxman_64" \
        --input train_waxman_96 train \
            "${collections_dir}/train_waxman_96" \
        --input train_waxman_128 train \
            "${collections_dir}/train_waxman_128" \
        --input validation_waxman_160 validation \
            "${collections_dir}/validation_waxman_160" \
        --input test_waxman_192 test \
            "${collections_dir}/test_waxman_192" \
        --input test_barabasi_128 test \
            "${collections_dir}/test_barabasi_128" \
        2>&1 | tee "${log_dir}/02_combine.log"
    test -f "${suite_dir}/online_milp_dataset.json"

    set_phase "阶段 3/5：训练 3 个随机种子的自回归 GNN"
    for training_seed in "${TRAINING_SEEDS[@]}"; do
        train_model \
            "${suite_dir}/online_milp_dataset.json" \
            "${models_dir}" \
            "${log_dir}" \
            "${training_seed}"
    done

    set_phase "阶段 4/5：在未见拓扑上在线比较 GNN 与 Q-CAST"
    for training_seed in "${TRAINING_SEEDS[@]}"; do
        local checkpoint="${models_dir}/seed_${training_seed}/online_milp_gnn.pt"
        local seed_online_dir="${online_dir}/seed_${training_seed}"
        run_online_case \
            "${checkpoint}" \
            "${seed_online_dir}/waxman_192" \
            "${log_dir}/04_online_${training_seed}_waxman_192.log" \
            30000 192 waxman \
            --waxman-alpha 0.12 --waxman-beta 0.60
        run_online_case \
            "${checkpoint}" \
            "${seed_online_dir}/barabasi_128" \
            "${log_dir}/04_online_${training_seed}_barabasi_128.log" \
            31000 128 barabasi_albert \
            --barabasi-attachment 2
    done

    set_phase "阶段 5/5：生成配对统计报告"
    for training_seed in "${TRAINING_SEEDS[@]}"; do
        local seed_online_dir="${online_dir}/seed_${training_seed}"
        local seed_analysis_dir="${analysis_dir}/seed_${training_seed}"
        if [[ -f "${seed_analysis_dir}/online_benchmark.json" ]]; then
            echo "统计报告 seed=${training_seed} 已完成，跳过。"
            continue
        fi
        "${PYTHON_BIN}" -m algorithms.telgen.analyze_online_gnn \
            "${seed_online_dir}/waxman_192/online_gnn_comparison.json" \
            "${seed_online_dir}/barabasi_128/online_gnn_comparison.json" \
            --output "${seed_analysis_dir}" \
            --bootstrap-samples 20000 \
            --randomization-samples 20000 \
            --random-seed "${training_seed}" \
            2>&1 | tee "${log_dir}/05_analysis_${training_seed}.log"
        test -f "${seed_analysis_dir}/online_benchmark.json"
    done

    touch "${run_root}/COMPLETED"
    set_phase "全部完成"
    echo "模型目录：${models_dir}"
    echo "在线评价：${online_dir}"
    echo "统计报告：${analysis_dir}"
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
        echo "训练锁已被占用，已有训练正在运行。"
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

    # worker 持有整个训练过程的锁，第二个 worker 会立即退出。
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
        "${TAIL_BIN}" -n 25 "${JOB_LOG}"
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

    pgid="$(${PS_BIN} -o pgid= -p "${pid}" | tr -d ' ')"
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
