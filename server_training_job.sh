#!/usr/bin/env bash
set -euo pipefail

REPOSITORY=/data02/qbit/quantum_sim
PYTHON=/data/anaconda3/envs/reliq/bin/python
CONFIG="${REPOSITORY}/configs/arcq_train.yaml"
OUTPUT="${REPOSITORY}/results/arcq/training"
CHECKPOINT="${OUTPUT}/arcq_latest.pt"
LOG="${OUTPUT}/training.log"
PID_FILE="${OUTPUT}/training.pid"
START_LOCK="${OUTPUT}/start.lock"

usage() {
  echo "usage: server_training_job.sh {check|start|status|log|stop}"
}

read_pid() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi
  local value
  value=$(<"${PID_FILE}")
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  printf '%s' "${value}"
}

is_arcq_process() {
  local process_id=$1
  [[ -r "/proc/${process_id}/cmdline" ]] || return 1
  local command_line
  command_line=$(tr '\0' ' ' < "/proc/${process_id}/cmdline")
  [[ "${command_line}" == *"algorithms.rl_routing.train"* ]] &&
    [[ "${command_line}" == *"${CONFIG}"* ]]
}

running_pid() {
  local process_id
  process_id=$(read_pid) || return 1
  kill -0 "${process_id}" 2>/dev/null || return 1
  is_arcq_process "${process_id}" || return 1
  printf '%s' "${process_id}"
}

training_complete() {
  [[ -f "${CHECKPOINT}" ]] || return 1
  cd "${REPOSITORY}"
  "${PYTHON}" -c \
    "from algorithms.rl_routing.checkpoint import load_arcq_checkpoint; from algorithms.rl_routing.train import load_training_config; c=load_training_config('configs/arcq_train.yaml'); _,m=load_arcq_checkpoint('results/arcq/training/arcq_latest.pt'); raise SystemExit(0 if int(m['training_state']['episodes_completed']) >= c.run.episode_count else 1)"
}

check_job() {
  [[ -d "${REPOSITORY}" ]]
  [[ -x "${PYTHON}" ]]
  [[ -f "${CONFIG}" ]]
  command -v flock >/dev/null
  cd "${REPOSITORY}"
  "${PYTHON}" -c \
    "from algorithms.rl_routing.train import load_training_config; load_training_config('configs/arcq_train.yaml'); print('ARC-Q training check passed')"
}

start_job() {
  local process_id
  mkdir -p "${OUTPUT}"
  exec 9> "${START_LOCK}"
  if ! flock -n 9; then
    echo "another ARC-Q start operation is in progress"
    return 1
  fi
  if process_id=$(running_pid); then
    echo "ARC-Q training is already running: pid=${process_id}"
    return 0
  fi
  if training_complete; then
    echo "ARC-Q training is already complete"
    return 0
  fi
  check_job
  cd "${REPOSITORY}"
  local resume_arguments=()
  if [[ -f "${CHECKPOINT}" ]]; then
    resume_arguments=(--resume "${CHECKPOINT}")
  fi
  nohup "${PYTHON}" -u -m algorithms.rl_routing.train \
    --config "${CONFIG}" "${resume_arguments[@]}" \
    >> "${LOG}" 2>&1 &
  process_id=$!
  printf '%s\n' "${process_id}" > "${PID_FILE}"
  sleep 1
  if ! running_pid >/dev/null; then
    echo "ARC-Q training failed to stay alive; inspect ${LOG}" >&2
    return 1
  fi
  echo "ARC-Q training started: pid=${process_id}"
}

status_job() {
  local process_id
  if process_id=$(running_pid); then
    echo "ARC-Q training is running: pid=${process_id}"
  elif training_complete; then
    echo "ARC-Q training is complete"
  else
    echo "ARC-Q training is not running"
  fi
  if [[ -f "${LOG}" ]]; then
    tail -n 20 "${LOG}"
  fi
}

log_job() {
  if [[ ! -f "${LOG}" ]]; then
    echo "ARC-Q training log does not exist"
    return 0
  fi
  tail -n 80 "${LOG}"
}

stop_job() {
  local process_id
  if ! process_id=$(running_pid); then
    echo "ARC-Q training is not running"
    return 0
  fi
  kill -TERM "${process_id}"
  for _ in {1..30}; do
    if ! kill -0 "${process_id}" 2>/dev/null; then
      echo "ARC-Q training stopped: pid=${process_id}"
      return 0
    fi
    sleep 1
  done
  echo "ARC-Q training did not stop after SIGTERM" >&2
  return 1
}

case "${1:-}" in
  check) check_job ;;
  start) start_job ;;
  status) status_job ;;
  log) log_job ;;
  stop) stop_job ;;
  *) usage; exit 2 ;;
esac
