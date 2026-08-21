#!/usr/bin/env bash
# Compare scheduling policies on the distributed inference workload.
#
# Set MODEL=models/mobilenet_v2.onnx to run the inference stage on a real ONNX model.
#
# Cluster shape (all processes on this machine):
#   node A  label=cpu  submits every request  (the origin)
#   node B  label=gpu  idle helper — runs the inference stage ~8x faster
#   node C  label=cpu  idle helper
#
# The origin cannot finish quickly on its own, and the expensive stage is one only the
# gpu node is good at. A policy that knows nothing about load or capability has to get
# that wrong more often than one that does.
#
#   ./scripts/bench_inference.sh [requests] [reps]

set -uo pipefail
cd "$(dirname "$0")/.."

REQUESTS=${1:-400}
REPS=${2:-3}
WORKERS=${WORKERS:-2}
# Set MODEL=models/mobilenet_v2.onnx to run the inference stage on a real ONNX model
# instead of the simulated cost. Every node must load the same model, since any node
# may end up executing any stage.
MODEL=${MODEL:-}
BIN=./build
mkdir -p results

run_policy() {
  local policy=$1
  local out; out=$(mktemp -d)
  local cp=$((9700 + RANDOM % 400))

  "$BIN/hydra_coordinator" --port "$cp" --seconds 90 > "$out/coord.log" 2>&1 &
  local coord_pid=$!
  sleep 0.4

  # Idle helpers. They hold no work of their own; anything they run was stolen.
  "$BIN/hydra_inference" --coordinator-port "$cp" --workers "$WORKERS" \
      --label gpu --policy "$policy" ${MODEL:+--model "$MODEL"} --seconds 60 > "$out/gpu.log" 2>&1 &
  local gpu_pid=$!
  "$BIN/hydra_inference" --coordinator-port "$cp" --workers "$WORKERS" \
      --label cpu --policy "$policy" ${MODEL:+--model "$MODEL"} --seconds 60 > "$out/cpu.log" 2>&1 &
  local cpu_pid=$!
  sleep 0.6

  # The origin: submits everything, then waits for the whole cluster to finish it.
  "$BIN/hydra_inference" --coordinator-port "$cp" --workers "$WORKERS" \
      --label cpu --policy "$policy" ${MODEL:+--model "$MODEL"} --bench "$REQUESTS" > "$out/origin.log" 2>&1 &
  local origin_pid=$!

  wait "$origin_pid" 2>/dev/null

  local line; line=$(grep "BENCH" "$out/origin.log" 2>/dev/null || true)
  local gpu_ran; gpu_ran=$(grep -oE "tasks_executed\":[0-9]+" "$out/gpu.log" 2>/dev/null | head -1 | grep -oE "[0-9]+$" || echo 0)

  kill "$coord_pid" "$gpu_pid" "$cpu_pid" 2>/dev/null
  wait 2>/dev/null
  rm -rf "$out"
  echo "$line"
}

echo "=== HydraRT: distributed inference engine — scheduling policy comparison ==="
echo "requests=$REQUESTS  reps=$REPS  workers/node=$WORKERS  model=${MODEL:-simulated}"
echo "cluster: origin(cpu) + helper(gpu) + helper(cpu); inference stage is 8x faster on gpu"
echo

printf "%-12s %10s %12s %10s %10s %10s %8s\n" \
       "policy" "time_ms" "req/s" "p50_ms" "p95_ms" "p99_ms" "stolen"
echo "---------------------------------------------------------------------------------"

for policy in none random load-aware adaptive; do
  best_time=""
  best_line=""
  for ((r = 1; r <= REPS; r++)); do
    line=$(run_policy "$policy")
    [[ -z "$line" ]] && continue
    t=$(echo "$line" | grep -oE "time=[0-9.]+" | cut -d= -f2)
    [[ -z "$t" ]] && continue
    if [[ -z "$best_time" ]] || (( $(echo "$t < $best_time" | bc -l) )); then
      best_time=$t
      best_line=$line
    fi
  done

  if [[ -z "$best_line" ]]; then
    printf "%-12s %10s\n" "$policy" "FAILED"
    continue
  fi

  thr=$(echo "$best_line" | grep -oE "throughput=[0-9.]+" | cut -d= -f2)
  p50=$(echo "$best_line" | grep -oE "p50=[0-9.]+" | cut -d= -f2)
  p95=$(echo "$best_line" | grep -oE "p95=[0-9.]+" | cut -d= -f2)
  p99=$(echo "$best_line" | grep -oE "p99=[0-9.]+" | cut -d= -f2)
  out=$(echo "$best_line" | grep -oE "stolen_out=[0-9]+" | cut -d= -f2)
  printf "%-12s %10s %12s %10s %10s %10s %8s\n" \
         "$policy" "$best_time" "$thr" "$p50" "$p95" "$p99" "$out"
done

echo
echo "Best of $REPS runs per policy. 'stolen' counts pipeline stages the origin handed"
echo "to another node. Latency percentiles are end-to-end per request (all four stages)."
