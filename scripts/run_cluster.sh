#!/usr/bin/env bash
# Launch a local HydraRT cluster: one coordinator plus N node processes.
#
# Node 1 submits the whole workload; the rest start idle and can only get work by
# stealing it over the network. Comparing --no-stealing against the default is the
# point of the exercise.
#
#   ./scripts/run_cluster.sh [num_nodes] [tasks] [task_cost] [--no-stealing]

set -uo pipefail
cd "$(dirname "$0")/.."

NUM_NODES=${1:-3}
TASKS=${2:-2000}
COST=${3:-20000}
STEAL_FLAG=""
if [[ "${4:-}" == "--no-stealing" ]]; then
  STEAL_FLAG="--no-stealing"
fi
# Workers per node. Keep NUM_NODES * WORKERS at or under the core count: these "nodes"
# are processes on one machine, so oversubscribing them just makes them fight for CPU.
WORKERS=${WORKERS:-4}

COORD_PORT=$((9000 + RANDOM % 500))
BIN=./build
OUT=$(mktemp -d)

cleanup() {
  # Kill anything still alive, then wait so the shell doesn't print job-control noise.
  kill "${COORD_PID:-}" ${NODE_PIDS[@]:-} 2>/dev/null
  wait 2>/dev/null
  rm -rf "$OUT"
}
trap cleanup EXIT

echo "=== HydraRT cluster: ${NUM_NODES} nodes, ${TASKS} tasks, cost=${COST} ${STEAL_FLAG:-(stealing on)} ==="

"$BIN/hydra_coordinator" --port "$COORD_PORT" --seconds 60 > "$OUT/coord.log" 2>&1 &
COORD_PID=$!
sleep 0.5

if ! kill -0 "$COORD_PID" 2>/dev/null; then
  echo "coordinator failed to start:"; cat "$OUT/coord.log"; exit 1
fi

NODE_PIDS=()
# Nodes 2..N are pure workers: they hold no work of their own and must steal to be useful.
for ((i = 2; i <= NUM_NODES; i++)); do
  "$BIN/hydra_node" --coordinator-port "$COORD_PORT" --workers "$WORKERS" \
      --label cpu --seconds 30 $STEAL_FLAG > "$OUT/node$i.log" 2>&1 &
  NODE_PIDS+=($!)
done

sleep 0.5

# Node 1 owns the workload.
"$BIN/hydra_node" --coordinator-port "$COORD_PORT" --workers "$WORKERS" --label cpu \
    --submit "$TASKS" --task-cost "$COST" $STEAL_FLAG > "$OUT/node1.log" 2>&1 &
SUBMIT_PID=$!
NODE_PIDS+=($SUBMIT_PID)

# Wait for the submitting node to finish its batch.
wait "$SUBMIT_PID"
SUBMIT_RC=$?

# Give the worker nodes a moment to flush their final RESULT lines, then stop them.
sleep 1
kill ${NODE_PIDS[@]} 2>/dev/null
sleep 1.5

echo
echo "--- coordinator ---"
grep -E "registered|listening" "$OUT/coord.log" || true

echo
echo "--- nodes ---"
cat "$OUT"/node*.log | grep -E "RESULT|submitting|complete|listening" || true

echo
echo "--- totals ---"
awk '
  /RESULT/ {
    for (i = 1; i <= NF; i++) {
      split($i, kv, "=")
      if (kv[1] == "executed")   executed += kv[2]
      if (kv[1] == "stolen_in")  stolen   += kv[2]
    }
    nodes++
  }
  END {
    printf "nodes reporting: %d\n", nodes
    printf "tasks executed across cluster: %d\n", executed
    printf "tasks moved over the network:  %d\n", stolen
  }
' "$OUT"/node*.log

exit $SUBMIT_RC
