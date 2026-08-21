# HydraRT Roadmap

HydraRT is a distributed adaptive work-stealing runtime in C++. The end goal is a
**distributed ML inference engine** built on top of it: clients submit inference jobs
(varying image size / sequence length / model), each job becomes a small task graph
(decode/tokenize → preprocess → inference → postprocess), and HydraRT distributes those
tasks across heterogeneous worker nodes (GPU vs CPU), stealing work to correct for
imbalance caused by unpredictable per-request cost. The runtime is the technically
interesting core; the inference engine is the workload that proves adaptive work-stealing
beats static assignment, backed by a real throughput/p95/p99-latency benchmark.

Scope decisions:
- Inference cost was **simulated first** (distributions keyed by request parameters),
  with a real ONNX Runtime model swapped in at Phase 12 once the scheduling story was
  proven. Both paths still work; the model is optional at build and run time.
- Heterogeneous workers are **real separate node processes/machines**, not worker-class
  tags simulated inside one process — so the inference engine depends on the runtime
  actually having a network layer (Phases 7-8) before it's meaningful.
- Job submission is a **local HTTP API**, not just an in-process benchmark driver.

## Status

- [x] Phase 0 — CMake infra, logging, config
- [x] Phase 1 — Global queue thread pool (baseline)
- [x] Phase 2 — Per-worker queues, round-robin
- [x] Phase 3 — Local work stealing
- [x] Phase 4 — Fork-join + Futures (`spawn`/`Future<T>`)
- [x] Phase 5 — Chase-Lev lock-free deque
- [x] Phase 6 — Metrics + tracing (p50/p95/p99, CSV)
- [x] Phase 7 — TCP node runtime + coordinator
- [x] Phase 8 — Remote work stealing
- [x] Phase 9 — Adaptive scheduler
- [x] Phase 10 — Fault tolerance (node death, task retry)
- [x] Phase 11 — ML Inference Engine (capstone)
- [x] Phase 12 — Real inference (ONNX Runtime)
- [x] Phase 13 — Performance report

## Phase details

**Phase 5 — Chase-Lev lock-free deque.** Replace the mutex-guarded `std::deque` inside
`WorkDeque` with a real lock-free Chase-Lev deque (owner-only push/pop on `bottom`,
CAS-based `steal` on `top`). Directly motivated by `results/phase3_bench.txt`: throughput
plateaus at ~2.39M tasks/sec across both 4 and 8 workers, the signature of lock contention
once steal traffic rises.

**Phase 6 — Metrics & tracing.** Per-task latency sampling with p50/p95/p99 and CSV
export — the instrumentation the eventual ML-engine benchmark suite depends on.
Each sample records submit/start/end timestamps, the executing worker, and whether the
task moved between workers, which decomposes into **queue wait** (the scheduling-quality
signal) and **execution time** (task granularity). Samples are sharded per worker so
recording never contends, mirroring the per-worker deque design. Tracing is toggleable
(`Runtime rt(n, /*enable_tracing=*/false)`) and costs ~3-10% throughput when on.
See `results/phase6_bench.txt` and `results/phase6_latency_*.csv`.

**Phase 7 — TCP node runtime + coordinator.** Done. A `Task` is now dual-mode: a
`std::function` closure (fast, local-only) *or* a `TaskType` tag plus a byte payload,
which is the only form that can cross the network — a closure captures pointers into one
process's memory and `std::function` is type-erased code that cannot be serialized. Every
node registers an identical handler table (`TaskRegistry`) mapping tag to code, so the
wire only carries tag + bytes. Around that: explicit little-endian encoding
(`net/Serialize.hpp`), length-prefixed framing over TCP (`net/Protocol.hpp` — a stream has
no message boundaries), a blocking socket wrapper (`net/Socket.hpp`), and a `Coordinator`
handling registration and heartbeats. Membership is request/response only: a node
heartbeats and gets the current node list back, so nothing is ever pushed and a missed
update is picked up on the next beat. Binaries: `hydra_coordinator`, `hydra_node`.

**Phase 8 — Remote work stealing.** Done. A node whose queue is below its worker count
picks a random peer and asks for work; the victim gives away half of its portable pool,
capped by what was requested (one task per round trip never moves enough load to pay for
the trip). Worker threads never touch the network — a dedicated steal thread pulls work
and a completion thread reports finished tasks home, because a blocking socket call on a
worker would stall that worker's whole share of the machine. Tasks handed out stay counted
as submitted at the origin until the thief reports completion, so `wait_all()` remains
correct across nodes.

Measured (`results/phase7_8_cluster.txt`, medians of 5 reps, 3 node processes on one
8-core M2): **1.73x** speedup with 2 workers/node and **1.36x** with 4 workers/node, with
25-38% of tasks crossing the network. Stealing helps whenever the origin node cannot use
the machine's full capacity alone; the stealing-on result lands close to a single-node
8-worker control, so the cluster recovers most of what shared memory would give for the
same cores, minus serialization and a TCP round trip per steal batch.

**Phase 9 — Adaptive scheduler.** Done. Victim selection became a policy knob
(`--policy none|random|load-aware|adaptive`) so the alternatives could be measured
rather than asserted. Nodes advertise a capability (`--label gpu` prefers the inference
task type) and a thief sends that preference with each steal request, so a victim hands
over the work the thief is fastest at; portable tasks are held in one pool per task type
so that selection is a lookup rather than a scan.

The measurement changed the design twice. Load figures originally travelled only on
300ms heartbeats, which is useless for a job that finishes sooner than that — the
victim's live queue depth now rides back on every steal response, including empty ones.
And the adaptive backoff originally used a lifetime success rate, which let startup
failures suppress stealing for an entire run; it is now exponential in *consecutive*
failures and resets on any success. Before those fixes the "smart" policies lost to
random stealing.

**Phase 10 — Fault tolerance.** Done. A node records every task it hands to a thief;
if no result comes back within `task_timeout_ms`, a reaper thread re-runs it locally.
Without this a peer that dies after a successful steal hangs the origin's `wait_all()`
forever, because the task is counted as submitted and the only thing that could ever
complete it is a message that is never coming. A late-arriving result for an
already-reaped task is ignored, so recovery cannot double-count. Semantics are
at-least-once, not exactly-once: a task may run on both the dead-looking peer and the
origin. Re-runs are made non-portable so the peer that just dropped one cannot steal it
straight back and drop it again. `NodeConfig::drop_completions` is a fault-injection
switch used by the test to simulate a node that takes work and vanishes.

**Phase 11 — ML Inference Engine.** Done. `hydra_inference` is a cluster node that also
serves HTTP: `POST /infer?model=text&seq=512`, `GET /status?id=N`, `GET /stats`.

Each request becomes a four-stage chain (decode → preprocess → infer → postprocess).
The stages are a dependency chain, not a fork-join, so each stage's task submits its
successor on completion instead of a thread blocking on a future in between — a request
in flight occupies no thread at all while it waits. `Future<T>::then()` exists for the
in-process case, but the distributed pipeline is driven by a task-completion observer on
the Runtime, which fires whether the stage ran locally or was stolen and run elsewhere.
The observer runs *before* the task is counted complete, so `wait_all()` cannot see a
momentarily balanced ledger and return with stages still to come.

Cost is simulated but deliberately skewed: transformer inference is quadratic in
sequence length, so two requests that look alike can differ by orders of magnitude.
That is what defeats any up-front split.

Measured (`results/phase11_inference_bench.txt`, best of 5, 3 nodes on one 8-core M2,
3000 requests): against static assignment, **2.07x throughput** (17.8k → 37.0k req/s)
and **52% lower p99 latency** (164ms → 78ms). Against random-victim stealing, 1.90x
throughput and 47% lower p99. Load-aware and adaptive tie — the capability affinity did
not beat plain load-awareness on this workload, and the results file says so.

**Phase 12 — Real inference.** Done. The inference stage runs a real MobileNetV2
forward pass through ONNX Runtime instead of a busy loop. `OnnxModel` wraps a single
shared `Ort::Session` — sessions are safe to `Run()` concurrently, and one per worker
would multiply the model's memory for no gain. ORT's own thread pool is pinned to one
thread: the runtime already has workers, and letting ORT spawn its own on top would
oversubscribe the machine and invalidate the measurements. Parallelism comes from
running many requests at once, not from splitting one.

Optional at build time (`-DHYDRA_WITH_ONNX=ON`, auto-detected) and at run time
(`--model path.onnx`); without either, everything falls back to simulated cost, so the
project still builds and every other result still stands on a machine with no model
runtime. Only the inference stage becomes real — the other three stages, and text
requests, stay simulated because the exported model is an image classifier.

    brew install onnxruntime
    python3 scripts/export_model.py models/mobilenet_v2.onnx
    MODEL=models/mobilenet_v2.onnx ./scripts/bench_inference.sh 400 3

Measured (`results/phase12_onnx_bench.txt`, best of 3, 400 requests): **2.45x
throughput** (204 → 500 req/s) and **60% lower p99 latency** (1939 → 780 ms) against
static assignment. Larger than the simulated figure (2.07x) simply because a real
forward pass is tens of milliseconds against a few hundred microseconds, so an idle
helper is worth much more.

The finding worth keeping: random, load-aware and adaptive all converge here (828 /
799 / 826 ms, within noise), where the simulated workload separated them sharply (154
vs 81 ms). When a single task is ~20ms of real model execution, any steal moves a lot
of work, so choosing the *best* victim matters far less than not being idle.
Sophisticated placement earns its keep on fine-grained tasks; on coarse ones it does
not.

**Phase 13 — Performance report.** Done: `results/phase11_inference_bench.txt`
(policy comparison, simulated cost), `results/phase12_onnx_bench.txt` (same comparison
with real model execution), `results/phase7_8_cluster.txt` (remote stealing) and
`results/phase6_bench.txt` (tracing overhead).
