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
- Inference cost is **simulated first** (distributions keyed by request parameters), with
  real ONNX Runtime models swapped in later once the scheduling story is proven.
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
- [ ] Phase 9 — Adaptive scheduler
- [ ] Phase 10 — Fault tolerance (node death, task retry)
- [ ] Phase 11 — ML Inference Engine (capstone)
- [ ] Phase 12 — Real inference (ONNX Runtime)
- [ ] Phase 13 — Performance report

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

**Phase 9 — Adaptive scheduler.** Nodes report a capability profile (measured
per-`TaskType` throughput, or a configured GPU/CPU tag) alongside queue depth, steal
success rate, and network latency as scheduling signals — this is where GPU-preferring
placement for inference tasks gets implemented.

**Phase 10 — Fault tolerance.** Node death detection (missed heartbeats) + task
retry/resubmission, so a multi-stage request pipeline doesn't hang if a node dies
mid-flight.

**Phase 11 — ML Inference Engine.**
- Chained continuations: `Future<T>::then(fn)` so a request pipeline (a linear
  dependency chain) can be built without blocking a thread on `get()` between stages —
  today's `spawn`/`Future` only supports fork + two-way join.
- Job/Request Manager: builds the 4-stage task graph per request, chains stages via
  `then()`.
- Simulated inference workload: cost varies by image resolution / sequence length /
  model id, drawn from a distribution instead of a flat constant.
- Local HTTP API (e.g. `cpp-httplib`): submit endpoint returns a job id, status/result
  endpoint polls the Job Manager.
- Benchmark suite: static round-robin vs random work-stealing vs HydraRT adaptive
  scheduling, under skewed multi-request-type load across the real multi-node cluster
  from Phases 7-8 — throughput, p95/p99 latency, worker utilization/idle time, steal
  overhead.

**Phase 12 — Real inference.** Swap ONNX Runtime in for at least one workload (e.g.
image classification) once the scheduling story is proven on simulated costs.

**Phase 13 — Performance report.** Write up the static vs random vs adaptive comparison
with real measured numbers.
