.PHONY: all build build-debug test bench clean

all: build

build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo > /dev/null
	cmake --build build

# Debug build: asserts are live (RelWithDebInfo defines -DNDEBUG, which silently
# no-ops every assert()) and -fsanitize=thread is wired in (see CMakeLists.txt) to
# catch data races — this is the build that actually verifies correctness.
build-debug:
	cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug > /dev/null
	cmake --build build-debug

test: build-debug
	./build-debug/test_deque
	./build-debug/test_runtime
	./build-debug/test_protocol
	./build-debug/test_cluster

bench: build
	./build/hydra_bench

clean:
	rm -rf build build-debug
