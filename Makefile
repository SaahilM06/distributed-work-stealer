.PHONY: all build test bench clean

all: build

build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo > /dev/null
	cmake --build build

test: build
	./build/test_deque
	./build/test_runtime

bench: build
	./build/hydra_bench

clean:
	rm -rf build
