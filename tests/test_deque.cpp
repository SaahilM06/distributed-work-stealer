#include "runtime/WorkDeque.hpp"
#include <cassert>
#include <cstdio>

static void test_push_pop() {
    WorkDeque q;
    Task t1; t1.task_id = 1;
    Task t2; t2.task_id = 2;

    q.push(t1);
    q.push(t2);
    assert(q.size() == 2);

    Task out;
    assert(q.pop(out) && out.task_id == 1);
    assert(q.pop(out) && out.task_id == 2);
    assert(q.size() == 0);
    std::printf("PASS test_push_pop\n");
}

static void test_pop_empty() {
    WorkDeque q;
    Task out;
    assert(!q.pop(out));
    std::printf("PASS test_pop_empty\n");
}

int main() {
    test_push_pop();
    test_pop_empty();
    std::printf("All tests passed.\n");
    return 0;
}
