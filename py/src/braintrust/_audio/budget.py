"""Conservative process-wide bound on retained recording source bytes."""

import threading


class ByteBudget:
    def __init__(self, maximum=64 * 1024 * 1024):
        self.maximum = maximum
        self.used = 0
        self.lock = threading.Lock()

    def reserve(self, size):
        with self.lock:
            if self.used + size > self.maximum:
                return False
            self.used += size
            return True

    def release(self, size):
        with self.lock:
            self.used -= size
            if self.used < 0:
                raise RuntimeError("Unbalanced audio budget release")


source_budget = ByteBudget()
