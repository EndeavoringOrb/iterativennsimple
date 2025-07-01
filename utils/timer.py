import time


class Timer:
    def __init__(self, label=None):
        self.label = label
        self.start = time.perf_counter()
        self._stopped = False

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def stop(self):
        if not self._stopped:
            elapsed = time.perf_counter() - self.start
            if self.label:
                print(f"[{self.label}] Elapsed time: {elapsed:.6f} seconds")
            else:
                print(f"Elapsed time: {elapsed:.6f} seconds")
            self._stopped = True

    def __del__(self):
        self.stop()
