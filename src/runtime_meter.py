import time


class RuntimeMeter:
    def __init__(self):
        self._start = None
        self._end = None
        self.elapsed_seconds = None

    def start(self):
        self._start = time.perf_counter()
        self._end = None
        self.elapsed_seconds = None
        return self

    def stop(self):
        if self._start is None:
            return self
        self._end = time.perf_counter()
        self.elapsed_seconds = float(self._end - self._start)
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False
