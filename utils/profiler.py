import os

ENABLE_PROFILING = os.environ.get("ENABLE_PROFILING") == "1"

if ENABLE_PROFILING:
    from line_profiler import profile
else:
    # No-op decorator
    def profile(func):
        return func
