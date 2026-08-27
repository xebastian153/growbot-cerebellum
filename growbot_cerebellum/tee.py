"""Mirror stdout into a run log under results/logs/ while a script runs."""
from __future__ import annotations
import sys


class Tee:
    """`sys.stdout = Tee(path)`: every write goes to the terminal and to `path`.

    One definition. There were four copies, and they had started to differ in whether
    `flush()` survived a closed file.
    """
    def __init__(self, path):
        self.f = open(path, "w")
    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)
    def flush(self):
        sys.__stdout__.flush()
        if not self.f.closed:
            self.f.flush()
