# logger.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from utils import make_dir


class Logger:
    """
    Minimal scientific logger:
      - writes one JSON per line to: <log_dir>/<name>.jsonl
      - keeps last seen values in-memory (useful for printing)
    """

    def __init__(self, log_dir, name= "train"):
        self.log_dir = make_dir(log_dir)
        self.name = name
        self.path = os.path.join(self.log_dir, f"{self.name}.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")
        self._last: Dict[str, Any] = {}
        self._t0 = time.time()

    def log(self, key, value, step = None) :
        self.log_dict({key: value}, step=step)

    def log_dict(self, d, step = None) :
        rec = dict(d)
        if step is not None:
            rec["step"] = int(step)
        rec["_time"] = time.time()
        rec["_elapsed"] = rec["_time"] - self._t0

        # update last cache (exclude internal fields)
        for k, v in d.items():
            self._last[k] = v

        self._fh.write(json.dumps(rec) + "\n")

    def last(self, key, default):
        return self._last.get(key, default)

    def flush(self) :
        self._fh.flush()

    def close(self) :
        try:
            self._fh.flush()
        finally:
            self._fh.close()