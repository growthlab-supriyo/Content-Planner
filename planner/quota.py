"""Online free-run limit: each visitor gets FREE_RUNS_PER_DAY runs on the owner's key, then uses their own.

Counts live in memory (they reset if the app restarts) and are keyed by a hash of the visitor's IP
and today's date, so no IP address is stored. A total daily cap protects the owner's free quota.
"""
from __future__ import annotations

import hashlib
import threading
from datetime import date

from .config import FREE_RUNS_PER_DAY, OWNER_KEY_DAILY_CAP


class Quota:
    def __init__(self, per_visitor: int = FREE_RUNS_PER_DAY, daily_cap: int = OWNER_KEY_DAILY_CAP):
        self.per_visitor, self.daily_cap = per_visitor, daily_cap
        self._lock = threading.Lock()
        self._day = None
        self._counts: dict[str, int] = {}
        self._total = 0

    def _roll(self):
        today = date.today().isoformat()
        if self._day != today:
            self._day, self._counts, self._total = today, {}, 0

    @staticmethod
    def visitor_id(ip: str | None) -> str:
        return hashlib.sha256(f"{ip or 'unknown'}|{date.today()}".encode()).hexdigest()[:16]

    def remaining(self, vid: str) -> int:
        with self._lock:
            self._roll()
            mine = self.per_visitor - self._counts.get(vid, 0)
            return max(0, min(mine, self.daily_cap - self._total))

    def use(self, vid: str) -> bool:
        with self._lock:
            self._roll()
            if self._total >= self.daily_cap or self._counts.get(vid, 0) >= self.per_visitor:
                return False
            self._counts[vid] = self._counts.get(vid, 0) + 1
            self._total += 1
            return True

    def refund(self, vid: str) -> None:
        """Give a free run back when it failed before producing anything (e.g. a provider error)."""
        with self._lock:
            self._roll()
            if self._counts.get(vid, 0) > 0:
                self._counts[vid] -= 1
                self._total = max(0, self._total - 1)

    def cap_reached(self) -> bool:
        with self._lock:
            self._roll()
            return self._total >= self.daily_cap
