"""Report rows that teach.

The same contract as MicroscopeVuilder's rule results: a row never says only "fine".
It carries the value, the governing equation with the user's own numbers in it and,
when something is off, what to change. The audience will check these against their
own microscope, so the explanation is the product and the status is incidental.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    INFO = "info"


@dataclass(frozen=True)
class Row:
    name: str
    summary: str
    equation: str = ""
    status: Status = Status.INFO
    remedy: str = ""

    @property
    def ok(self) -> bool:
        return self.status is not Status.FAIL

    def format(self) -> str:
        mark = {
            Status.PASS: "PASS",
            Status.WARN: "WARN",
            Status.FAIL: "FAIL",
            Status.INFO: " -- ",
        }[self.status]
        line = f"[{mark}] {self.name}: {self.summary}"
        if self.equation:
            line += f"\n         {self.equation}"
        if self.remedy and self.status in (Status.WARN, Status.FAIL):
            line += f"\n         remedy:  {self.remedy}"
        return line


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)

    def add(self, row: Row) -> None:
        self.rows.append(row)

    def failures(self) -> list[Row]:
        return [r for r in self.rows if r.status is Status.FAIL]

    def warnings(self) -> list[Row]:
        return [r for r in self.rows if r.status is Status.WARN]

    def format(self) -> str:
        return "\n".join(r.format() for r in self.rows)
