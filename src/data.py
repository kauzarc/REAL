from dataclasses import dataclass


@dataclass
class Batch:
    inputs: list[str]
    labels: list[str]
