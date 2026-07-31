import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abc import ABC, abstractmethod
from typing import List


# https://pythonguides.com/python-interface/
class SignalStrategyInterface(ABC):
    @abstractmethod
    def tink_test_intervals(self, ticker: str, from_iso: str, to_iso: str, intervals: List[str], metric: str) -> str:
        """Find the best time-interval for strategy Tink"""
        pass
