import numpy as np
from ruptures.base import BaseCost

class MyCost(BaseCost):
    """Custom cost for percentage difference between segment medians."""
    model = ""
    min_size = 2

    def fit(self, signal):
        """Set the internal parameter."""
        self.signal = signal
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end]."""
        segment = self.signal[start:end]
        segment_mean = np.mean(segment)
        absolute_diff = np.abs(segment - segment_mean)
        cost = np.sum(absolute_diff)
        return cost