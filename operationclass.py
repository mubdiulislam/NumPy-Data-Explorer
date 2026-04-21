import numpy as np


class BranchSales:
    """One branch's sales data"""

    def __init__(self, data, name="Store"):
        self.data = data
        self.name = name

    def show(self):
        """Display basic info"""
        print(f"\n{self.name}: {self.data}")
        print(f"Months: {len(self.data)}")
        print(f"Total: ${np.sum(self.data):,}")
        print(f"Average: ${np.mean(self.data):.2f}")