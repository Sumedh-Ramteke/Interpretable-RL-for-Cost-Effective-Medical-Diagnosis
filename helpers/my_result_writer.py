class MyResultWriter:
    def __init__(self):
        self.count = 0

    def write_row(self, epinfo):
        # Lightweight: just count entries instead of accumulating dicts.
        # The full log is never consumed during training, only during analysis.
        self.count += 1

    def close(self):
        pass

    def clear(self):
        self.count = 0
