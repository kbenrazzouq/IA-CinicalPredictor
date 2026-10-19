
class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_data(self):
        return self.X, self.y

    def __len__(self):
        return len(self.X)
