class Serializable:
    def load(self, path):
        ...

    def store(self, path):
        ...

    @property
    def store_under(self):
        ...
