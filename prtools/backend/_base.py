
class BackendLibrary:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)
