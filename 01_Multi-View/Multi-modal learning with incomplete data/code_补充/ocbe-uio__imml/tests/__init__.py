import builtins
from _pytest import monkeypatch

original_import = builtins.__import__
def mock_import(name, *args):
    if name in ["oct2py"]:
        raise ImportError("oct2py not installed")
    return original_import(name, *args)

def mock_test():
    monkeypatch.setattr(builtins, "__import__", mock_import)
