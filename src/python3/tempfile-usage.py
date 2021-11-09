from tempfile import TemporaryDirectory

with TemporaryDirectory(suffix=None, prefix=None, dir=None) as dirname:
    print(dirname)
