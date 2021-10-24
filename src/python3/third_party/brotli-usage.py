from brotli import brotli


def compress_decompress(data: bytes):
    with open("test.BR", 'wb') as f:
        content = brotli.compress(data, quality=11)
        f.write(content)
    with open("test.BR", 'rb') as f:
        data = brotli.decompress(f.read())
