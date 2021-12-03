v = memoryview(b"hello,world")
print(bytes(v[1:4]))

import time

for n in (100000, 200000, 300000, 400000):
    data = b'x' * n
    start = time.time()
    b = data
    while b:
        b = b[1:]
    print(f'     bytes {n} {time.time() - start:0.3f}')

for n in (100000, 200000, 300000, 400000):
    data = b'x' * n
    start = time.time()
    b = memoryview(data)
    while b:
        b = b[1:]
    print(f'memoryview {n} {time.time() - start:0.3f}')
