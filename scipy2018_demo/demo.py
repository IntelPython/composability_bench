import time, numpy as np
from multiprocessing.pool import ThreadPool

p = ThreadPool()
data = np.random.random((2000, 2000))

for j in range(3):
    t0 = time.time()
    p.map(np.linalg.qr, [data for i in range(10)])
    print(time.time() - t0)
