#!/usr/bin/env python
import dask, time
import dask.array as da

x = da.random.random((100000, 2000), chunks=(10000, 2000))
t0 = time.time()

q, r = da.linalg.qr(x)
test = da.all(da.isclose(x, q.dot(r)))
test.compute() # compute(get=dask.threaded.get) by default

print(time.time() - t0)
