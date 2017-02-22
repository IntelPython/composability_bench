#!/usr/bin/env python
import dask, timeit
import dask.array as da
import dask.multiprocessing
import numpy as np


class common_bench:
    sx, sy = 320000, 1000
    cx, cy = 10000,  1000

class dask_bench(common_bench):
    def setup(self):
        self.x = da.random.random((self.sx, self.sy), chunks=(self.cx, self.cy))

    def _bench(self, get):
        q, r = da.linalg.qr(self.x)
        test = da.all(da.isclose(self.x, q.dot(r)))
        test.compute(get=get)

    def time_threaded(self):
        self._bench(dask.threaded.get)

    def time_multiproc(self):
        self._bench(dask.multiprocessing.get)


class numpy_bench(common_bench):
    def setup(self):
        self.x = np.random.random((self.sx, self.sy))

    def time_pure(self):
        q, r = np.linalg.qr(self.x)
        test = np.allclose(self.x, q.dot(r))

print("Numpy  ", timeit.repeat('b.time_pure()', 'from __main__ import numpy_bench as B; b=B();b.setup()', number=1, repeat=3))
print("Dask-MT", timeit.repeat('b.time_threaded()', 'from __main__ import dask_bench as B; b=B();b.setup()', number=1, repeat=3))
print("Dask-MP", timeit.repeat('b.time_multiproc()', 'from __main__ import dask_bench as B; b=B();b.setup()', number=1, repeat=3))
