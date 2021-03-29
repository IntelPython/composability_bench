#!/usr/bin/env python
# Copyright (c) 2017, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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

    def _bench(self, sch):
        q, r = da.linalg.qr(self.x)
        test = da.all(da.isclose(self.x, q.dot(r)))
        test.compute(scheduler=sch)

    def time_threaded(self):
        self._bench('threads')

    def time_multiproc(self):
        self._bench('processes')


class numpy_bench(common_bench):
    def setup(self):
        self.x = np.random.random((self.sx, self.sy))

    def time_pure(self):
        q, r = np.linalg.qr(self.x)
        test = np.allclose(self.x, q.dot(r))

print("Warning: it takes minutes to complete..")
print("Numpy  ", timeit.repeat('b.time_pure()', 'from __main__ import numpy_bench as B; b=B();b.setup()', number=1, repeat=3))
print("Dask-MT", timeit.repeat('b.time_threaded()', 'from __main__ import dask_bench as B; b=B();b.setup()', number=1, repeat=3))
#print("Dask-MP", timeit.repeat('b.time_multiproc()', 'from __main__ import dask_bench as B; b=B();b.setup()', number=1, repeat=3))
