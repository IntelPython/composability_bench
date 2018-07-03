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


from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from numpy.random import random
from numpy.random import seed
from numpy import empty
import sys
import numpy

n = 1024
m = 64
data_mt = random((n, n))

def bench(n):
    seed([777])
    #data = random((n, n))
    result = numpy.linalg.eig(data_mt)
    return result[0][0].real

def bench_mt(n):
    result = numpy.linalg.eig(data_mt)
    return result[0][0].real

def main():
    np = 1

    if len(sys.argv) > 1:
        np = int(sys.argv[1])
    print("Number of processes: ", np)

    pool_type = "multiprocessing"
    if len(sys.argv) > 2:
        pool_type = sys.argv[2]
    print("Pool type: ", pool_type)

    if pool_type == "multiprocessing":
        p = Pool(np)
        result = p.map(bench, [n for i in range(m)])
        s = 0
        for val in result:
            s = s + val
        print(s)
    elif pool_type == "concurrent":
        with ProcessPoolExecutor(np) as p:
            s = 0
            for val in p.map(bench, [n for i in range(m)]):
                s = s + val
            print(s)
    elif pool_type == "threading":
        p = ThreadPool(np)
        result = p.map(bench_mt, [n for i in range(m)])
        s = 0
        for val in result:
            s = s + val
        print(s)
    else:
        print("Unsupported pool type: " + pool_type)

if __name__ == '__main__':
    main()
