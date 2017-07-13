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


# http://software.intel.com/en-us/intel-mkl
# https://code.google.com/p/numexpr/wiki/NumexprVML

from __future__ import print_function
import datetime
import sys
#from scipy import stats
import numpy as np
#import numexpr as ne
import time
import gc
import os.path
#import cPickle as pickle
import os
import argparse
from pprint import pprint
import logging

data_dir = './'

tests = ("qr", "eig", "svd", "inv", "cholesky", "dgemm") #, "det" - gives warning

log = logging.getLogger(__name__)

def prepare_default(N=100, dtype=np.double):
    return ( np.asarray(np.random.rand(N, N), dtype=dtype), )
    #return toc/trials, (4/3)*N*N*N*1e-9, times

def prepare_eig(N=100, dtype=np.double):
    N/=4
    return ( np.asarray(np.random.rand(int(N), int(N)), dtype=dtype), )

def prepare_svd(N=100, dtype=np.double):
    N/=2
    return ( np.asarray(np.random.rand(int(N), int(N)), dtype=dtype), False )

#det:    return toc/trials, N*N*N*1e-9, times

def kernel_dot(A, B):
    """
    Dot product
    """
    np.dot(A, B)

def prepare_dot(N=100, dtype=np.double):
    N=N*N*10
    A = np.asarray(np.random.rand(int(N)), dtype=dtype)
    return (A, A)
    #return 1.0*toc/(trials), 2*N*N*N*1e-9, times

def kernel_ivi(A, B):
    """
    Collaborative filtering
    """
    A.dot(B)

def prepare_ivi(N=100, dtype=np.double):
    A = np.random.rand(3260, 3260)
    B = np.random.rand(3260, 3000)
    return (A, B)
    #return 1.0*toc/(trials), 2*N*N*N*1e-9, times

def prepare_dgemm(N=100, trials=3, dtype=np.double):
    LARGEDIM = int(N*2)
    KSIZE = int(N/2)
    A = np.asarray(np.random.rand(LARGEDIM, KSIZE), dtype=dtype)
    B = np.asarray(np.random.rand(KSIZE, LARGEDIM), dtype=dtype)
    return (A, B)

def kernel_dgemm(A, B):
    """
    DGEMM
    """
    A.dot(B)

def prepare_cholesky(N=100, dtype=np.double):
    N = int(N*2)
    A = np.asarray(np.random.rand(N, N), dtype=dtype)
    return ( A*A.transpose() + N*np.eye(N), )
    #return toc/trials, N*N*N/3.0*1e-9, times

#inv:    return toc/trials, 2*N*N*N*1e-9, times


##################################################################################


class tbb_job:
    def __init__(self, i, body):
        self._i = i
        self._body = body
    def __call__(self):
        self._body(self._i)

def run_tbb(n, body):
    """PlainTBB"""
    import TBB
    pool = TBB.task_group()
    global nested_tbb
    if 'nested_tbb' not in globals():
        log.debug("Creating TBB task_group")
        nested_tbb = TBB.task_arena()
    for i in n:
        b = tbb_job(i, body)
        pool.run(b) #, nested_tbb)
    pool.wait()

def run_tbbpool(n, body):
    """TBB.Pool"""
    from TBB import Pool
    global reused_pool, numthreads
    if 'reused_pool' not in globals():
        log.debug("Creating TBB.Pool(%s)" % numthreads)
        reused_pool = Pool(int(numthreads))
    reused_pool.map(body, n)

def run_tp(n, body):
    """ThreadPool.map"""
    from multiprocessing.pool import ThreadPool
    global reused_pool, numthreads
    if 'reused_pool' not in globals():
        log.debug("Creating ThreadPool(%s)" % numthreads)
        reused_pool = ThreadPool(int(numthreads))
    reused_pool.map(body, n)

def run_pp(n, body):
    """Process Pool.map"""
    from multiprocessing.pool import Pool
    global reused_pool, numthreads
    global args
    if 'reused_pool' not in globals():
        log.debug("Creating Pool(%s)" % numthreads)
        reused_pool = Pool(int(numthreads))
    reused_pool.map(body, n)

def run_tpaa(n, body):
    """ThreadPool.apply_async"""
    from multiprocessing.pool import ThreadPool
    global reused_pool, numthreads
    if 'reused_pool' not in globals():
        log.debug("Creating ThreadPool(%s) for apply_async()" % numthreads)
        reused_pool = ThreadPool(int(numthreads))
    reused_pool.map(body, range(n))
    wait_list = []
    for i in n:
        b = tbb_job(i, body)
        a = reused_pool.apply_async(b)
        wait_list.append(a)
    for a in wait_list:
        a.wait()

def run_seq(n, body):
    """Sequential"""
    for i in n:
        body(i)

def empty_work(i):
    pass

class body:
    def __init__(self, trials):
        self.trials = trials

    def __call__(self, i):
        global args, kernel, out
        for j in range(self.trials):
            t_start = time.time()
            kernel(*args[i])
            out[i,j] = time.time() - t_start

def bench_on(runner, sym, Ns, trials, dtype=None):
    global args, kernel, out, mkl_layer
    prepare = globals().get("prepare_"+sym, prepare_default)
    kernel  = globals().get("kernel_"+sym, None)
    if not kernel:
       kernel = getattr(np.linalg, sym)
    out_lvl = runner.__doc__.split('.')[0].strip()
    func_s  = kernel.__doc__.split('.')[0].strip()
    log.debug('Preparing input data for %s (%s).. ' % (sym, func_s))
    args = [prepare(int(i)) for i in Ns]
    it = range(len(Ns))
    # pprint(Ns)
    out = np.empty(shape=(len(Ns), trials))
    b = body(trials)
    tic, toc = (0, 0)
    log.debug('Warming up %s (%s).. ' % (sym, func_s))
    runner(range(1000), empty_work)
    kernel(*args[0])
    runner(range(1000), empty_work)
    log.debug('Benchmarking %s on %s: ' % (func_s, out_lvl))
    gc_old = gc.isenabled()
#    gc.disable()
    tic = time.time()
    runner(it, b)
    toc = time.time() - tic
    if gc_old:
        gc.enable()
    if 'reused_pool' in globals():
        del globals()['reused_pool']

    #calculate average time and min time and also keep track of outliers (max time in the loop)
    min_time = np.amin(out)
    max_time = np.amax(out)
    mean_time = np.mean(out)
    stdev_time = np.std(out)

    #print("Min = %.5f, Max = %.5f, Mean = %.5f, stdev = %.5f " % (min_time, max_time, mean_time, stdev_time))
    #final_times = [min_time, max_time, mean_time, stdev_time]

    print('## %s: Outter:%s, Inner:%s, Wall seconds:%f\n' % (sym, out_lvl, mkl_layer, float(toc)))
    return out


if __name__ != '__main__':
    print("not running as main? ", __name__)
else:

    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--threads', required=True, help="append number of threads used in benchmark to output resuts file")
    parser.add_argument('--parallel', required=False, default='tp', help="Specify outermost parallelism")
    parser.add_argument('--test', required=False, help="Run specified tests, comma-separated.")
    args = parser.parse_args()

    global numthreads, mkl_layer
    numthreads = args.threads
    runner_name= 'run_'+args.parallel
    if runner_name not in globals():
        print('--parallel=', args.parallel, " is not implemented, running sequential tests")
        runner = run_seq
    else:
        runner = globals()[runner_name]

    mkl_layer = os.environ.get('MKL_THREADING_LAYER')
    log.debug('MKL_THREADING_LAYER = ', mkl_layer)
    log.debug('MKL_NUM_THREADS = ', os.environ.get('MKL_NUM_THREADS'))
    log.debug('OMP_NUM_THREADS = ', os.environ.get('OMP_NUM_THREADS'))
    log.debug('KMP_AFFINITY = ', os.environ.get('KMP_AFFINITY'))


    trials = 3
    dtype = np.double
    log.debug('Number of iterations:', trials)

    #Ns = [600,1000,2000,5000]*10
    Ns = [1500,3000]*17
    #Ns = [5000]*3 + [512]*207
    #Ns = [10]*100

    if args.test:
        tests = args.test.split(",")

    for sym in tests:
        bench_on(runner, sym, Ns, trials)


