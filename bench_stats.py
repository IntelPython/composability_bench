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

data_dir = './'

tests = ("qr", "eig", "svd", "det", "inv", "cholesky", "dgemm")

def prepare_default(N=100, dtype=np.double):
    return ( np.asarray(np.random.rand(N, N), dtype=dtype), )
    #return toc/trials, (4/3)*N*N*N*1e-9, times

def prepare_eig(N=100, dtype=np.double):
    N/=8
    return ( np.asarray(np.random.rand(N, N), dtype=dtype), )

def prepare_svd(N=100, dtype=np.double):
    N/=4
    return ( np.asarray(np.random.rand(N, N), dtype=dtype), False )

#det:    return toc/trials, N*N*N*1e-9, times

def kernel_dot(A, B):
    """
    Dot product
    """
    np.dot(A, B)

def prepare_dot(N=100, dtype=np.double):
    N=N*N*10
    A = np.asarray(np.random.rand(N), dtype=dtype)
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

def kernel_dgemm(A, B):
    """
    DGEMM
    """
    A.dot(B)

def prepare_dgemm(N=100, trials=3, dtype=np.double):
    LARGEDIM = 7000
    KSIZE = N
    A = np.asarray(np.random.rand(LARGEDIM, KSIZE), dtype=dtype)
    B = np.asarray(np.random.rand(KSIZE, LARGEDIM), dtype=dtype)
    return (A, B)
    #return 1.*toc/trials, 2E-9*LARGEDIM*LARGEDIM*KSIZE, times

def prepare_cholesky(N=100, dtype=np.double):
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
        print("Creating TBB task_group")
        nested_tbb = TBB.task_arena()
    for i in n:
        b = tbb_job(i, body)
        pool.run(b) #, nested_tbb)
    pool.wait()

def run_tbbpool(n, body):
    """TBB.Pool"""
    from TBB import Pool
    global tp_pool, numthreads
    if 'tp_pool' not in globals():
        print("Creating TBB.Pool(%s)" % numthreads)
        tp_pool = Pool(int(numthreads))
    tp_pool.map(body, n)

def run_tp(n, body):
    """ThreadPool.map"""
    from multiprocessing.pool import ThreadPool
    global tp_pool, numthreads
    if 'tp_pool' not in globals():
        print("Creating ThreadPool(%s)" % numthreads)
        tp_pool = ThreadPool(int(numthreads))
    tp_pool.map(body, n)

def run_tpaa(n, body):
    """ThreadPool.apply_async"""
    from multiprocessing.pool import ThreadPool
    global tp_pool, numthreads
    if 'tp_pool' not in globals():
        print("Creating ThreadPool(%s) for apply_async()" % numthreads)
        tp_pool = ThreadPool(int(numthreads))
    tp_pool.map(body, range(n))
    wait_list = []
    for i in n:
        b = tbb_job(i, body)
        a = tp_pool.apply_async(b)
        wait_list.append(a)
    for a in wait_list:
        a.wait()

def run_seq(n, body):
    """Sequential"""
    for i in n:
        body(i)

def bench_on(runner, sym, Ns, trials, dtype=None):
    prepare = globals().get("prepare_"+sym, prepare_default)
    kernel  = globals().get("kernel_"+sym, None)
    if not kernel:
       kernel = getattr(np.linalg, sym)
    out_lvl = runner.__doc__.split('.')[0].strip()
    func_s  = kernel.__doc__.split('.')[0].strip()
    print('Preparing input data for %s (%s).. ' % (sym, func_s))
    args = [prepare(i) for i in Ns]
    it = range(len(Ns))
    # pprint(Ns)
    out = np.empty(shape=(len(Ns), trials))
    def body(i):
        for j in range(trials):
            t_start = time.time()
            kernel(*args[i])
            out[i,j] = time.time() - t_start
    tic, toc = (0, 0)
    print('Warming up %s (%s).. ' % (sym, func_s))
    runner(range(1000), lambda i: True)
    kernel(*args[0])
    runner(range(1000), lambda i: True)
    print('Benchmarking %s on %s: ' % (func_s, out_lvl))
    gc_old = gc.isenabled()
#    gc.disable()
    tic = time.time()
    runner(it, body)
    toc = time.time() - tic
    if gc_old:
        gc.enable()

    #calculate average time and min time and also keep track of outliers (max time in the loop)
    min_time = np.amin(out)
    max_time = np.amax(out)
    mean_time = np.mean(out)
    stdev_time = np.std(out)

    print("Min = %.5f, Max = %.5f, Mean = %.5f, stdev = %.5f " % (min_time, max_time, mean_time, stdev_time))
    #final_times = [min_time, max_time, mean_time, stdev_time]

    #pprint(out)
    global mkl_layer
    print('## %s: Outter:%s, Inner:%s, Wall seconds:%f' % (sym, out_lvl, mkl_layer, float(toc)))
    #pprint(out)
    return out

def dump_data(data, data_dir, backend, algo, threads):
    filename = backend + '-' + algo + '-' + str(threads) + '.pkl'
    out_pickle = os.path.join(data_dir, filename)
#    with open(out_pickle,'w') as data_file:
#        pickle.dump(data, data_file)


def print_data(Ns, inputdata, execution_times, benchmark, backend, threads):
#    print("inputdata",inputdata)
    i = 0
    outfilename = backend + '-' + benchmark + '-' + threads + '-' + 'times.txt' 
    with open(outfilename, 'w') as data_file:
        for size in Ns:
            print('Benchmark = %s, Size = %d, Average GFlop/sec = %.5f' %  (benchmark, size, inputdata[i][1]))
            #final_times = [min_time, max_time, mean_time, stdev_time]
            if i == 0:
                data_file.write("#Threads = %s" % threads)
                data_file.write("#Python Distro = %s" % pydistro)
                data_file.write('#Benchmark,  Matrix Size,  Min Time,    Max Time,     Mean Time,    Stdev Time,  GFlops  \n' )
                data_file.write('%s, %15d, %14.5f, %11.5f,  %11.5f,  %11.5f, %11.5f \n' %  (benchmark, size, execution_times[Ns[i]][0],  execution_times[Ns[i]][1],  execution_times[Ns[i]][2],  execution_times[Ns[i]][3], inputdata[i][1]))
            else:
                #data_file.write('Benchmark = %15s, Size = %5d,Min Time %.5f, Max Time = %.5f, Mean Time = %.5f, Stdev Time = %.5f, GFlops = %.5f \n' %  (benchmark, size, execution_times[i][0],  execution_times[i][1],  execution_times[i][2],  execution_times[i][3], inputdata[i][1]))
                data_file.write('%s, %15d, %14.5f, %11.5f,  %11.5f,  %11.5f, %11.5f \n' %  (benchmark, size, execution_times[Ns[i]][0],  execution_times[Ns[i]][1],  execution_times[Ns[i]][2],  execution_times[Ns[i]][3], inputdata[i][1]))
            i = i + 1


if __name__ != '__main__':
    print("not running as main? ", __name__)
else:

    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--pydistro', required=False, default='unknown', help="prepend name for output results file")
    parser.add_argument('--threads', required=True, help="append number of threads used in benchmark to output resuts file")
    parser.add_argument('--parallel', required=False, default='tp', help="Specify outermost parallelism")
    parser.add_argument('--test', required=False, help="Run specified tests.")
    args = parser.parse_args()

    global numthreads, pydistro, mkl_layer
    pydistro = args.pydistro
    numthreads = args.threads
    runner_name= 'run_'+args.parallel
    if runner_name not in globals():
        print('--parallel=', args.parallel, " is not implemented, running sequential tests")
        runner = run_seq
    else:
        runner = globals()[runner_name]

    """
    #add to the path the distro
    if pydistro == 'intel':
        myalias = 'pyintel27'
    elif pydistro == 'anaconda':
        myalias = 'pyana27'
    elif pydistro == 'accelerate':
        myalias = 'pyanamkl'

    print('setting up in path %s' % myalias)
    os.system(myalias)
    """

    try:
        import mkl
        have_mkl = True
        backend = pydistro
        print("Running with MKL Acceleration")
    except ImportError:
        have_mkl = False
        myPythonVersion = sys.version
        if 'Anaconda' in myPythonVersion:
            backend = 'Anaconda'
        elif 'Anaconda' not in myPythonVersion:
            backend = pydistro
        print("Running with normal backends")

    #Set parameters for MKL and OMP
    
    #os.environ["MKL_NUM_THREADS"] = args.threads
    #os.environ["MKL_DYNAMICS"] = 'FALSE'
    #os.environ["OMP_NUM_THREADS"] = '1'
  
    #if numthreads == 32:
    #    os.environ['MKL_NUM_THREADS'] = '32'
    #else:
    #    os.environ['MKL_NUM_THREADS'] = '1'

    mkl_layer = os.environ.get('MKL_THREADING_LAYER')
    print('MKL_THREADING_LAYER = ', mkl_layer)
    print('MKL_NUM_THREADS = ', os.environ.get('MKL_NUM_THREADS'))
    print('OMP_NUM_THREADS = ', os.environ.get('OMP_NUM_THREADS'))
    print('KMP_AFFINITY = ', os.environ.get('KMP_AFFINITY'))


    trials = 3
    dtype = np.double
    print('Number of iterations:', trials)
 
    #Ns = np.array([23000])
    #det_data = bench(test_det, Ns, trials)
    #dump_data(det_data, data_dir, backend, 'Determinant')


    #Ns = np.array([10000, 15000, 20000])
    #Ns = [600,1000,2000,5000]*10
    Ns = [1024,3000]*17
    #Ns = [5000]*3 + [512]*207
    #Ns = [10]*100
    #For PSF
    #Ns =  np.array([15000,]*10)
    #Ns =  np.array([100])

    if args.test:
        tests = args.test.split(",")
    
    for sym in tests:
        bench_on(runner, sym, Ns, trials)
    

