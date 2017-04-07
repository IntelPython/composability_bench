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


import sys
import time
import timeit
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import random
import argparse
import numba

number_of_users = 400000
features = 3260
chunk = 10000

try:
    import numpy.random_intel as rnd
    numpy_ver="intel"
except:
    import numpy.random as rnd
    numpy_ver="std"

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--math', required=False, default='dask_numba', help="Select computation approach: numpy, dask, numpy_numba, dask_numba")
parser.add_argument('--features', required=False, default=features, help="Number of features to process")
parser.add_argument('--users', required=False, default=number_of_users, help="Number of users to process")
parser.add_argument('--chunk', required=False, default=chunk, help="Chunk size for splitting dask arrays")
parser.add_argument('--verbose', '-v', required=False, default=False, help="show progress information")
parser.add_argument('--prefix', required=False, default="@", help="Prepend result output with this string")
args = parser.parse_args()

features = int(args.features)
number_of_users = int(args.users)
chunk = int(args.chunk)

print("Generating fake similarity")
#topk = da.random.normal(size=(features, features), chunks=(features, features)).compute()
topk = rnd.normal(size=(features, features))
t = da.from_array(topk, chunks=(features, features))

print("Generating fake user data")
#users = da.random.normal(size=(features, number_of_users), chunks=(features, chunk)).compute()
#users = rnd.normal(size=(features, number_of_users))
users = np.zeros(shape=(features, number_of_users), dtype=np.float64)
objects_idx = np.arange(features)
rated = rnd.randint(0, 10, size=number_of_users, dtype=np.int32)
for user in range(number_of_users):
    rnd.shuffle(objects_idx)
    items_rated = rated[user]
    users[objects_idx[:items_rated], user] = rnd.randint(1, 5, size=items_rated, dtype=np.int32)

u = da.from_array(users, chunks=(features, chunk), name=False)

def run_numpy():
    x = topk.dot(users)
    x = np.where(users>0, 0, x)
    return x.argmax(axis=0)


def run_dask():
    x = t.dot(u)
    x = da.where(u>0, 0, x)
    r = x.argmax(axis=0)
    return r.compute()


@numba.guvectorize('(f8[:],f8[:],i4[:])', '(n),(n)->()', nopython=True, target="parallel")
def recommendation(x, u, r):
    maxx = x[0]
    r[0] = -1
    for i in range(x.shape[0]):
        if u[i] == 0 and maxx < x[i]: # if user has no rank for the item
           maxx = x[i]
           r[0] = i


def run_numpy_numba():
    x = topk.dot(users)
    return recommendation(x, users)


def run_dask_numba():
    x = t.dot(u)
    r = da.map_blocks(recommendation, x, u, drop_axis=0)
    return r.compute()


# ======================

# ======================

runner_name= 'run_'+args.math
if runner_name not in globals():
    print('--math=', args.math, " is not implemented, running numpy")
    runner = run_numpy
else:
    runner = globals()[runner_name]

if args.verbose:
    ProgressBar().register()

print("Running recommendation system")
for i in range(3):
    tic = time.time()
    r = runner()
    toc = time.time()
    time_diff = toc - tic
    if args.verbose:
        print("Result shape: ", r.shape, " strides: ", r.strides, " ", r)

    print("%s run=%d numpy=%s users=%d math=%s, chunk=%d in %.2f sec, %f users/sec" % \
         (str(args.prefix), i, numpy_ver, number_of_users, args.math, chunk, time_diff,
          float(number_of_users)/time_diff))
    sys.stdout.flush()
