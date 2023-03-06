#!/bin/bash

export KMP_BLOCKTIME=0

echo "Running balanced workload (dask_sh_mt) with high innermost parallelism"
echo "python dask_sh_mt.py"
python dask_sh_mt.py

echo "Running balanced workload (dask_sh_mt) with high innermost parallelism"
echo "env KMP_COMPOSABILITY=mode=counting python dask_sh_mt.py"
env KMP_COMPOSABILITY=mode=counting python dask_sh_mt.py

echo "Running balanced workload (numpy_sl_mp.py) with low innermost parallelism"
echo "python numpy_sl_mp.py 4"
python numpy_sl_mp.py 4

echo "Running balanced workload (numpy_sl_mp.py) with low innermost parallelism"
echo "python -m tbb numpy_sl_mp.py 4"
python -m tbb numpy_sl_mp.py 4

echo "Running unbalanced workload (dask_dh_mt.py) with high innermost parallelism"
echo "python dask_dh_mt.py"
python dask_dh_mt.py

echo "Running unbalanced workload (dask_dh_mt.py) with high innermost parallelism"
echo "env KMP_COMPOSABILITY=mode=counting python dask_dh_mt.py"
env KMP_COMPOSABILITY=mode=counting python dask_dh_mt.py

echo "Running unbalanced workload () with low innermost parallelism"
echo "python numpy_dl_mt.py"
python numpy_dl_mt.py

echo "Running unbalanced workload () with low innermost parallelism"
echo "python -m tbb numpy_dl_mt.py"
python -m tbb numpy_dl_mt.py

