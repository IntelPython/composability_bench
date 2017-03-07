# Composability Benchmarks
Show effects of over-subscription and ways to fix that

## TBB module
Install TBB module for Python in order to evaluate benefits of composable multithreading
```
conda install -c intel tbb
```
Alternatively, run `set_python_envs.sh` to set up environment with components of Intel Distribution for Python

## Runing
Effects are visible on big enough machine with 32 and more cores.
Run in two modes: with and without `-m TBB` switch for Python. For example:

```
python -m TBB collab_filt.py
python -m TBB dask_bench2.py
```

## See also
https://software.intel.com/en-us/blogs/2016/04/04/unleash-parallel-performance-of-python-programs
