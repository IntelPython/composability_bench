# Composability Benchmarks
Show effects of over-subscription and ways to fix that

## Installation
Install TBB and SMP modules for Python in order to evaluate benefits of composable multithreading
```
conda install -c intel tbb4py smp
```
Alternatively, run `set_python_envs.sh` to set up environment with components of Intel Distribution for Python

## Running
Effects are visible on big enough machine with 32 and more cores.
Run following modes:

```
python -m tbb collab_filt.py
python -m tbb dask_bench2.py
python -m smp collab_filt.py
python -m smp dask_bench2.py
```

## See also
https://software.intel.com/en-us/blogs/2016/04/04/unleash-parallel-performance-of-python-programs
