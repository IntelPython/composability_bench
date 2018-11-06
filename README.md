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
Blog: https://software.intel.com/en-us/blogs/2016/04/04/unleash-parallel-performance-of-python-programs

Paper: "[Composable Multi-Threading and Multi-Processing for Numeric Libraries](http://conference.scipy.org/proceedings/scipy2018/pdfs/anton_malakhov.pdf)" by Anton Malakhov, David Liu, Anton Gorshkov, Terry Wilmarth. Proceedings of the 17th Python in Science Conference (SciPy 2018), Austin, Texas (July 9 - 15, 2018). DOI 10.25080/Majora-4af1f417-003
