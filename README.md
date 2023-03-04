# Composability Benchmarks
Show effects of oversubscription and ways to fix that

## Installation
Install TBB and SMP modules for Python in order to evaluate composable multithreading:
```
conda install -c intel tbb4py smp
```
If not sure how to do this, just run `set_python_envs.sh` to set up conda environment with all the necessary components of Intel Distribution for Python and follow instructions for environment activation, e.g. `conda activate intel3`.

## Running
Effects are visible on big enough machine with 32 and more cores.
Run following modes for example:

```
python collab_filt.py
python dask_bench2.py
python -m tbb collab_filt.py
python -m tbb dask_bench2.py
python -m smp collab_filt.py
python -m smp dask_bench2.py
```

## Composability Modes
There are the folloing composability modes for testing:

### `-m tbb`
Enables TBB threading for MKL, Numpy, Dask, Python's multiprocessing.ThreadPool

### `-m tbb --ipc`
Same as `-m tbb` but also enables interprocess coordination for multiprocessing applications.

### `-m smp`
Statically allocates CPU resources between the nested parallel regions using affinity masks and OpenMP API. Supports both multithreading and multiprocessing parallelism.

### `-m smp -o`
Enables `KMP_COMPOSABILITY=mode=counting` for Intel OpenMP runtime when parallel regions are ordered using a semaphore. Supports both multithreading and multiprocessing parallelism.

## See also

Paper: "[Composable Multi-Threading and Multi-Processing for Numeric Libraries](http://conference.scipy.org/proceedings/scipy2018/pdfs/anton_malakhov.pdf)" by Anton Malakhov, David Liu, Anton Gorshkov, Terry Wilmarth. Proceedings of the 17th Python in Science Conference (SciPy 2018), Austin, Texas (July 9 - 15, 2018). DOI 10.25080/Majora-4af1f417-003
