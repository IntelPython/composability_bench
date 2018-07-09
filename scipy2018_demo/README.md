# Composability demo for SciPy 2018

## Installation
Install TBB and SMP modules for Python along with MKL-enabled numpy
in order to evaluate benefits of composable multithreading
```
conda install -c intel mkl numpy tbb4py smp
```

## Runing
Effects are visible on big enough machine with 32 and more cores.

```
python demo.py
KMP_BLOCKTIME=0 python demo.py  # Align settings with GNU OpenMP
python -m tbb demo.py
python -m smp demo.py
python -m smp -c demo.py
```

## Jupyter notebook
For running on the same host, `start_jupyter.sh`.

For running a remote session from Windows machine, copy&edit `demo_config_example.bat` as `demo_config.bat`,
then run `start_jupyter.bat`. In Jupyter, open `composability_demo.ipynb` and follow instructions.

## See also
https://software.intel.com/en-us/blogs/2016/04/04/unleash-parallel-performance-of-python-programs
