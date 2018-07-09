#. activate scipy2018
pwd
jupyter kernelspec install python-tbb/ --sys-prefix
jupyter kernelspec install python-smp/ --sys-prefix
jupyter notebook --notebook-dir=`pwd` --NotebookApp.token=Default
