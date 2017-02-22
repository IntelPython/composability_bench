DIR=$HOME/miniconda3
rm -r $DIR
mkdir -p $DIR
cd $DIR
curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $DIR -f \
    && rm ./Miniconda3-latest-Linux-x86_64.sh
#export ACCEPT_INTEL_PYTHON_EULA=yes
CONDA=$DIR/bin/conda

$CONDA create -y -n intel3 -c intel python=3 numpy scipy scikit-learn tbb dask numba
