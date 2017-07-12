#!/bin/bash
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


DIR=$HOME/local/miniconda3
CONDA=$DIR/bin/conda
ENAME=intel3
alias log='echo'
shopt -s expand_aliases
mkdir -p $DIR
[ -x $CONDA ] || (
    log "== Installing miniconda =="
    pushd $DIR
    [ -f Miniconda3-latest-Linux-x86_64.sh ] || curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $DIR -f
    popd
    [ -x $CONDA ] || exit 1
)
[ -d $DIR/envs/$ENAME ] || (
    log "== Installing environment =="
    $CONDA create -y -n $ENAME -c intel python=3.5 numpy tbb smp dask || exit 1
)
source $DIR/bin/activate $ENAME || exit 1
LIBIOMP=$DIR/envs/$ENAME/lib/libiomp5.so
[ -f $LIBIOMP ] || exit 1
if [ `strings $LIBIOMP | grep -c KMP_COMPOSABILITY` != 0 ]; then
   if [`strings $LIBIOMP | grep -c "Using exclusive mode instead"` != 0]; then
      comp="KMP_COMPOSABILITY=mode=exclusive"
   else
      comp="KMP_COMPOSABILITY=mode=counting"
   fi
   log "$comp support detected for OpenMP!"
elif [ `strings $LIBIOMP | grep -c KMP_FOREIGN_THREAD_LOCK` != 0 ]; then
   log "Limited composable OpenMP support detected! (Deprecated interface)"
   comp="KMP_FOREIGN_THREAD_LOCK=1"
else
   log "New OpenMP composability interface is not available in this environment"
fi
alias log=':'
set -x
log "== Numpy (Static mode) =="
log "Default"
KMP_BLOCKTIME=0 python numpy_sl_mt.py
log "OMP_NUM_THREADS=1"
OMP_NUM_THREADS=1 python numpy_sl_mt.py
log "SMP"
python -m smp -f 1 numpy_sl_mt.py
log "TBB"
python -m tbb numpy_sl_mt.py
log "Composable OpenMP"
[ -z $comp ] || env $comp python numpy_sl_mt.py

log "== Dask (Static mode) =="
log "Default"
KMP_BLOCKTIME=0 python dask_sh_mt.py
log "OMP_NUM_THREADS=1"
OMP_NUM_THREADS=1 python dask_sh_mt.py
log "SMP"
python -m smp -f 1 dask_sh_mt.py
log "TBB"
python -m tbb dask_sh_mt.py
log "Composable OpenMP"
[ -z $comp ] || env $comp python dask_sh_mt.py

log "== Numpy (Dynamic mode) =="
log "Default"
KMP_BLOCKTIME=0 python numpy_dl_mt.py
log "OMP_NUM_THREADS=1"
OMP_NUM_THREADS=1 python numpy_dl_mt.py
log "SMP"
python -m smp -f 1 numpy_dl_mt.py
log "TBB"
python -m tbb numpy_dl_mt.py
log "Composable OpenMP"
[ -z $comp ] || env $comp python numpy_dl_mt.py

log "== Dask (Dynamic mode) =="
log "Default"
KMP_BLOCKTIME=0 python dask_dh_mt.py
log "OMP_NUM_THREADS=1"
OMP_NUM_THREADS=1 python dask_dh_mt.py
log "SMP"
python -m smp -f 1 dask_dh_mt.py
log "TBB"
python -m tbb dask_dh_mt.py
log "Composable OpenMP"
[ -z $comp ] || env $comp python dask_dh_mt.py
