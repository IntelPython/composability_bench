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
ENAME=scipy2018
log='echo'
mkdir -p $DIR
[ -x $CONDA ] || (
    $log "== Installing miniconda =="
    pushd $DIR
    [ -f Miniconda3-latest-Linux-x86_64.sh ] || curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $DIR -f
    popd
    [ -x $CONDA ] || exit 1
)
[ -d $DIR/envs/$ENAME ] || (
    $log "== Installing environment =="
    $CONDA create -y -n $ENAME -c intel python=3 numpy tbb4py smp dask || exit 1
)
source $DIR/bin/activate $ENAME || exit 1
LIBIOMP=$DIR/envs/$ENAME/lib/libiomp5.so
[ -f $LIBIOMP ] || exit 1
if [ `strings $LIBIOMP | grep -c KMP_COMPOSABILITY` != 0 ]; then
   compe="KMP_COMPOSABILITY=mode=exclusive"
   if [ `strings $LIBIOMP | grep -c "Using exclusive mode instead"` == 0 ]; then
      compc="KMP_COMPOSABILITY=mode=counting"
   fi
   $log "$compc $compe support detected for OpenMP!"
elif [ `strings $LIBIOMP | grep -c KMP_FOREIGN_THREAD_LOCK` != 0 ]; then
   $log "Limited composable OpenMP support detected! (Deprecated interface)"
   compe="KMP_FOREIGN_THREAD_LOCK=1"
else
   $log "New OpenMP composability interface is not available in this environment"
fi

#log=':'
#set -x
: >results.csv
for f in numpy_sl_mt dask_sh_mt numpy_dl_mt dask_dh_mt; do
  echo == $f
  (
    $log "#Default"
    KMP_BLOCKTIME=0 python $f.py
    $log "#OMP=1"
    OMP_NUM_THREADS=1 python $f.py
    $log "#SMP"
    python -m smp -f 1 $f.py
    $log "#TBB"
    python -m tbb $f.py
    if [ ! -z $compe ]; then
      $log "#OMP=exclusive"
      env $compe python $f.py
    fi
    if [ ! -z $compc ]; then
      $log "#OMP=counting"
      env $compc python $f.py
    fi
  ) |& tee $f.log
  sed -z 's/\n/ /g;s/#/\n/g' $f.log | sed "s/^/$f /;y/ /,/" >>results.csv
done
