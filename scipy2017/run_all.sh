#!/bin/bash
log=echo
for i in `seq 10`; do
 $log === Run $i
 for f in numpy_sl_mp numpy_dl_mp numpy_sl_mt numpy_dl_mt; do
  $log == $f
  (for p in 88 64 44 32 22 16 8 4 2 1; do
    $log "#Default $p"
    KMP_BLOCKTIME=0 python $f.py $p
    $log "#SMP $p"
    python -m smp -f 1 $f.py $p
    if [[ "$f" =~ "_mp" ]]; then
      $log "#TBB-ipc $p"
      python -m tbb --ipc $f.py $p
    else
      $log "#TBB $p"
      python -m tbb -p $p $f.py $p
    fi
    $log "#OMP=counting $p"
    env KMP_COMPOSABILITY=mode=counting python $f.py $p
  done) |& tee $f.log
  sed -z 's/\n/ /g;s/#/\n/g' $f.log | sed "s/^/$f /;y/ /,/" >>scalability.csv
 done
done
