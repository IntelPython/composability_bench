python -m smp bench_stats.py --parallel tp --threads 44 |& grep '##'
python -m tbb bench_stats.py --parallel tp --threads 44 |& grep '##'
KMP_BLOCK_TIME=0 python bench_stats.py --parallel tp --threads 44
#expected to fail with:
#OMP: Error #34: System unable to allocate necessary resources for OMP thread:
#OMP: System error #11: Resource temporarily unavailable
#OMP: Hint: Try decreasing the value of OMP_NUM_THREADS.
OMP_NUM_THREADS=1 python bench_stats.py --parallel tp --threads 44 |& grep '##'

python -m smp bench_stats.py --parallel pp --threads 44 --test svd  |& grep '##'
python -m tbb bench_stats.py --parallel pp --threads 44 --test svd  |& grep '##'
python bench_stats.py --parallel pp --threads 44 --test svd         |& grep '##'
