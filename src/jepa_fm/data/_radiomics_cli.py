"""Console entry point: one BLAS/OpenMP thread per worker, set before numpy loads.

Doing this inside main() is too late: the runtimes read these at import time and
forked workers inherit the parent's thread pool, which measured ~13x slower
(0.57 vs 7.5 lesions/s on 8 workers).
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from jepa_fm.data.radiomics import main  # noqa: E402

if __name__ == "__main__":
    main()
