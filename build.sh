#!/bin/bash
# Reproducible build for FastEddy v5.0.0 on this node.
# The stock SRC/FEMAIN/Makefile does not pass CUDA/NetCDF include & lib paths,
# so we inject them via make variable overrides (Makefile left pristine).
#
# It also has an explicit "-lmpi" in TEST_LIBS. The local MPICH
# (/opt/wrf_dependencies/mpich) ships only static libs (libmpich.a, no libmpi.so),
# so "-lmpi" resolves to the SYSTEM OpenMPI (/lib/.../libmpi.so.40) instead. The
# resulting binary then runs OpenMPI's MPI_Init and aborts under `srun --mpi=pmi2`
# ("OMPI was not built with SLURM's PMI support"). The mpicc wrapper already adds
# -lmpich, so we DROP -lmpi from TEST_LIBS/TEST_CU_LIBS to statically link MPICH
# (matching the working /opt FastEddy binary, which has no dynamic MPI dependency).
set -e
source ~/workspace/fasteddy_env.sh   # sets CUDA_HOME, NCAR_ROOT_MPI, NETCDF_C_ROOT, PATH, LD_LIBRARY_PATH
cd "$(dirname "$0")/SRC/FEMAIN"

make "$@" \
  OTHER_INCLUDES="-I${NCAR_ROOT_MPI}/include -I${CUDA_HOME}/include -I${NETCDF_C_ROOT}/include" \
  TEST_LDFLAGS="-L. -L${CUDA_HOME}/lib64 -L${NETCDF_C_ROOT}/lib" \
  TEST_LIBS="-lm -lstdc++ -lcurand -lcudart -lnetcdf" \
  TEST_CU_LIBS="-lm -lcudart"

echo "=== build complete: $(ls -la "$(pwd)/FastEddy") ==="
