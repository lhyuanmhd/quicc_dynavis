#!/bin/bash -l
#SBATCH --job-name="Visu"
#SBATCH --account=project_465001528
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --output=slurm_out.%j
#SBATCH --error=slurm_err.%j

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load LUMI
module load PrgEnv-gnu
module load cray-python
module load cray-fftw
module load cray-hdf5-parallel
module load Boost
module load buildtools
module load craype-x86-milan

source /scratch/project_465001528/lhyuan/myenv/bin/activate

python --version
python -c "import h5py; print(h5py.__version__)"

#export PYTHONPATH=/users/burmannf/QuICC/Builds/lib64/python
#echo "[INFO] PWD=$(pwd)"
#echo "[INFO] Host=$(hostname)"
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"

# ---- ensure exactly one state file exists in this visu dir ----
statefile=$(ls -1 state*.hdf5 | head -n 1)
if [[ -z "${statefile}" ]]; then
  echo "[ERROR] No state*.hdf5 found in $(pwd)"
  exit 2
fi

echo "[INFO] Using ${statefile} -> state4Visu.hdf5"
rm -f state4Visu.hdf5
ln -s "${statefile}" state4Visu.hdf5

echo "[INFO] Running QuICC visu executable..."
srun /project/project_465001528/lhyuan/QuICC-Solver/build/Models/BoussinesqSphereInertialessDynamo/Executables/BoussinesqSphereInertialessDynamoImplicitVisu > run.log 2>&1

echo "[INFO] QuICC visu finished, start postprocess..."
bash ./postprocess_visu.sh >> run.log 2>&1

echo "[INFO] Done."


