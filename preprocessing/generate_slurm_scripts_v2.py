# generate_slurm_scripts.py

def generate_slurm_scripts():
    slurm_template = """#!/bin/bash
  
#SBATCH -n 1
#SBATCH -o out-{year}-%j
#SBATCH -e eo-{year}-%j
#SBATCH -p huce_ice
#SBATCH --contiguous
#SBATCH --mail-type=END
#SBATCH --mail-user=zeyuan_hu@fas.harvard.edu
#SBATCH -J create_data_{year} # job name
#SBATCH -t 60
#SBATCH --mem-per-cpu=128000
#SBATCH --no-requeue

cd /n/home00/zeyuanhu/ClimCorrector
source activate /n/holylfs04/LABS/kuang_lab/Lab/kuanglfs/zeyuanhu/mamba_env/climcorr

python preprocessing/preprocess_climcorr_train_data_v2.py {year}
"""
    # Path to save SLURM scripts
    slurm_path = "/Users/zeyuanhu/Documents/research/ClimCorrector/preprocessing/slurm_v2_preprocessing/"
    os.makedirs(slurm_path, exist_ok=True)

    # Generate scripts for each year
    for year in range(1980, 2020):
        script_content = slurm_template.format(year=year)
        script_filename = f"{slurm_path}preprocess_{year}.sh"
        with open(script_filename, "w") as f:
            f.write(script_content)

    print(f"SLURM scripts saved to {slurm_path}")

if __name__ == "__main__":
    import os
    generate_slurm_scripts()