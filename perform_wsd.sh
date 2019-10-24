#!/bin/bash
#BATCH --time=96:00:00
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

module load python/3.5.2
module load cuda10.0/toolkit
module load cuda10.0/blas
module load cuda10.0
module load cuDNN/cuda10.0/7.4

python run_bert_wsd.py --input_folder="example_files" --meanings_path="/var/scratch/mcpostma/BERT-setup/resources/semcor_wngt_averaged_synset_embeddings.p"  --naf_pos="N-V-G-A" --use_pos_in_candidate_selection="yes" --verbose=2
