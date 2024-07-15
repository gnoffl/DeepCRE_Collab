### project name
#$ -N deepCRE_ssr_training

### Giving the name of the output log file
#$ -o /mnt/data/personal/simon/logs

#### Combining output/error messages into one file
#$ -j y

#### nodes:ppn - how many nodes & cores per node (ppn) that you require. Needs only one for head job
#$ -pe smp 1

#### mem: amount of memory that the job will need (per splot)
#$ -l h_vmem=128g,gpu=1

#$ -P GPU.p
#$ -cwd 
############ change ######################
source /mnt/data/personal/simon/miniconda3/etc/profile.d/conda.sh
#conda activate /mnt/bin/deepCRE/deepCRE-p3.7-conda

conda activate /mnt/data/personal/simon/miniconda3/envs/deepCRE-p3.7

python /mnt/data/personal/simon/projects/DeepCRE-git/DeepCRE/model/train_ssr_ssc_models_leaf.py
######
