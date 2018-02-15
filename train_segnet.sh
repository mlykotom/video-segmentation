#!/bin/bash
#PBS -m abe
#PBS -q gpu
#PBS -l walltime=9:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=8gb:gpu_cap=cuda35
#xxPBS -l scratch_ssd=1gb
# chybovy vystup pripoji ke standarnimu vystupu a posle mail pri skonceni ulohy
# direktivy si upravte/umazte dle potreb sveho vypoctu

# nastaveni uklidu SCRATCHE pri chybe nebo ukonceni
# (pokud nerekneme jinak, uklidime po sobe)
trap 'clean_scratch' TERM EXIT

source /storage/ostrava1/home/mlyko/.profile
cd /storage/ostrava1/home/mlyko/gta-segmentation
python train.py -r -m segnet