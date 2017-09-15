#!/bin/bash
#PBS -q gpupascal
#PBS -l ngpus=1
#PBS -l ncpus=6
#PBS -l walltime=12:00:00
#PBS -l mem=32GB

module load tensorflow/1.2.1-python2.7
export PYTHONPATH="$PYTHONPATH:/home/900/jxg900/gdata_home/keras:/home/900/jxg900/gdata_home/python_libs/h5py/lib/python2.7/site-packages"

cd /g/data3/fr5/jxg900/deep_weather/weather_cnn2d_resnet
python weather_cnn2d_resnet.py EIDW 2,3,22,2



