#!/bin/bash
#PBS -q gpupascal
#PBS -l ngpus=1
#PBS -l ncpus=6
#PBS -l walltime=12:00:00
#PBS -l mem=128GB

module load tensorflow/1.2.1-python2.7
export PYTHONPATH="$PYTHONPATH:/home/900/jxg900/gdata_home/keras:/home/900/jxg900/gdata_home/python_libs/h5py/lib/python2.7/site-packages"

cd /g/data3/fr5/jxg900/deep_weather/weather_cnn_lstm
python weather_cnn_lstm..global_avg_pooling.py EIDW



