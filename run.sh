#!/bin/bash

date=`date +%Y%m%d`
time=`date +%H%M%S`
log_dir=run_logs/${date}
if [ ! -d ${log_dir} ]; then
mkdir -p ${log_dir}
fi
log_file_name=${log_dir}/train_${date}_${time}.log

# 生成有毒数据
# nohup python -u gen_targetdata.py -data cifar10 -poi True > gen_data.out 2>&1 &


# cifar10
# 后门
# nohup python -u main.py -mode 0 -gr 100 -data cifar10 -did 1 >> ${log_dir}/cifar101_0_0.out 2>&1 &
# nohup python -u main.py -mode 1 -gr 100 -data cifar10 -did 2 >> ${log_dir}/cifar101_1_0.out 2>&1 &
# nohup python -u main.py -mode 2 -gr 100 -data cifar10 -did 4 >> ${log_dir}/cifar101_2_0.out 2>&1 &
# nohup python -u main.py -mode 3 -gr 100 -data cifar10 -did 1 >> ${log_dir}/cifar101_3_0.out 2>&1 &
# nohup python -u main.py -mode 4 -gr 100 -data cifar10 -did 2 -lr 0.0001 >> ${log_dir}/cifar101_4_0.out 2>&1 &
# nohup python -u main.py -mode 5 -gr 100 -data cifar10 -did 2 -lr 0.0001 >> ${log_dir}/cifar101_5_0.out 2>&1 &
# nohup python -u main.py -mode 6 -gr 100 -data cifar10 -did 2 -lr 0.0005 >> ${log_dir}/cifar101_6_0.out 2>&1 &
# nohup python -u main.py -mode 7 -gr 1 -data cifar10 -did 1 -uls 1 -tau 0.5 -mu 1.0 -lr 0.01 >> ${log_dir}/cifar101_7_0.out 2>&1 &
nohup python -u main.py -mode 7 -data cifar10 -did 0 >> ${log_dir}/hyper.out 2>&1 &

# nohup python -u mia.py -data cifar10 -did 2 >> ${log_dir}/cifar101_mia.out 2>&1 &
# nohup python -u mia1.py -data cifar10 -did 1 >> ${log_dir}/cifar101_mia1.out 2>&1 &
# nohup python -u mia2.py -data cifar10 -did 2 >> ${log_dir}/cifar101_mia2.out 2>&1 &

# fmnist
# 后门
# nohup python -u main.py -mode 0 -gr 100 -data fmnist -did 3 >> ${log_dir}/fmnist1_0_0.out 2>&1 &
# nohup python -u main.py -mode 1 -gr 100 -data fmnist -did 4 >> ${log_dir}/fmnist1_1_0.out 2>&1 &
# nohup python -u main.py -mode 2 -gr 100 -data fmnist -did 4 >> ${log_dir}/fmnist1_2_0.out 2>&1 &
# nohup python -u main.py -mode 3 -gr 100 -data fmnist -did 1 >> ${log_dir}/fmnist1_3_0.out 2>&1 &
# nohup python -u main.py -mode 4 -gr 100 -data fmnist -did 1 >> ${log_dir}/fmnist1_4_0.out 2>&1 &
# nohup python -u main.py -mode 5 -gr 100 -data fmnist -did 1 >> ${log_dir}/fmnist1_5_0.out 2>&1 &
# nohup python -u main.py -mode 6 -gr 100 -data fmnist -did 1 >> ${log_dir}/fmnist1_6_0.out 2>&1 &
# nohup python -u main.py -mode 7 -gr 100 -data fmnist -did 1 -tau 0.5 -uls 1 -mu 1.0 -lbs 128 -lr 0.001 >> ${log_dir}/fmnist1_7_0.out 2>&1 &

# nohup python -u mia.py -data fmnist -did 0 >> ${log_dir}/fmnist1_mia.out 2>&1 &
# nohup python -u mia1.py -data fmnist -did 0 >> ${log_dir}/fmnist1_mia1.out 2>&1 &
# nohup python -u mia2.py -data fmnist -did 0 >> ${log_dir}/fmnist1_mia2.out 2>&1 &

# svhn
# nohup python -u main.py -mode 0 -gr 100 -data svhn -did 5 >> ${log_dir}/svhn1_0_0.out 2>&1 &
# nohup python -u main.py -mode 1 -gr 100 -data svhn -did 6 >> ${log_dir}/svhn1_1_0.out 2>&1 &
# nohup python -u main.py -mode 7 -gr 100 -data svhn -did 1 -tau 0.5 -uls 1 -mu 1.0 -lr 0.001 >> ${log_dir}/svhn1_7_0.out 2>&1 &

# cifar100
# nohup python -u main.py -mode 0 -gr 300 -data cifar100 -did 7 -lr 0.001 -ls 2 >> ${log_dir}/ciafr1001_0_0.out 2>&1 &
# nohup python -u main.py -mode 1 -gr 150 -data cifar100 -did 1 -lr 0.005 >> ${log_dir}/ciafr1001_1_0.out 2>&1 &
# nohup python -u main.py -mode 7 -gr 150 -data cifar100 -did 1 -tau 0.5 -uls 1 -mu 1.0 -lbs 50 -lr 0.001 >> ${log_dir}/cifar1001_7_0.out 2>&1 &