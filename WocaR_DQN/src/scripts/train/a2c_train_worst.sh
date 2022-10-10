#! /bin/bash  

# Environment Type. e.g. AlienNoFrameskip-v4
ENV=$1
# CUDA ID
CUDA=$2

STEPS=1000000
DIR='./data/a2c_results/train_worst'
LOG='./data/log/log_train_worst'
HORIZON=64
NPROC=4

echo "run on cuda ${CUDA}"

if [ -d './data/a2c_results' ]; then
    echo "dir exists"
else
    echo "create a new dir"
    mkdir './data/a2c_results'
fi

if [ -d ${DIR} ]; then
    echo "dir exists"
else
    echo "create a new dir"
    mkdir ${DIR}
fi

python trainer_victim/a2c_train_worst.py --algo a2c --env-name ${ENV} --cuda-id ${CUDA} --num-env-steps ${STEPS} --res-dir ${DIR} --log-dir ${LOG} --use-linear-lr-decay --norm-env --log-interval 100 --save-dir learned_worst
python evaluator/a2c_test.py --algo a2c --env-name ${ENV} --cuda-id ${CUDA} --res-dir ${DIR} --log-dir ${LOG} --test-episodes 100 --victim-dir learned_worst/a2c