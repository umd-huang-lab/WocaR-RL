#! /bin/bash  

# Environment Type. e.g. PongNoFrameskip-v4
ENV=$1
# CUDA ID
CUDA=$2

STEPS=6000000
DIR='./data/dqn_results/train'
LOG='./data/log/log_train_dqn'
VICTIM_PATH="./learned_models/dqn/"  
HORIZON=64
NPROC=4

echo "run on cuda ${CUDA}"

if [ -d './data/dqn_results' ]; then
    echo "dir exists"
else
    echo "create a new dir"
    mkdir './data/dqn_results'
fi

if [ -d ${DIR} ]; then
    echo "dir exists"
else
    echo "create a new dir"
    mkdir ${DIR}
fi

if [ -d ${VICTIM_PATH} ]; then
    echo "dir exists"
else
    echo "create a new dir"
    mkdir ${VICTIM_PATH}
fi

python trainer_victim/dqn.py --env ${ENV} --frame_total ${STEPS} --lr 6.25e-5 -doubleQ -prioritized_replay -cuda ${CUDA} --learning_starts 50000 --exp_name ${ENV} --log_steps 100
python -W ignore evaluator/dqn_test.py --env-name ${ENV} --cuda-id ${CUDA} --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} --test-episodes 100
