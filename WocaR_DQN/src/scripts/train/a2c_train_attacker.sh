#! /bin/bash  

# Environment Type. e.g. AlienNoFrameskip-v4
ENV=$1
# CUDA ID
CUDA=$2

# path to the trained victim model. 
VICTIM_PATH="./released_models/a2c_victim/" ## use this line if attack our pre-trained victim 
# VICTIM_PATH="./learned_models/a2c/"   ## use this line if attack a user-trained victim


STEPS=10000000 # total number of training steps
TEST_NUM=100 # total number of testing episodes
ROOT="./data/a2c_results/" # path to save results

if [ "${ENV}" = "AlienNoFrameskip-v4" ]; then
    # For Alien 
    EPS=0.001
    ALR=0.0001
    DIR='./data/a2c_results/alien/'
    LOG='./data/log/log_alien/'
    HORIZON=64
    NPROC=4
elif [ "${ENV}" = "BreakoutNoFrameskip-v4" ]; then
    # For Breakout 
    EPS=0.0005
    ALR=0.00005
    DIR='./data/a2c_results/breakout'
    LOG='./data/log/log_breakout'
    HORIZON=64
    NPROC=4
elif [ "${ENV}" = "PongNoFrameskip-v4" ]; then
    # For Pong
    EPS=0.0005
    ALR=0.00005
    DIR='./data/a2c_results/pong'
    LOG='./data/log/log_pong'
    HORIZON=32
    NPROC=32
elif [ "${ENV}" = "RoadRunnerNoFrameskip-v4" ]; then
    # For RoadRunner
    EPS=0.002
    ALR=0.0002
    DIR='./data/a2c_results/roadrunner'
    LOG='./data/log/log_roadrunner'
    HORIZON=64
    NPROC=4
elif [ "${ENV}" = "SeaquestNoFrameskip-v4" ]; then
    # For Seaquest
    EPS=0.005
    ALR=0.0005 
    DIR='./data/a2c_results/seaquest'
    LOG='./data/log/log_seaquest'
    HORIZON=64
    NPROC=4
elif [ "${ENV}" = "TutankhamNoFrameskip-v4" ]; then
    # For Tutankham
    EPS=0.001
    ALR=0.0001
    DIR='./data/a2c_results/tutankham'
    LOG='./data/log/log_tutankham'
    HORIZON=128
    NPROC=4
else
    # For other environments
    EPS=0.001  # epsilon: attack budget 
    ALR=0.0001   # attacker's learning rate (if use PGD)
    DIR='./data/a2c_results/'${ENV}
    LOG='./data/log/log_'${ENV}
    HORIZON=64
    NPROC=4
fi

echo "run on cuda ${CUDA}"

if [ -d ${DIR} ]; then
    echo "dir exists"
else
    if [ -d ${ROOT} ]; then
        echo "create a new dir"
        mkdir ${DIR}
    else
        echo "recursively create a new dir"
        mkdir ${ROOT}
        mkdir ${DIR}
    fi
fi

### train and test pa-ad attack ########
python trainer_adv/a2c_pa_attacker.py --env-name ${ENV} --algo acktr --epsilon ${EPS} --cuda-id ${CUDA} --num-env-steps ${STEPS} --num-steps ${HORIZON} --num-processes ${NPROC} --use-linear-lr-decay --fgsm --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} 
python evaluator/a2c_test.py --env-name ${ENV} --algo acktr --epsilon ${EPS} --cuda-id ${CUDA} --attacker paad --fgsm --res-dir ${DIR} --log-dir ${LOG} --det --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
########################################


### test natural reward ################
python evaluator/a2c_test.py --env-name ${ENV} --cuda-id ${CUDA} --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
########################################


### test heuristic attacks #############
# random attack
python evaluator/a2c_test.py --env-name ${ENV} --cuda-id ${CUDA} --attacker random --epsilon ${EPS} --fgsm --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
# minbest attack
python evaluator/a2c_test.py --env-name ${ENV} --cuda-id ${CUDA} --attacker minbest --epsilon ${EPS} --fgsm  --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
# minbest+momentum attack
python evaluator/a2c_test.py --env-name ${ENV} --cuda-id ${CUDA} --attacker minbest --epsilon ${EPS} --momentum --attack-steps 10 --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
# maxdiff attack
python evaluator/a2c_test.py --env-name ${ENV} --cuda-id ${CUDA} --attacker maxdiff --epsilon ${EPS} --attack-steps 10 --attack-lr ${ALR}  --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
########################################


### train and test sa-rl attack ########
python trainer_adv/a2c_sa_attacker.py --env-name ${ENV} --epsilon ${EPS} --cuda-id ${CUDA} --num-env-steps ${STEPS} --num-steps ${HORIZON} --num-processes ${NPROC} --use-linear-lr-decay --res-dir ${DIR} --log-dir ${LOG} --max-grad-norm 0.1 --lr 1e-6 --victim-dir ${VICTIM_PATH} 
python evaluator/a2c_test.py --env-name ${ENV} --epsilon ${EPS} --cuda-id ${CUDA} --attacker sarl --res-dir ${DIR} --log-dir ${LOG} --det --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
########################################

