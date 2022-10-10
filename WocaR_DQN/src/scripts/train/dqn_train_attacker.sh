#! /bin/bash  

# Environment Type. e.g. AlienNoFrameskip-v4
ENV=$1
# CUDA ID
CUDA=$2

# path to the trained victim model. 
VICTIM_PATH="./learned_models/" ## use this line if attack our pre-trained victim 
# VICTIM_PATH="./learned_models/dqn/"   ## use this line if attack a user-trained victim

STEPS=6000000 # total number of training steps
TEST_NUM=100 # total number of testing episodes
ROOT="./data/dqn_results/" # path to save results

if [ "${ENV}" = "AlienNoFrameskip-v4" ]; then
    # For Alien 
    EPS=0.00075
    ALR=0.000075
    DIR='./data/dqn_results/alien/'
    LOG='./data/log/log_alien/'
    HORIZON=128
    NPROC=4
elif [ "${ENV}" = "BoxingNoFrameskip-v4" ]; then
    # For Boxing
    EPS=0.001
    ALR=0.00001
    DIR='./data/dqn_results/boxing'
    LOG='./data/log/log_boxing'
    HORIZON=128
    NPROC=4
elif [ "${ENV}" = "FreewayNoFrameskip-v4" ]; then
    # For Freeway
    EPS=0.0003
    ALR=0.00003
    DIR='./data/dqn_results/freeway'
    LOG='./data/log/log_freeway'
    HORIZON=64
    NPROC=4
elif [ "${ENV}" = "PongNoFrameskip-v4" ]; then
    # For Pong
    EPS=0.0002
    ALR=0.00002
    DIR='./data/dqn_results/pong'
    LOG='./data/log/log_pong'
    HORIZON=64
    NPROC=4
elif [ "${ENV}" = "RoadRunnerNoFrameskip-v4" ]; then
    # For RoadRunner
    EPS=0.0005
    ALR=0.00005
    DIR='./data/dqn_results/roadrunner'
    LOG='./data/log/log_roadrunner'
    HORIZON=64
    NPROC=4
elif [ "${ENV}" = "SeaquestNoFrameskip-v4" ]; then
    # For Seaquest
    EPS=0.0005 
    ALR=0.00005
    DIR='./data/dqn_results/seaquest'
    LOG='./data/log/log_seaquest'
    HORIZON=64
    NPROC=4
elif [ "${ENV}" = "TutankhamNoFrameskip-v4" ]; then
    # For Tutankham
    EPS=0.00075
    ALR=0.000075
    DIR='./data/dqn_results/tutankham'
    LOG='./data/log/log_tutankham'
    HORIZON=128
    NPROC=4
else
    # For other environments
    EPS=0.0005  # epsilon: attack budget 
    DIR='./data/dqn_results/${ENV}'
    LOG='./data/log/log_${ENV}'
    HORIZON=128
    NPROC=4
fi


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

### test natural reward ################
python evaluator/dqn_test.py --env-name ${ENV} --cuda-id ${CUDA} --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
########################################

### test heuristic attacks #############
# random attack
python evaluator/dqn_test.py --env-name ${ENV} --cuda-id ${CUDA} --attacker random --epsilon ${EPS} --fgsm --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
# minbest attack
python evaluator/dqn_test.py --env-name ${ENV} --cuda-id ${CUDA} --attacker minbest --epsilon ${EPS} --fgsm  --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
# minbest+momentum attack
python evaluator/dqn_test.py --env-name ${ENV} --cuda-id ${CUDA} --attacker minbest --epsilon ${EPS} --momentum --attack-steps 10 --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
# minq attack
python evaluator/dqn_test.py --env-name ${ENV} --cuda-id ${CUDA} --attacker minq --epsilon ${EPS} --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
# maxdiff attack
python evaluator/dqn_test.py --env-name ${ENV} --cuda-id ${CUDA} --attacker maxdiff --epsilon ${EPS} --attack-steps 10 --attack-lr ${ALR}  --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
########################################

### train and test pa-ad attack ########
python trainer_adv/dqn_pa_attacker.py --env-name ${ENV} --algo acktr --epsilon ${EPS} --cuda-id ${CUDA} --num-env-steps ${STEPS} --num-steps ${HORIZON} --num-processes ${NPROC} --use-linear-lr-decay --fgsm --res-dir ${DIR} --log-dir ${LOG} --victim-dir ${VICTIM_PATH}
python evaluator/dqn_test.py --env-name ${ENV} --algo acktr --cuda-id ${CUDA} --attacker paad --epsilon ${EPS} --fgsm --res-dir ${DIR} --log-dir ${LOG} --det --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
########################################

### train and test sa-rl attack ########
python trainer_adv/dqn_sa_attacker.py --env-name ${ENV} --epsilon ${EPS} --cuda-id ${CUDA} --num-env-steps ${STEPS} --num-steps ${HORIZON} --num-processes ${NPROC} --use-linear-lr-decay --res-dir ${DIR} --log-dir ${LOG} --max-grad-norm 0.1 --lr 1e-6 --victim-dir ${VICTIM_PATH} 
python evaluator/dqn_test.py --env-name ${ENV} --cuda-id ${CUDA} --attacker sarl --epsilon ${EPS} --res-dir ${DIR} --log-dir ${LOG} --det --victim-dir ${VICTIM_PATH} --test-episodes ${TEST_NUM}
########################################

