# WocaR_DQN: Efficiently Improving the Robustness of Reinforcement Learning Agents
This repo contains a reference implementation for our WocaR_DQN algorithm for robust RL.


## setup environment

Please run the following command to install required packages (suggested python version: 3.7.0)

```
# requirements
pip install -r requirements.txt

# our packages
pip install -e .
```

Python 3.7+ is required. Note that you need to install atari-py first to use the OpenAI Gym environments. 

## train WocaR_DQN agents
```
python dqn.py --config ./config/wrdqn.json  
# change the env-name to select environments
```
## test the trained WocaR_DQN agents
```
# the option --attacker can be one of "minbest", "momentum", "minq", "maxdiff", "random"
python evaluator/test_attack.py --env-name PongNoFrameskip-v4 --v-type dqn --v-path learned_models/YOUR-EXP --attacker minbest --epsilon 0.0002  --test-episodes 100
```

## train PA-AD attacker for robust models

```
# change the victim-dir to your exp dir
bash scripts/train/dqn_train_attacker.sh PongNoFrameskip-v4 0
```

## test robust models uding PA-AD attacks
```
python evaluator/test_attack.py --env-name PongNoFrameskip-v4 --v-type dqn --v-path learned_models/YOU_MODEL --attacker paad --det --attack-model learned_models/YOUR-ATTACKER --epsilon 0.0002 --test-episodes 100
```