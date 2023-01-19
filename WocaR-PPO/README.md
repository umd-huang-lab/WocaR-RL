# WocaR-PPO: Efficiently Improving the Robustness of Reinforcement Learning Agents
This repo contains a reference implementation for our WocaR-PPO algorithm for robust RL.
The code is based on [SA-PPO](https://github.com/huanzhang12/SA_PPO).

## setup environment
The required packages are available in requirements.

```bash
git submodule update --init
pip install -r requirements.txt
```

Python 3.7+ is required. Note that you need to install MuJoCo 1.5 first to use the OpenAI Gym environments. 
auto_LiRPA is an external package. Please see [auto_LiRPA](https://github.com/KaidiXu/auto_LiRPA) for detailed instructions.

## train WocaR-PPO agents
```
python run.py --config-path config_hopper_robust_q_ppo_sgld.json 
python run.py --config-path config_walker_robust_q_ppo_sgld.json 
python run.py --config-path config_halfcheetah_robust_q_ppo_sgld.json 
python run.py --config-path config_ant_robust_q_ppo_sgld.json 
```
## test the trained WocaR-PPO agents
```
# Change the --exp-id to match your experiments
python test.py --config-path config_hopper_robust_q_ppo_sgld.json --exp-id YOUR_EXP_ID --deterministic
```

## test released WocaR-PPO models

To evaluate natural rewards without attack:
```
python test.py --config-path config_hopper_robust_q_ppo_sgld.json --load-model release_models/WocaR-PPO/hopper.model --deterministic 
```

## test under attacks

Directly use the random and MaxDiff attack to evaluate:
```
# Random attack
python test.py --config-path config_hopper_robust_q_ppo_sgld.json --load-model release_models/WocaR-PPO/hopper.model --deterministic --attack-eps=0.075 --attack-method random --deterministic
# MaxDiff attack
python test.py --config-path config_hopper_robust_q_ppo_sgld.json --load-model release_models/WocaR-PPO/hopper.model --deterministic --attack-eps=0.075 --attack-method action --deterministic
```

To train SA-RL and PA-AD attack, see [ATLA](https://github.com/huanzhang12/ATLA_robust_RL) and [PA-AD](https://arxiv.org/abs/2106.05087) for instructions.