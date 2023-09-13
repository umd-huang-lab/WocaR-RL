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
python test.py --config-path config_hopper_robust_q_ppo_sgld.json --load-model release_models/WocaR-PPO/wocar_hopper.model --deterministic 
```

## test under attacks

Directly use the random and MaxDiff attack to evaluate:
```
# Random attack
python test.py --config-path config_hopper_robust_q_ppo_sgld.json --load-model release_models/WocaR-PPO/wocar_hopper.model --deterministic --attack-eps=0.075 --attack-method random --deterministic
# MaxDiff attack
python test.py --config-path config_hopper_robust_q_ppo_sgld.json --load-model release_models/WocaR-PPO/wocar_hopper.model --deterministic --attack-eps=0.075 --attack-method action --deterministic
```

To train SA-RL and PA-AD attack, see [ATLA](https://github.com/huanzhang12/ATLA_robust_RL) and [PA-AD](https://arxiv.org/abs/2106.05087) for instructions.

**Pretrained Models**

We have updated several pretrained models for WocaR-PPO and present their performance below. It's important to note that these pretrained models were selected randomly from training runs using the best hyperparameters. In RL algorithms, variance across training runs can be substantial. Therefore, to provide a robust evaluation, we conducted 30 training runs for each agent configuration. The reported performance metrics in our paper represent the median performance under the strongest attacks, rather than the best or worst case scenarios. Consequently, there may be difference but in variance (+-200) between our reported results and the performance of the pretrained models.

| Environment        | No attack | Heuristic attack | Evasion attack |
| ------------------ | --------- | ---------------- | -------------- |
| Ant-v2 (pertained) | 5465      | 4624             | 3342           |
| Reported           | 5596      | 5831             | 3164           |
| HalfCheetah-v2     | 6336      | 5319             | 4432           |
| Reported           | 6032      | 5226             | 4269           |
| Hopper-v2          | 3624      | 3056             | 2642           |
| Reported           | 3616      | 3277             | 2579           |
| Walker2d           | 4132      | 4157             | 2932           |
| Reported           | 4156      | 4093             | 2722           |

