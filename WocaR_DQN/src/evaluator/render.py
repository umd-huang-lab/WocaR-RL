import gym
from VaR_DQN.utils.dqn_core import *
from VaR_DQN.utils.atari_utils import *
from VaR_DQN.utils.monitor import Monitor
from VaR_DQN.attacker.attacker import *
from VaR_DQN.attacker.pa_obs_attacker import *
from dqn_test import *
import os
import time

def main(args):
    env = gym.make(args.env)
    env = Monitor(env)
    env = make_env(env, frame_stack=True, scale=False)
    env.seed(1000)

    Param(torch.FloatTensor, torch.device("cpu"))
    Q_Atari = model_get('Atari', num_actions = env.action_space.n)
    agent = DQN_Agent(Q_Atari)
    agent_dir = "./learned_models/dqn/{}".format(args.env)
    agent.load_state_dict(torch.load(agent_dir, map_location=Param.device))
    action_meaning = env.unwrapped.get_action_meanings()
    
    if args.attacker == 'Huang':
        attacker = Huang_Attack()
    elif args.attacker == 'Pattanaik':
        attacker = Pattanaik_Attack()
    elif args.attacker == 'Obspol':
        exp_name = "dqn_obspol_attacker_{}_e{}_fgsm".format(args.env, args.epsilon)
        old_steps, pa_attacker, _ = \
                            torch.load(os.path.join("./learned_adv/{}/".format('acktr'),
                            exp_name + ".pt"), map_location=Param.device)
        pa_attacker.to(Param.device)
        attacker = ObsPol_Attack(pa_attacker, det=True, cont=False)
        masks = torch.ones(1,1)
        recurrent = torch.zeros(1, pa_attacker.recurrent_hidden_state_size, device=Param.device)
    else:
        attacker = None
        
    obs = env.reset()
    for i in range(args.num_episodes):
        while True:
            time.sleep(args.sleep)
            if args.attacker == 'Huang':
                obs = attacker.attack(agent.Q, obs/255., epsilon=args.epsilon, fgsm=True).squeeze(0)
            elif args.attacker == 'Pattanaik':
                obs = attacker.attack(agent.Q, obs/255., epsilon=args.epsilon, fgsm=True)
            elif args.attacker == 'Obspol':
                obs = torch.from_numpy(obs).unsqueeze(0)/255.
                obs, recurrent = attacker.attack_torch(agent, obs, recurrent, masks, epsilon=args.epsilon, fgsm=True)
            else:
                obs = obs/255.
            if args.attacker != 'Pattanaik' and args.attacker != 'Obspol':
                action = int(agent.select_epilson_greedy_action(obs, 0.01))
            else:
                action = int(agent.step_torch_epsilon_greedy(obs, 0.01))
            if args.print_actions:
                print(action_meaning[action])
            obs, r, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()
                break                
                
                
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--attacker', type=str, default='None')
    parser.add_argument('--epsilon', type=float, default=0.0001)
    parser.add_argument('--num_episodes', type=int, default=29)
    parser.add_argument('--sleep', type=float, default=0.)
    parser.add_argument('--print_actions',type=bool, default=False)
    
    args = parser.parse_args()
    main(args)