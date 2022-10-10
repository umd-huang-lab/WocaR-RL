import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--adv-lr', type=float, default=7e-4, help='adversary learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--adv-entropy-coef',
        type=float,
        default=0.01,
        help='adversary entropy term coefficient (default: 0.01)')   
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=64,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--policy-update-steps',
        type=int,
        default=10,
        help='number of policy update in an alternative training iteration')
    parser.add_argument(
        '--adv-update-steps',
        type=int,
        default=10,
        help='number of adversary update in an alternative training iteration')   
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=2,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='LunarLander-v2',
        help='environment to train on (default: LunarLander-v2)')
    parser.add_argument(
        '--log-dir',
        default='./data/log/',
        help='directory to save agent logs (default: ./data/log/)')
    parser.add_argument(
        '--save-dir',
        default='./learned_models/',
        help='directory to save agent logs (default: ./learned_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='stop printing information')

    parser.add_argument(
        '--cuda-id',
        type=int,
        default=0)
    parser.add_argument(
        '--test-episodes',
        type=int,
        default=1000,
        help='number of episodes to test return (default: 1000)')
    parser.add_argument(
        '--attacker',
        type=str,
        default='obspol',
        help='the attacker algorithm (default: None)')
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.01,
        help='the attack budget')
    parser.add_argument(
        '--eps-scheduler-opts',
        type=str,
        default='start=1,length=1000',
        help='alternating training eps scheduler')
    parser.add_argument(
        '--attack-lr',
        type=float,
        default=0.01,
        help='PGD attack learning rate')
    parser.add_argument(
        '--attack-steps',
        type=int,
        default=100,
        help='PGD attack learning steps')
    parser.add_argument(
        '--res-dir',
        default='./data/a2c_results/',
        help='directory to save agent rewards (default: ./data/a2c_results/)')
    
    parser.add_argument(
        '--load-adv',
        action='store_true',
        default=False,
        help='load pretrained attacker model')
    parser.add_argument(
        '--load',
        action='store_true',
        default=False,
        help='load pretrained victim models')
    parser.add_argument(
        '--train-nn',
        action='store_true',
        default=False,
        help='train obs attack nn')
    parser.add_argument(
        '--fgsm',
        action='store_true',
        default=True,
        help='whether to use fgsm')
    parser.add_argument(
        '--no-attack',
        action='store_true',
        default=False)
    parser.add_argument(
        '--beta',
        action='store_true',
        default=False,
        help='whether to use beta policy')
    parser.add_argument(
        '--det',
        action='store_true',
        default=False,
        help='whether to use deterministic policy')
    parser.add_argument(
        '--v-det',
        action='store_true',
        default=False,
        help='whether victim uses deterministic policy')
    parser.add_argument(
        '--use-nn',
        action='store_true',
        default=False,
        help='whether to use neural network observation attacker for pa attacks')
    parser.add_argument('--nn-hiddens', nargs='+', type=int)
    parser.add_argument(
        '--v-algo', 
        default='a2c', 
        help='algorithm to attack: a2c | ppo | acktr')
    parser.add_argument(
        '--plot',
        action='store_true',
        default=False)
    parser.add_argument(
        '--eppoch_smoothed',
        type=int,
        default=10,
        help='rewards smoothed over epochs for plotting')
    parser.add_argument(
        '--adv-ratio',
        type=float,
        default=0.5,
        help='attack ratio in collecting victim trajectary')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
