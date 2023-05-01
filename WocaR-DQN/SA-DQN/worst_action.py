import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
sys.path.append(__file__)

from ibp import *

import torch
import numpy as np
import random


def worst_action_select(worst_q, upper_q, lower_q):
    mask = torch.zeros(upper_q.size())
    for i in range(upper_q.size()[1]):
        upper = upper_q[:, i].view(upper_q.size()[0], 1)
        if_perturb = (upper > lower_q).all(1)
        mask[:, i] = if_perturb.byte()

    worst_q = worst_q.masked_fill_(mask==0, 1e9)
    worst_actions = worst_q.min(1)[-1].unsqueeze(1)
    worst_q = worst_q.gather(1, worst_actions)

    return worst_actions, worst_q

def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    random.seed(1234)
    np.random.seed(123)
    num_actions = 3

    worst_q = torch.FloatTensor([[0.058, 0.063, 0.024], [3.0, 4.0, 1.0]])
    upper_q = torch.FloatTensor([[0.027, 0.039, 0.042], [4.3, 4.5, 2.7]])
    lower_q = torch.FloatTensor([[0.028, 0.015, 0.021], [2.8, 3.6, 2.6]])

    actions, worst_q = worst_action_select(worst_q, upper_q, lower_q)
    print(actions)
    print(worst_q)
    worst_q_value = worst_q.min(1)[0]
    print(worst_q_value)

if __name__ == "__main__":
    main()
    

    