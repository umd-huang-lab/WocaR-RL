# Updated WocaR-DQN Implementation 

## Training

To train a WocaR-DQN model on Pong, run this command:

```
python main.py --env PongNoFrameskip-v4 --robust --wocar --load-path "trained_models/PongNoFrameskip-v4_wocar.pt" --total-frames 2000000 --exp-epsilon-decay 1
```
To speed up training by using gpu x (in a system with one gpu x=0) add the following argument `--gpu-id x`.
To schedule the worst-case value learning by fixing configs 'worst_rate' and 'worst_ratio'


## Evaluation

To evaluate our pretrained WocaR model, use the following command:

```
python evaluate.py --env PongNoFrameskip-v4 --load-path "trained_models/PongNoFrameskip-v4_wocar.pt" --pgd
```

If you want to test under the SA-RL or PA-AD attacks, you should attack the model on the attacking code. 

## Tips

We set a lot of configs for the environment wrapper, when you use pretrained models, please take care of the shape of states and actions. And you should mention in your work about these information.