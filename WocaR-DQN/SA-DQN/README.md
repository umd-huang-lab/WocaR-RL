# Improving DQN Training Speed with SA-DQN-based WocaR-DQN Implementation

## Introduction

We have implemented a new version of WocaR-DQN based on the SA-DQN implementation. The new implementation provides faster training speeds by using parallel sampling and efficient data transfer methods. In this document, we outline the differences between our implementation and the version described in our research paper, and provide some initial results for the new implementation.

## Highlight:

1. Faster Training Speeds: The new implementation of WocaR-DQN uses parallel sampling and efficient data transfer methods, resulting in faster training speeds. Users can adjust the training process by modifying configs.

2. Differences from Research Paper: Our new implementation of WocaR-DQN differs from the one described in our research paper. One reason is that the implementation of SA-DQN uses multiple different Atari wrappers from our original codebase. The wrapper differences in the Atari environment can cause variations in the output state and restricted action number, leading to differences in the results compared to the pretrained model. We recommend emphasizing the impact of wrapper differences when using WocaR-DQN as a baseline in your research papers.

3. Preliminary Results: We have provided some pretrained models for the new implementation of WocaR-DQN (using PGD but not convex relaxation). However, these results do not represent the best possible model with parameter search, and we plan to update the model with additional improvements in the future. Feel free to use these pretrained models and please highlight the difference mentioned above in your paper.

4. Please note that we have added worst-case-loss to the new implementation to perform backward, which we have observed can exacerbate training instability. Therefore, you may consider setting 'worst_kappa_end' to 1 to avoid using worst-case-loss. And in the early training, you can try to set a lower 'worst_ratio' to decrease the effect from the worst value.

5. We also provide another implementation of WocaR-DQN in 'RS-DQN' document, which, however, trains much slower than the SA-DQN implementation. Nevertheless, the performance of WocaR-DQN in this other implementation is consistent with the results reported in our paper. You can flexibly consider using these implementation based on your time and computing resources or directly use the models we provided.

