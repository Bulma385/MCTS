### Overview

Poker:

Angenommen jeder kann nur ein mal betten.
Rollout nach Flop pro gesampleter Gegner Händen:
- 50 Turn * 10 opponent * 10 own * 50 river * 10 op * 10 own = 25 mio
Rollout im turn:
- 10 opponent * 10 own * 50 river * 10 op * 10 own = 500k
Rollout im River:
- 10 op * 10 own = 100                                   ( hier mehrfach betten ermöglichen)


Grundidee:::
- 1 Tranformer call pro situation  -> Range of cards (50x50/2)              (oder clustern? selber clustern oder variable?)
- für die top k hands in range -> mcts rollouts

Zweite Idee::
- abstrahierte States
- und range estimation

Dritte Idee:
- eigener Baum mit nur action sequences without actual states and hands

Vierte Idee:
- aus representationen + opponent policy head opponent reaktions samplen

Problem::
- damit der search tree wächst muss geclustert werden weil gesamplete start werte zu viele 1k * 1k * (50 über 3 = 20k) = 20 milliarden
- das problem hat man nicht bei schach etc weil gleicher start punkt und optimale trajektorien verlaufen ähnlich

- 


### On Transforming Reinforcement Learning with
Transformers: The Development Trajectory (https://arxiv.org/pdf/2212.14164.pdf)

### Decision Transformer: Reinforcement
Learning via Sequence Modeling (https://proceedings.neurips.cc/paper/2021/file/7f489f642a0ddb10272b5c31057f0663-Paper.pdf)

### A Survey on Transformers in Reinforcement Learning (https://arxiv.org/pdf/2301.03044.pdf)



### Representation Matters: The Game of Chess Poses a Challenge to Vision Transformers (https://arxiv.org/pdf/2304.14918.pdf)
- net AlphaVile (Comb of AlphaZero, MobileNet, NextVit) convolutional transformer hybrid (https://github.com/QueensGambit/CrazyAra/pull/196)
- extend representations
- new vlaue loss representations, ie. add new objectives with loss to train

##### Networks
New more efficient Transformers:
Next-vit: Next
generation vision transformer for efficient deployment in realistic industrial scenarios. [16]
and
Trt-vit: Tensorrt-oriented
vision transformer [30]

CNN: Mobile Net [10]

Sidenotes:
. For training, we also make use of stochastic depth [12] in order to both speed up training and improve convergence. The scaling technique that allows to
generate networks in different sizes is adapted from EfficientNet [26].


