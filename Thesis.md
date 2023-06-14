### Thesis::

Introduction:
- MCTS guarenteed to conv to Nash Eq for 2p0s perf info games
- determinization techniques have problems and do not converge, e.g. strategy fusion (old sources Finding optimal strategies for imperfect information games + . Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search)
- ISMCTS addresses this problem (strategy fusion) (paper Information set monte carlo tree search)
- 


### Overview

Poker:

Angenommen jeder kann nur ein mal betten.
Rollout nach Flop pro gesampleter Gegner Händen:
- 50 Turn * 10 opponent * 10 own * 50 river * 10 op * 10 own = 25 mio
Rollout im turn:
- 10 opponent * 10 own * 50 river * 10 op * 10 own = 500k
Rollout im River:
- 10 op * 10 own = 100                                   ( hier mehrfach betten ermöglichen)


Representation Learning for Transformer Encoder:
- Equity Calculator  (supervised)
- Sequence reconstruction from its representation (self supervised)

IDEE::
- our turn -> net prior action probs -> sample action -> opponent turn -> trans sample opponent cards -> net sample opponent action -> ... own and opponent action until chance node -> sample chance node outcome -> ... repeat until terminal node -> backprop reward
- repeat from our turn, sample own action again using uct & prior probs, sample opponent cards etc.
- no strategy fusion problem due to never letting the net know what the sampled cards are???

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



### On Transforming Reinforcement Learning with
Transformers: The Development Trajectory (https://arxiv.org/pdf/2212.14164.pdf)

### Decision Transformer: Reinforcement
Learning via Sequence Modeling (https://proceedings.neurips.cc/paper/2021/file/7f489f642a0ddb10272b5c31057f0663-Paper.pdf)

### Multi-Game Decision Transformers (https://arxiv.org/abs/2205.15241)
- released models -> use this transformer?

### DeepStack


### Information Set Monte Carlo Tree Search (https://eprints.whiterose.ac.uk/75048/1/CowlingPowleyWhitehouse2012.pdf)
- not the same as determinization, combats problems as strat fusion (Russell and Norvig [24] “averaging over clairvoyance.”)
- nodes in the tree are information set -> more efficient comp budget since nodes share comput
- model state distributions inside information sets as uniform -> no belief distributions
- ISMCTS finds mixed policies for the small, solved game of Kuhn Poker [14].
- 


### Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search (https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf)
-  1998, Frank and Basin published an extensive critique of the PIMC
- Problem 1 : strategy fusion: incorrect assumption that one could use different strategies in sampled games 
- Problem 2: non-locality: in perf info games,  the value of a node only depends on the children
- 3 properties in focus:
1. leaf correlation
2. bias
3. disambiguation factor






### (Brown) Superhuman AI for heads-up no-limit poker: Libratus beats top professionals (https://www.science.org/doi/pdf/10.1126/science.aao1733?casa_token=ArKnIuhMTPgAAAAA:cs7WEuyGlX_DsiUrADYYxMJsyy72Vvcqldwwffs4HIY3Gw3OMmq_ng2F2vowVFTLbQVCqG8yxMaQ_3M; file:///C:/Users/ccoem/Downloads/aao1733_brown_sm.pdf)
- until flop use fine graned abstractoion; then more abstract, calc blueprint, after flop sub game solving  (14–16, 42) in real time
1. Abstraction solving (via MCCFR) yields blueprint strat
- action abstraction (source 29)
- algorithmic hand abstraction
- probabilistically skip over unpromising actions  (30, 39) (bucket strategies benefit from that)
2. Sub Game Solving
- unsafe and save sub game solving (14,15,42,48)
- unsafe in practice better
- Idea: bound by blueprint strat payoff
- force opponent to adapt to different bet sizes by changing them in subgames (49)
-  de-emphasizing hands the opponent would only be holding if she had made an earlier mistake
3. Self-Improvement
- add actions that opponents actually take to the blueprint and solve in the background (54)
- 



### Combining Prediction of Human Decisions with ISMCTS in Imperfect Information Games (https://arxiv.org/pdf/1709.09451.pdf)
- ISMCTS  [6, 18, 22]
- predict (determine) opponent actions, rest ismcts
- given information states -> 
1. estimate opponents (previous) actions' distribution
2. for each action estimate payoff for own action by performing semi-determinized rollout 
3. expected value for each action given information state is weighted sum of prob and estimated payoff
- information state abstraction 
- node representation decrease - byte to bit uint? hash mapping structure?


### A Survey on Transformers in Reinforcement Learning (https://arxiv.org/pdf/2301.03044.pdf)
Challenges:
1. training data depends on current policy
2. RL algos highly sensitive to design choices
3. transformers high comp cost

Offline RL::
- static offline dataset (e.g. replay buffer), "constrain the learned policy close to the data distribution, to avoid out-of-distribution actions"
- popular trend RL via SL (RvS) [Emmons et al., 2021],

Goal Conditioned RL::
- Goal-Conditioned Reinforcement Learning: Problems and Solutions (https://arxiv.org/pdf/2201.08299.pdf)

Model-based RL:: (learn model interesting)
- Mastering atari, go, chess and shogi by planning with a learned model
- generate imaginary trajectories: Dream to control: Learning behaviors by latent imagination
(stehen geblieben 2.2 Tranformers!!!!!!!)







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


