### Thesis::

general RL problems: bootstrapping, deadlz triad

https://pytorch.org/tutorials/beginner/transformer_tutorial.html

minGPT https://github.com/karpathy/minGPT

Pre-Trained Language Models for Interactive Decision Making https://arxiv.org/pdf/2202.01771.pdf

A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems

Long Range Arena: A Benchmark for Efficient Transformers

Introduction:
- MCTS guarenteed to conv to Nash Eq for 2p0s perf info games
- determinization techniques have problems and do not converge, e.g. strategy fusion (old sources Finding optimal strategies for imperfect information games + . Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search)
- ISMCTS addresses this problem (strategy fusion) (paper Information set monte carlo tree search)
- success in poker source N. Brown and T. Sandholm, “Superhuman AI for multiplayer poker,


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

Idee:
sequence o0,s0,a0,o1,s1,a1,... with masked sk when normal playing

Problem::
- damit der search tree wächst muss geclustert werden weil gesamplete start werte zu viele 1k * 1k * (50 über 3 = 20k) = 20 milliarden
- das problem hat man nicht bei schach etc weil gleicher start punkt und optimale trajektorien verlaufen ähnlich



### DeepStack


### Weighting Information Sets with Siamese Neural Networks in Reconnaissance Blind Chess (file:///D:/Dokumente/Downloads/IEEE_COG_2023.pdf)
- predicts only one perfect information state
- trained using triplets <Ot, pt, nti> ; observation, positive true state, negative states
- triplet loss l = max(dp - dn + m, 0) , m being a margin
- siamese networks usually expects same input data (here 128x8x8), hence
- observation and board encoding networks (5 layers, 64filters, elu act) to map into matching latent encodings 
- the distance of the outputs in the embedding space yields the probabilities
- 'contextual preference ranking'  (“Predicting human card ¨selection in Magic: The Gathering with contextual preference ranking)
- Siamese Network (10 conv layers, 128 filters, elu, skip, from 3x3 to 1x1 filters to combine into one oputput, fc neurons 512)
- lr 0.0001, AdamW, in each epoch: each anchor and positive example once, sample k  negatives and use the closest
- Problems:1) The agent ignores potentially dangerous boards, that
could lead to immediate defeat if they are not regarded
as the most likely one.
2) In contrast to most other agents, our agent has no
concept of “cautious” moves that perform adequately
on many boards. Instead, it takes gambles based on
guessing the correct board


### (Integrating Opponent Models with Monte-Carlo Tree Search in Poker 2010)


### Memory Bounded Monte Carlo Tree Search (Powley, Cowling 2017) file:///D:/Dokumente/Downloads/12932-Article%20Text-16449-1-2-20201228.pdf
- IS-MCTS with bounded memory
- tree stored as children singly linked list (leftmost child, and right sibling)
- recycle nodes via queue, recycling the lastly updated nodes (node that children are updated before their parents so lastly updated nodes always have leaf nodes as children)
- c++ implementation with 48 or 56 bytes per node repect without and with queue


### Information Set Monte Carlo Tree Search (https://eprints.whiterose.ac.uk/75048/1/CowlingPowleyWhitehouse2012.pdf)
- not the same as determinization, combats problems as strat fusion (Russell and Norvig [24] “averaging over clairvoyance.”)
- nodes in the tree are information set -> more efficient comp budget since nodes share comput
- model state distributions inside information sets as uniform -> no belief distributions
- ISMCTS finds mixed policies for the small, solved game of Kuhn Poker [14].
-  strategy fusion is detrimental, ISMCTS shows great promise. However, in domains such as Dou Di Zhu, where information sets have large numbers of legal moves and the effect of strategy fusion is not so clear, ISMCTS offers no immediate benefit over existing approaches


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



### STABILIZING TRANSFORMERS FOR REINFORCEMENT LEARNING



### Multi-Game Decision Transformers (https://arxiv.org/abs/2205.15241)
- released models -> use this transformer?

### Awesome Decision Transformer Collection (https://github.com/opendilab/awesome-decision-transformer)
Papers:
- LATTE: LAnguage Trajectory TransformEr
- Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL
- Multi-Game Decision Transformers
- Deep Transformer Q-Networks for Partially Observable Reinforcement Learning
- You Can't Count on Luck: Why Decision Transformers and RvS Fail in Stochastic Environments
- Online Decision Transformer
- (CQL Conservative Q-Learning for Offline Reinforcement Learning)
- (upside down RL ?)
- (Generalized Decision Transformer for
Offline Hindsight Information Matching. )

### When does return-conditioned supervised learning work for offline reinforcement learning?

### Vector Quantized Models for Planning
- mcts 

 Zheng
et al. [36] introduced the online DT (ODT), a learning framework for RL that incorporates online fine-tuning intothe DT pretraining algorithm. To balance the explorationexploitation tradeoff

### On Transforming Reinforcement Learning with Transformers: The Development Trajectory (https://arxiv.org/pdf/2212.14164.pdf)
- feature representations or environmental representations
- advantage function = difference between action value and state value
- dynamic programming, model-free, and model based method
- Dynamic programming is used to compute the optimal policy given a known MDP. In modelfree methods, since we do not know the MDP, we use
the sampling method instead. We can directly optimize thepolicy through a policy gradient method (e.g., REINFORCE[71], PG [72], or NPG [73]), use a value iteration methodto determine the policy from the value function (e.g., the
deep Q-network (DQN) [74], Double DQN [75], or DuelingDQN [70]), or use an actor-critic method that adopts thecritic’s value to improve the actor’s policy (e.g., DDPG[76], PPO [77], or SAC [78]). In model-based methods (e.g.,MOReL [79], MOPO [80], and COMBO [81]),
- positional encodings
- residual connections
- actor learner destillation “Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures
- stehen geblieben canonical RL



### A Transformer-based Framework for Multivariate Time Series Representation Learning
- "In particular, each training sample.."
- normalize features for each dimension
- then lin proj into model dim (emb)
- then add pos encodings
- An important consideration regarding time series data is that
individual samples may display considerable variation in length.
This issue is effectively dealt with in our framework: after setting a
maximum sequence length𝑤 for the entire dataset, shorter samples
are padded with arbitrary values, and we generate a padding mask
which adds a large negative value to the attention scores for the
padded positions, before computing the self-attention distribution
with the softmax function. This forces the model to completely
ignore padded positions, while allowing the parallel processing of
samples in large minibatches.
- here batch normalization, originally´layer norm
- final representations are mapped into desired regression/classification output by 1 fc layer
- mse or cross entropy loss
- skipped pretraining experiments, adding samples


### Deep Transformer Q-Networks for Partially Observable Reinforcement Learning
- transformer decoder architecture; learned positional encodings; learned obs emb
- input: previous k observations projected linearly into model dim; output: Q values for every step (use all Q values for training!!! (like in (Al-Rfou et al., 2019) in NLP), only last for inference)
- NLP pos enc (Wang & Chen, 2020) https://arxiv.org/abs/2010.04903
- 4.2 Transformer decoder structur:  GPT architecture (Radford et al., 2018),  masked multi-headed self-attention and a position-wise feedforward network, combine layers
- ? Note that the for loop depicted in Algorithm 1 can be done in one forward pass of the network because of the causally-masked self-attention mechanism.
- ! 5.3 options for combine and layer norm + A comparison of different transformer backbones, such as Big Bird (Zaheer et al., 2020), sparse transformers (Child et al., 2019), or the TransformerXL would be interesting future study
- double DQN ((Van Hasselt et al., 2016)) strategy
- similarity scores of positional encodings -> indicate usefullness
- github


### You Can't Count on Luck: Why Decision Transformers and RvS Fail in Stochastic Environments
- code: https://sites.google.com/view/esper-paper
- In environments where a decently-strong dynamics model cannot be learned, the adversarial clustering in ESPER may fail
- DT falsely assume in stoch environments that actions that end up in achieving a particular goal are optimal for achieving that goal
- if conditioned on positive reward, the model doesnt get to see traj with negative reward that result from the same actions
- bad performance independent of amout of data
- condition on outcome that is fully determinized by actions
- learn environment-storasticity-independent representations and achives stronger alignment between target return and actual expected return
- interested in statistics that are independent to uncontrollable radnomness
ESPER:
- train neural net to learn stoch ind statistics in 3 phases
1. discrete representation cluster by auto-encoder with adversarial loss??
2. learn to predict average trajectory return
3. RvS agent conditioned on state and avg return
- DT arch. A.5.1
- Problems:
1. in environments where adecently-strong dynamics model cannot be learned, the adversarial clustering in ESPER may fail


### Decision Transformer: Reinforcement Learning via Sequence Modeling (https://proceedings.neurips.cc/paper/2021/file/7f489f642a0ddb10272b5c31057f0663-Paper.pdf)
- autoregressive modelling
- causal masking ?
- can utitlize language and vision tranformers
- no need for specializd RL frameworks
- work effectively with sparse and distracting rewards
- offline RL: only access to limited trainingdata
- GPT arch., attention calc with only j in 1 to i (only consider previous elements)
- return to go = sum of future rewards
- for each modality learn linear embedding that projects into embedding dim followed bz normalization (Ba et al. Layer normalization)
- (for visual input -> conv layers)
- additionallz embedding for each time step (note 3 tokens, i.e. reward, state, action have same pos emb)
- ? prediction head corresponding to input st trained to predict at (with cross-entropy loss or MSE if cont.) and losses for each time step are averaged
- init generation: set enviroment starting state and set e.g. max return; after each action decrement return to go
- robust for delayed rewards
- sequence modelling possible even without access to the reward function

questions: 
- we return-to-go modefizieren? 0 abziehen? oder im neuen state schatzen?
- sometimes assumes access to expert training data

 SM may be interesting!


### Offline Reinforcement Learning as One Big Sequence Modeling Problem
- discretization uniform or quantile (comparison 4.2)
- beam search
- rest same as DT

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
Divided into a) World model learning; b) sequential decision making; c) 
Papers:
Pre-Trained Language Models for Interactive Decision-Making
Online Decision Transformer
Prompting Decision Transformer for Few-Shot Policy Generalization
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting


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


