# MCTS

good tutorial for mcts in go: https://jonathan-hui.medium.com/monte-carlo-tree-search-mcts-in-alphago-zero-8a403588276a


# AlphaZe
to do: lesen:
-sample possible states from the current input and runs MCTS combined with the neural network, like described in the work of Silver et al. (2016).
- crazy ara framework (opt. c++)
- OpenSpiel (Lanctot et al., 2019) 
- architecture RISEv2-mobile as introduced in Czech et al. (2020

PAPER::::::::::::
new method: policz combining pimc
intro::
- pimc perf. info mc memory consumption scales better than cfr
- tsl true sight learning_:early stage acc

(POMDPs)MDPs and not POMDPs) to be more efficient, turning imperfect state information into perfect information via sampling

Pipeline Policy-Space Response Oracles (P2SRO; McAleer et al., 2020) and DeepNash (Perolat et al., 2022)
PSRO

information set
Some well-known approaches in this field are Counterfactual Regret Minimization (Zinkevich et al., 2007; Burch et al., 2012; Brown et al., 2019) and Fictitious Play (Heinrich et al., 2015; Heinrich and Silver, 2016).

 ensemble techniques such as bagging

? algo 2 mcts
? pi correct policy, p predicted one algo 3

2 inspirations:
Recursive Belief-based Learning (Brown et al., 2020) and Partially Observable Monte-Carlo Planning (Silver and Veness, 2010).

meine idee: 
zu game states samplen von information set:
encoder decoder architecture - encodes info set into representation, decodes into perfect information game states - prefers states that are more porbable

Deep Counterfactual Regret Minimization (nach Pluribus)
http://proceedings.mlr.press/v97/brown19b/brown19b.pdf

MCCFR (Burch et al., 2012)

Another method that tackles the problem of uncertainty is Information Set MCTS (ISMCTS; Cowling et al., 2012), which constructs a game tree with each node representing an information set instead of a specific board position. Edges correspond to actions between information sets from the point of view of the player who plays them if we treat all moves as fully observable. This makes the computation less budget-heavy and improves the decision-making process compared to other methods like determinization. Adaptations of ICMCTS, such Many-Tree ISMCTS (Cowling et al., 2015) and Semi ISMCTS (Bitan and Kraus, 2017) advance the idea of ISMCTS. In particular, Semi ISMCTS, which tries to combine the advantages of PIMC and ISMCTS, could be interesting for future work. However, due to their complexity and their distance from the classical MCTS, they contradict our idea of a simple adaptation.


## Silver(2016) Mastering the Game of Go with Deep Neural Networks and Tree Search
ansatz: selecting moves/positions using policy net, evaluating them using value net, combine value with mc rollouts

#### 1. Supervised Learning Policy Network
- conv. layers and relu, final softmax (13 layers)
- supervised (state, action) pair (30 million), maximize likelihood of action (achieved 57% acc.)

#### 2. RL Policy
- train against previous iterations 
- sga max expected outcome loglikelihood * reward
- sample moves from distribution, not maximum
- also small fast policy 

#### RL Value Network
- predicts outcome from position (if both players play according to the same policy p)
- sgd on regression  on (state, outcome) pairs, min mse
- use only one sample per game, else overfitting
-  value network more accurate than monte carlo rollouts using way less computations

#### Searching: MCTS with Value and Policy Network
- action value Q(s,a), visit count, prior probability P(s,a)
- Select action: argmax(Q + u), u=P/1+N
- New leaf -> Expand tree: 
1. P(s,a) = p_sig(s,a) : SL policy network -> prior probabilities for each legal move
2. v_theta : call value network for the leaf node -> v_theta
3. z_L <- p_pi : fast rollout using the fast (small) policy -> z_L
4. V = lam * z_L + (1-lam) * z_L : conv combine values of 2 and 3  -> V (lambda = 0.5)
5. update Q: 1/N * sum(V)  (avg V)

execute simulations on cpus and network evaluations on gpus in paralell

stopped: Selection (Figure 4a).

## Alpha Zero (https://www.science.org/doi/full/10.1126/science.aar6404?casa_token=4qwwONQWeL0AAAAA%3AKqUW01n3YdSoacToD4bHddWeD6Ukl5wcCNID_8SwcYfcZE6s6T0EkUxoBGrR14N5UQ6H07blOT3lfMY)

- (p,v)=f_theta(s) 
- min mse and cross-entr. loss : l=(z-v)^2-pi^Tlogp + c norm(theta)
- only update wheights if new wins 55% against best player
- board state, actions encoded as spatial planes? anhang?
- same conv arch. as AlphaGo
- Hyperparameter tuning : bazsian optimization!

#### MCTS Algo for AlphaZero:
(s,a) state action pair stores:
- (N,W,V,P): N visit count, W total action value, Q mean value, P prior probability
- PUCT variation: a_t = argmax Q + U, where U =C(s)P(s,a) sqrt(N)/1+N, where C is exploration rate that grows slowly with search time (or constant)
- simulate until leaf node s_L, then:
1. call net (p,v)=f_theta(s_L)
2. expand leaf node: init pairs (s_L,a) with (0,0,0, p_a) 
3. backwardpass: N(s_t,a)++, W+=v, Q=W/N

input representations and action representation

architecture: conv batch norm rectified residual with skip conn

configutation?

800 simulations per move

## Analysis of AlphaZero 
https://arxiv.org/pdf/1902.04522v5.pdf



###MCTRANSFORMER: COMBINING TRANSFORMERS
AND MONTE-CARLO TREE SEARCH FOR OFFLINE REINFORCEMENT LEARNING
https://openreview.net/pdf?id=-94tJCOo7OM





#### Brown(2020): Combining Deep Reinforcement Learning and Search
for Imperfect-Information Games
- https://github.com/facebookresearch/rebel
- public belief states
- Related Work:
1. Pomdps for public belief states
2. deepstack poker AI (2017): sample PBSs using exerpt knowledge
3. pluribus poker ai: train population of ais without search then choose best during search?
There exist a number of iterative algorithms for solving imperfect-information games [5, 61, 28,
36, 35]. We describe ReBeL assuming the counterfactual regret minimization - decomposition
(CFR-D) algorithm is used [61, 16, 40].


















MCTS Tutorial https://zsalloum.medium.com/monte-carlo-in-reinforcement-learning-the-easy-way-564c53010511



## AI Blogs from netflix, meta & co
https://netflixtechblog.com/
https://www.tripadvisor.com/engineering/
https://blog.duolingo.com/tag/engineering/
https://medium.com/meta-analytics
https://engineering.atspotify.com/category/data-science/
https://www.deeplearning.ai/the-batch/

