# LLM informed Monte Carlo Tree Self Refine Algorithm for System Verilog Assertion Generation

## Methodology

Monte Carlo Tree Search is a reinforcement learning algorithm which searches the state space of the graph using a list of possible actions. In each rollout selects a node while balancing exploration and exploitation and selects the best action for that node based on some policy. It then simulates the rest of the game using random decisions and ultimately assignes a win/loss point to the node and each of its parent. After a fixed number of rollouts you get a graph of best actions at any particular time step in a Partially Observable Markov Decision Process.

![](https://www.researchgate.net/profile/Diego-Perez-Liebana/publication/274485244/figure/fig1/AS:294858334982145@1447311113569/MCTS-algorithm-steps.png)

This search algorithm can be used in combination with a LLM policy and a commonsense or finetuned model of the world that is provided by an LLM in order to exploit LLMs for better reasoning.

Such a search algorithm have been recently used for better reasoning with LLMs particularly on olympiad level mathematic reasoning datasets [1][2][3]. These reasoning strategy have even outperformed GPT4 on some datasets.

The general strategy that these papers used when using LLMs is as follows:
![](https://i.ibb.co/RcCKKPd/Screenshot-2024-10-24-125622.png)

## Algorithm Procedure Details

### Node Selection

All papers follow the same strategy of UCT algorithm (Upper confidence bound applied to Trees)
![](https://i2.paste.pics/69d6d5610de0b0b8cb99f4719056edc5.png?trs=ad25653d928f3ee6485e06e70882834d62814b847f2676f1a615622d09d200b0&rand=SszkrJ4Vdw)
![](https://i2.paste.pics/dabb4348a28fbe9029413c76ccce75a7.png?trs=ad25653d928f3ee6485e06e70882834d62814b847f2676f1a615622d09d200b0&rand=DxmSz0r7i8)

### Self-refine

We will use what is followed in Llama berry (https://arxiv.org/pdf/2410.02884) or
https://arxiv.org/pdf/2406.07394 (original paper)
![](https://i2.paste.pics/78ba0fd46c6c5de2997d2a329398d15c.png?trs=ad25653d928f3ee6485e06e70882834d62814b847f2676f1a615622d09d200b0&rand=hFKwlm7GYp)

### Self-evaluation

![](https://i2.paste.pics/3664cbf77aca0272d6c2b886ec93abea.png?trs=ad25653d928f3ee6485e06e70882834d62814b847f2676f1a615622d09d200b0&rand=qb7w3ASeJn)

### Backpropagation

Either one of the following 2 strategies will be used:

- ![](https://i2.paste.pics/2f665aeb1fd34d6f8d4f6bee65130c0b.png?trs=ad25653d928f3ee6485e06e70882834d62814b847f2676f1a615622d09d200b0&rand=d1qmZPblSh)
- ![](https://i2.paste.pics/a2d60840726e9891949eb02088995771.png?trs=ad25653d928f3ee6485e06e70882834d62814b847f2676f1a615622d09d200b0&rand=k9zyJndQ6F)

As of now, I have implemented the below strategy, but a very recent work using Pairwise Performance Reward Model [3] highlights the importane of focusing on future rewards more than current rewards and thus proposes the above backpropogation rule.

## Current Update

The methodology I will be following for SVA generation from specification is similar to above diagram.

I will be doing 20 rollouts in each simulation and 2 simulations per specification.

The dataset I will be using is the following:

- https://arxiv.org/pdf/2402.00386v1 (https://github.com/hkust-zhiyao/AssertLLM/tree/main)

LLM being used as base policy and world model: GPT4o or Llama 3.2 - 8b (depending on whether I can get access to GPT4o API or not (which is also the current blocker as the code is mostly ready)).

As of now, the MCTSr Algorithm Code is ready to be used with a Llama model.

The current blockers I am working on right now are:

- Possibly getting API access for GPT4o for better base policy.
- If using Llama, I need to shift all my computation to Param Kamrupa or CSE servers for computation.

## References

1. Zhang, D., et. al. (2024). Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report. arXiv preprint arXiv:2406.07394. https://arxiv.org/pdf/2406.07394
2. Gao, Z., Niu, B., He, X., Xu, H., Liu, H., Liu, A., Hu, X., & Wen, L. (2024). Interpretable Contrastive Monte Carlo Tree Search Reasoning. arXiv preprint arXiv:2410.01707. https://arxiv.org/pdf/2410.01707
3. Zhang, D., Wu, J., Lei, J., Che, T., Li, J., Xie, T., Huang, X., Zhang, S., Pavone, M., Li, Y., Ouyang, W., & Zhou, D. (2024). LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning. arXiv preprint arXiv:2410.02884. https://arxiv.org/pdf/2410.02884
