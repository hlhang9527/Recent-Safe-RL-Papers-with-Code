# A List of Recent Safe Reinforcement Learning Papers with Code
This repo  lists most recent paper with their code in safe RL, some of papers without available code is not included. Welcome to edit this list if more papers found. 

## Algorithms
### Safe Exploration

- [Safe Reinforcement Learning Using Advantage-Based Intervention](https://arxiv.org/abs/2106.09110), 2021 arxiv, [code](https://github.com/nolanwagener/safe_rl) 
- [Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones](https://arxiv.org/pdf/2010.15920.pdf), 2021 IEEE ROBOTICS AND AUTOMATION LETTERS, [code](https://github.com/abalakrishna123/recovery-rl)
- [Safe Reinforcement Learning via Curriculum Induction](https://paperswithcode.com/paper/safe-reinforcement-learning-via-curriculum), 2020 Nips, [code](https://github.com/zuzuba/CISR_NeurIPS20) 
- [AlwaysSafe: Reinforcement Learning without Safety Constraint Violations during Training](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1226.pdf), 2021 AAMAS, [code](https://github.com/AlgTUDelft/AlwaysSafe)
>Deploying reinforcement learning (RL) involves major concerns
around safety. Engineering a reward signal that allows the agent
to maximize its performance while remaining safe is not trivial.
Safe RL studies how to mitigate such problems. For instance, we
can decouple safety from reward using constrained Markov decision processes (CMDPs), where an independent signal models the
safety aspects. In this setting, an RL agent can autonomously find
tradeoffs between performance and safety. Unfortunately, most RL agents designed for CMDPs only guarantee safety after the learning
phase, which might prevent their direct deployment. In this work, we investigate settings where a concise abstract model of the safety
aspects is given, a reasonable assumption since a thorough understanding of safety-related matters is a prerequisite for deploying
RL in typical applications. Factored CMDPs provide such compact
models when a small subset of features describe the dynamics relevant for the safety constraints. We propose an RL algorithm that
uses this abstract model to learn policies for CMDPs safely, that is
without violating the constraints. During the training process, this
algorithm can seamlessly switch from a conservative policy to a
greedy policy without violating the safety constraints. We proveƒ
that this algorithm is safe under the given assumptions. Empirically,
we show that even if safety and reward signals are contradictory,
this algorithm always operates safely and, when they are aligned,
this approach also improves the agent’s performance.
- [Safe Reinforcement Learning in Constrained Markov Decision Processes](https://paperswithcode.com/paper/safe-reinforcement-learning-in-constrained), 2020 ICML, [code](https://github.com/akifumi-wachi-4/safe_near_optimal_mdp), [slide](https://icml.cc/media/icml-2020/Slides/5904.pdf)
- [A Safe and Fast Reinforcement Learning Safety Layer for Continuous](https://arxiv.org/abs/2011.08421), 2021 IEEE Robotics and Automation Letters, [code](https://github.com/roahmlab/reachability-based_trajectory_safeguard)
- [Safe Exploration in Continuous Action Spaces](https://arxiv.org/pdf/1801.08757.pdf),  2018 IEEE CDC, [code](https://github.com/AgrawalAmey/safe-explorer)
>We address the problem of deploying a reinforcement learning (RL) agent on 
>a physical system such as a datacenter cooling unit or robot, where critical 
>constraints must never be violated. We show how to exploit the typically smooth dynamics of these systems and enable RL algorithms to never violate constraints during learning. Our technique is to directly add to the policy a safety layer that analytically solves an action correction formulation per each state. The novelty of obtaining an elegant closed-form solution is attained due to a linearized model, learned on past trajectories consisting of arbitrary actions. This is to mimic the real-world circumstances where data logs were generated with a behavior policy that is implausible to describe mathematically; such cases render the known safety-aware off-policy methods inapplicable. We demonstrate the efficacy of our approach on new representative physics-based environments, and prevail where reward shaping fails by maintaining zero constraint violations.
- [Conservative Agency via Attainable Utility Preservation](https://dl.acm.org/doi/abs/10.1145/3375627.3375851), 2020 AAAI AES, [code](https://github.com/alexander-turner/attainable-utility-preservation)
>Reward functions are easy to misspecify; although designers can make corrections after observing mistakes, an agent pursuing a misspecified reward function can irreversibly change the state of its environment. If that change precludes optimization of the correctly specified reward function, then correction is futile. For example, a robotic factory assistant could break expensive equipment due to a reward misspecification; even if the designers immediately correct the reward function, the damage is done. To mitigate this risk, we introduce an approach that balances optimization of the primary reward function with preservation of the ability to optimize auxiliary reward functions. Surprisingly, even when the auxiliary reward functions are randomly generated and therefore uninformative about the correctly specified reward function, this approach induces conservative, effective behavior.
- [CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee](https://proceedings.mlr.press/v139/xu21a.html), 2021 ICML, no code 
- [Control Regularization for Reduced Variance Reinforcement Learning](http://proceedings.mlr.press/v97/cheng19a.html). 2019 ICML, [matlab code](https://github.com/rcheng805/CORE-RL)
>Dealing with high variance is a significant challenge in model-free reinforcement learning (RL). Existing methods are unreliable, exhibiting high variance in performance from run to run using different initializations/seeds. Focusing on problems arising in continuous control, we propose a functional regularization approach to augmenting model-free RL. In particular, we regularize the behavior of the deep policy to be similar to a policy prior, i.e., we regularize in function space. We show that functional regularization yields a bias-variance trade-off, and propose an adaptive tuning strategy to optimize this trade-off. When the policy prior has control-theoretic stability guarantees, we further show that this regularization approximately preserves those stability guarantees throughout learning. We validate our approach empirically on a range of settings, and demonstrate significantly reduced variance, guaranteed dynamic stability, and more efficient learning than deep RL alone.
- [Batch Policy Learning under Constraints](http://proceedings.mlr.press/v97/le19a.html), 2020 ICML, [code](https://github.com/clvoloshin/constrained_batch_policy_learning) 
- [Safe Exploration for Interactive Machine Learning](https://arxiv.org/abs/1910.13726), 2019 Nips, no code
- [Density Constrained Reinforcement Learning](http://proceedings.mlr.press/v139/qin21a.html), 2021 ICML, no code
- [Inverse Constrained Reinforcement Learning](https://arxiv.org/abs/2011.09999), 2021 ICML, [code](https://github.com/shehryar-malik/icrl)
- [Glas: Global-to-local safe autonomy synthesis for multi-robot motion planning with end-to-end learning](https://ieeexplore.ieee.org/abstract/document/9091314?casa_token=88NbMVeL7CoAAAAA:0LcjqPAgswvMtw1KhS2kN9m0TjwcOAc94XrPFfzUuKiKvYfFDHfHaC7I63CMvA_17-MWrTD8GUY), 2020 IEEE Robotics and Automation Letters,
[code](https://github.com/bpriviere/glas)
- [Constrained Markov Decision Processes via Backward Value Functions](http://proceedings.mlr.press/v119/satija20a.html), 2020 ICML, [code](https://github.com/hercky/cmdps_via_bvf)



### Safe Planning
- [Actor-Critic Reinforcement Learning for Control With Stability Guarantee](file:///Users/huiliangzhang/Downloads/09146733.pdf), 2020 IEEE ROBOTICS AND AUTOMATION LETTERS, 
[code](https://github.com/hithmh/Actor-critic-with-stability-guarantee)
>Reinforcement Learning (RL) and its integration with
deep learning have achieved impressive performance in various
robotic control tasks, ranging from motion planning and navigation
to end-to-end visual manipulation. However, stability is not guaranteed in model-free RL by solely using data. From a control-theoretic
perspective, stability is the most important property for any control
system, since it is closely related to safety, robustness, and reliability
of robotic systems. In this letter, we propose an actor-critic RL
framework for control which can guarantee closed-loop stability
by employing the classic Lyapunov’s method in control theory.
First of all, a data-based stability theorem is proposed for stochastic
nonlinear systems modeled by Markov decision process. Then we
show that the stability condition could be exploited as the critic
in the actor-critic RL to learn a controller/policy. At last, the
effectiveness of our approach is evaluated on several well-known
3-dimensional robot control tasks and a synthetic biology gene
network tracking task in three different popular physics simulation
platforms. As an empirical evaluation on the advantage of stability,
we show that the learned policies can enable the systems to recover
to the equilibrium or way-points when interfered by uncertainties
such as system parametric variations and external disturbances to
a certain extent.
- [Safe Planning via Model Predictive Shielding](https://arxiv.org/pdf/1905.10691.pdf), 2019 ACC, [code](https://github.com/obastani/model-predictive-shielding)
- [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods](http://proceedings.mlr.press/v119/stooke20a.html), 2020 ICML, [code](https://github.com/astooke/rlpyt/tree/master/rlpyt/projects/safe)
>Lagrangian methods are widely used algorithms for constrained optimization problems, but their learning dynamics exhibit oscillations and overshoot which, when applied to safe reinforcement learning, leads to constraint-violating behavior during agent training. We address this shortcoming by proposing a novel Lagrange multiplier update method that utilizes derivatives of the constraint function. We take a controls perspective, wherein the traditional Lagrange multiplier update behaves as \emph{integral} control; our terms introduce \emph{proportional} and \emph{derivative} control, achieving favorable learning dynamics through damping and predictive measures. We apply our PID Lagrangian methods in deep RL, setting a new state of the art in Safety Gym, a safe RL benchmark. Lastly, we introduce a new method to ease controller tuning by providing invariance to the relative numerical scales of reward and cost. Our extensive experiments demonstrate improved performance and hyperparameter robustness, while our algorithms remain nearly as simple to derive and implement as the traditional Lagrangian approach.
- [IPO: Interior-Point Policy Optimization under Constraints](https://ojs.aaai.org/index.php/AAAI/article/view/5932), 2020 AAAI, no code
### Policy Learning
- [Accelerating Safe Reinforcement Learning with Constraint-mismatched Baseline Policies](https://proceedings.mlr.press/v139/yang21i.html), 2021 ICML, no code
- [A Lyapunov-based Approach to Safe Reinforcement Learning](https://arxiv.org/pdf/1805.07708.pdf), 2018 Nips
[code](https://github.com/befelix/safe_learning)
>In many real-world reinforcement learning (RL) problems, besides optimizing 
>the main objective function, an agent must concurrently avoid violating a 
>number of constraints. In particular, besides optimizing performance it is 
>crucial to guarantee the safety of an agent during training as well as deployment (e.g. a robot should avoid taking actions - exploratory or not - which irrevocably harm its hardware). To incorporate safety in RL, we derive algorithms under the framework of constrained Markov decision problems (CMDPs), an extension of the standard Markov decision problems (MDPs) augmented with constraints on expected cumulative costs. Our approach hinges on a novel \emph{Lyapunov} method. We define and present a method for constructing Lyapunov functions, which provide an effective way to guarantee the global safety of a behavior policy during training via a set of local, linear constraints. Leveraging these theoretical underpinnings, we show how to use the Lyapunov approach to systematically transform dynamic programming (DP) and RL algorithms into their safe counterparts. To illustrate their effectiveness, we evaluate these algorithms in several CMDP planning and decision-making tasks on a safety benchmark domain. Our results show that our proposed method significantly outperforms existing baselines in balancing constraint satisfaction and performance.
- [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528), 2017 ICML, [code](https://paperswithcode.com/paper/constrained-policy-optimization)
- [Enforcing robust control guarantees within neural network policies](https://arxiv.org/abs/2011.08105), 2021 ICML, [code](https://github.com/locuslab/robust-nn-control)
- [Constrained Cross-Entropy Method for Safe Reinforcement Learning](https://proceedings.neurips.cc/paper/2018/file/34ffeb359a192eb8174b6854643cc046-Paper.pdf), 2018 Nips, [code](https://github.com/oscarkey/constrained-cem-mpc)

- [Constrained Model-based Reinforcement Learning with Robust Cross-Entropy Method](https://arxiv.org/abs/2010.07968), 2020 arxiv, [code](https://github.com/liuzuxin/safe-mbrl)
>This paper studies the constrained/safe reinforcement learning (RL) problem with sparse indicator signals for constraint violations. We propose a model-based approach to enable RL agents to effectively explore the environment with unknown system dynamics and environment constraints given a significantly small number of violation budgets. We employ the neural network ensemble model to estimate the prediction uncertainty and use model predictive control as the basic control framework. We propose the robust cross-entropy method to optimize the control sequence considering the model uncertainty and constraints. We evaluate our methods in the Safety Gym environment. The results show that our approach learns to complete the tasks with a much smaller number of constraint violations than state-of-the-art baselines. Additionally, we are able to achieve several orders of magnitude better sample efficiency when compared with constrained model-free RL approaches. The code is available at \url{this https URL}.

- [End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks](https://rcheng805.github.io/files/aaai2019.pdf), 2019 AAAI, [Code](https://github.com/rcheng805/RL-CBF)
- [Risk-Constrained Reinforcement Learning with Percentile Risk Criteria](https://arxiv.org/pdf/1512.01629.pdf), 2017 arxiv, no code
- [Convergent Policy Optimization for Safe Reinforcement Learning](https://arxiv.org/abs/1910.12156), 2019 Nips,
[code](https://github.com/ming93/Safe_reinforcement_learning)
>We study the safe reinforcement learning problem with nonlinear function 
>approximation, where policy optimization is formulated as a constrained 
>optimization problem with both the objective and the constraint being nonconvex functions. For such a problem, we construct a sequence of surrogate convex constrained optimization problems by replacing the nonconvex functions locally with convex quadratic functions obtained from policy gradient estimators. We prove that the solutions to these surrogate problems converge to a stationary point of the original nonconvex problem. Furthermore, to extend our theoretical results, we apply our algorithm to examples of optimal control and multi-agent reinforcement learning with safety constraints.
- [Lyapunov-based Safe Policy Optimization for Continuous Control](https://arxiv.org/pdf/1901.10031.pdf)
- [Neural Lyapunov Control](http://papers.nips.cc/paper/8587-neural-lyapunov-control.pdf)
- [CAQL: CONTINUOUS ACTION Q-LEARNING](https://arxiv.org/pdf/1909.12397.pdf), 2020 ICLR, no code

### Safe Reinforcement Learning with Stability Guarantees
- [Safe Model-based Reinforcement Learning with Stability Guarantees](https://papers.nips.cc/paper/6692-safe-model-based-reinforcement-learning-with-stability-guarantees.pdf)
- [The Lyapunov Neural Network: Adaptive Stability Certification for Safe Learning of Dynamical Systems](https://arxiv.org/pdf/1808.00924.pdf)
- [Safe Learning of Regions of Attraction for Uncertain, Nonlinear Systems with Gaussian Processes](https://arxiv.org/pdf/1603.04915.pdf), 2017 CDC
- [Code](https://github.com/befelix/safe_learning)

### Human in the loop
- [Trial without Error: Towards Safe Reinforcement Learning via Human Intervention](https://arxiv.org/abs/1707.05173)

## Surveys
- [A Comprehensive Survey on Safe Reinforcement Learning](http://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf)
- [Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning](https://paperswithcode.com/paper/safe-learning-in-robotics-from-learning-based), 2021, [code](https://github.com/utiasDSL/safe-control-gym)
## Benchmarks
- [Benchmarking Safe Exploration in Deep Reinforcement Learning](https://d4mucfpksywv.cloudfront.net/safexp-short.pdf)

## Thesis
[SAFE REINFORCEMENT LEARNING](https://people.cs.umass.edu/~pthomas/papers/Thomas2015c.pdf)

## Lectures
[Safe Reinforcement Learning](https://web.stanford.edu/class/cs234/slides/2017/cs234_guest_lecture_safe_rl.pdf)

