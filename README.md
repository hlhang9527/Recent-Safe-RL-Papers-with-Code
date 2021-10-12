# A List of Recent Safe Reinforcement Learning Papers
This repo lists most recent papers with their code in safe RL, some papers without available code are not included. Welcome to change this list if additional documents found.
## Algorithms
### Safe Exploration
- [Conservative Exploration in Reinforcement Learning](http://proceedings.mlr.press/v108/garcelon20a.html), International Conference on Artificial Intelligence and Statistics 2020
- [Projection-based Constrained Policy Optimization (PCPO)](https://openreview.net/forum?id=M3NDrHEGyyO), 2020 ICLR, no code.
- [Accelerating Safe Reinforcement Learning with Constraint-mismatched Baseline Policies](https://proceedings.mlr.press/v139/yang21i.html), 2021 ICML, [code](https://sites.google.com/view/spacealgo)
- [A Lyapunov-based Approach to Safe Reinforcement Learning](https://arxiv.org/pdf/1805.07708.pdf), 2018 Nips
[code](https://github.com/befelix/safe_learning)
- [Provably efficient safe exploration via primal-dual policy optimization](http://proceedings.mlr.press/v130/ding21d/ding21d.pdf), 2021 ICML, no code, [slide](https://slideslive.com/s/dongsheng-ding-24136)

- [LS3: Latent Space Safe Sets for Long-Horizon Visuomotor Control of Sparse Reward Iterative Tasks](https://arxiv.org/pdf/2107.04775.pdf), 2021 arxiv, [code](https://github.com/albertwilcox/latent-space-safe-sets)
- [Learning Barrier Certificates: Towards Safe Reinforcement Learning with Zero Training-time Violations](https://arxiv.org/pdf/2108.01846.pdf), 2021 arxiv, no code
- [Safe Reinforcement Learning Using Advantage-Based Intervention](https://arxiv.org/abs/2106.09110), 2021 ICML, [code](https://github.com/nolanwagener/safe_rl) 
- [Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones](https://arxiv.org/pdf/2010.15920.pdf), 2021 IEEE ROBOTICS AND AUTOMATION LETTERS, [code](https://github.com/abalakrishna123/recovery-rl)
- [Safe Reinforcement Learning via Curriculum Induction](https://paperswithcode.com/paper/safe-reinforcement-learning-via-curriculum), 2020 Nips, [code](https://github.com/zuzuba/CISR_NeurIPS20) 
- [AlwaysSafe: Reinforcement Learning without Safety Constraint Violations during Training](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1226.pdf), 2021 AAMAS, [code](https://github.com/AlgTUDelft/AlwaysSafe)
- [Safe Reinforcement Learning in Constrained Markov Decision Processes](https://paperswithcode.com/paper/safe-reinforcement-learning-in-constrained), 2020 ICML, [code](https://github.com/akifumi-wachi-4/safe_near_optimal_mdp), [slide](https://icml.cc/media/icml-2020/Slides/5904.pdf)
- [A Safe and Fast Reinforcement Learning Safety Layer for Continuous](https://arxiv.org/abs/2011.08421), 2021 IEEE Robotics and Automation Letters, [code](https://github.com/roahmlab/reachability-based_trajectory_safeguard)
- [Safe Exploration in Continuous Action Spaces](https://arxiv.org/pdf/1801.08757.pdf),  2018 IEEE CDC, [code](https://github.com/AgrawalAmey/safe-explorer)
- [Conservative Agency via Attainable Utility Preservation](https://dl.acm.org/doi/abs/10.1145/3375627.3375851), 2020 AAAI AES, [code](https://github.com/alexander-turner/attainable-utility-preservation)
- [CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee](https://proceedings.mlr.press/v139/xu21a.html), 2021 ICML, no code 
- [Control Regularization for Reduced Variance Reinforcement Learning](http://proceedings.mlr.press/v97/cheng19a.html). 2019 ICML, [matlab code](https://github.com/rcheng805/CORE-RL)
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
- [Safe Planning via Model Predictive Shielding](https://arxiv.org/pdf/1905.10691.pdf), 2019 ACC, [code](https://github.com/obastani/model-predictive-shielding)
- [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods](http://proceedings.mlr.press/v119/stooke20a.html), 2020 ICML, [code](https://github.com/astooke/rlpyt/tree/master/rlpyt/projects/safe)
- [IPO: Interior-Point Policy Optimization under Constraints](https://ojs.aaai.org/index.php/AAAI/article/view/5932), 2020 AAAI, no code
### Policy Learning

- [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528), 2017 ICML, [code](https://paperswithcode.com/paper/constrained-policy-optimization)
- [Enforcing robust control guarantees within neural network policies](https://arxiv.org/abs/2011.08105), 2021 ICML, [code](https://github.com/locuslab/robust-nn-control)
- [Constrained Cross-Entropy Method for Safe Reinforcement Learning](https://proceedings.neurips.cc/paper/2018/file/34ffeb359a192eb8174b6854643cc046-Paper.pdf), 2018 Nips, [code](https://github.com/oscarkey/constrained-cem-mpc)

- [Constrained Model-based Reinforcement Learning with Robust Cross-Entropy Method](https://arxiv.org/abs/2010.07968), 2020 arxiv, [code](https://github.com/liuzuxin/safe-mbrl)

- [End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks](https://rcheng805.github.io/files/aaai2019.pdf), 2019 AAAI, [Code](https://github.com/rcheng805/RL-CBF)
- [Risk-Constrained Reinforcement Learning with Percentile Risk Criteria](https://arxiv.org/pdf/1512.01629.pdf), 2017 arxiv, no code
- [Convergent Policy Optimization for Safe Reinforcement Learning](https://arxiv.org/abs/1910.12156), 2019 Nips,
[code](https://github.com/ming93/Safe_reinforcement_learning)
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


## Applications
- [Safe reinforcement learning using risk mapping by similarity](https://journals.sagepub.com/doi/full/10.1177/1059712319859650?casa_token=6BNIZh3WTNQAAAAA%3ADyP_V7fqfFw8EtEUaXPJRek5GytQvozallCQK5PZB642fZDRsFIWsyojC0hgbQdbCHdQNZltjaqMAr4)
- [Autonomous navigation via deep reinforcement learning for resource constraint edge nodes using transfer learning](https://ieeexplore.ieee.org/abstract/document/8978577)
- [Safe deep reinforcement learning-based constrained optimal control scheme for active distribution networks]
- [Deep reinforcement learning with reference system to handle constraints for energy-efficient train control](https://www.sciencedirect.com/science/article/pii/S0020025521004291?casa_token=IntFGhMSP5QAAAAA:2E2XZ0WaBZjL9QSdLLCC0qPfK1pLVUR3bSxJSHf4z_pHj8B9y86bcTXR4N9VFd8dWszdgyxdgexC)
- [Deep reinforcement learning driven inspection and maintenance planning under incomplete information and constraints](https://ieeexplore.ieee.org/abstract/document/9310351?casa_token=ij05d_SG1BsAAAAA:Jea6CZG_a86irUUsBo15dPspnqCsiPITTNUrba62M6ECgvlPCDYoVCgQcK_vu1OGRyG9fMPSBiE)
- [Deep-Reinforcement-Learning-Based Capacity Scheduling for PV-Battery Storage System](https://ieeexplore.ieee.org/abstract/document/9310351?casa_token=ij05d_SG1BsAAAAA:Jea6CZG_a86irUUsBo15dPspnqCsiPITTNUrba62M6ECgvlPCDYoVCgQcK_vu1OGRyG9fMPSBiE)
- [Multi-Agent Safe Policy Learning for Power Management of Networked Microgrids](https://ieeexplore.ieee.org/abstract/document/9244070?casa_token=YGeQ-psZnSIAAAAA:0ewUNYMNfFybmsnR3MTvpbsORQ8nZe5ReX4_cdRCTfynwfJZpxYvoLNz6_LBHez_QH5JfTAm6Gc)
- [Constrained EV Charging Scheduling Based on Safe Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/8910361?casa_token=6VikMwPZGgwAAAAA:v6XtBm2bJ-AGtqJeinqwOTXQ10G_mUM7YbGTo8OqIR-6Bd-9SCshidm6U0uBefejD2JdxWrvgeU)
- [Constrained Dual-Level Bandit for Personalized Impression Regulation in Online Ranking Systems](https://dl.acm.org/doi/fullHtml/10.1145/3461340)
- [Estimating and Penalizing Preference Shift in Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3460231.3478849)
- [Short-term wind speed forecasting using deep reinforcement learning with improved multiple error correction approach](https://www.sciencedirect.com/science/article/pii/S0360544221023768?casa_token=-Be0aQLq640AAAAA:GhRzGMioBFB_ZnWGtGLiWMtcTjgsqkQd969u6jp-ORwjADhPbazlLfY-0bhCxzpgyt68Jc8UbQQ6)
- [An Improved Reinforcement Learning for Security-Constrained Economic Dispatch of Battery Energy Storage in Microgrids](https://link.springer.com/chapter/10.1007/978-981-16-5188-5_22)
- [Safe Reinforcement Learning for Emergency LoadShedding of Power Systems](https://arxiv.org/abs/2011.09664)
- [Barrier Function-based Safe Reinforcement Learning for Emergency Control of Power Systems](https://arxiv.org/abs/2103.14186)
- [Safe reinforcement learning-based resilient proactive scheduling for a commercial building considering correlated demand response](https://ieeexplore.ieee.org/abstract/document/9371751)
- [A Learning-based Optimal Market Bidding Strategy for Price-Maker Energy Storage](https://arxiv.org/abs/2106.02396)
- [Energy-Efficient Secure Video Streaming in UAV-Enabled Wireless Networks: A Safe-DQN Approach](https://ieeexplore.ieee.org/abstract/document/9475975?casa_token=C2XRMAYDyywAAAAA:YA44uHtZya3Jqvs3DwrT42anOFECyjtH2bZ_yGMpxsxbFMpiFasPoP9vZ6Lbifk1lvGFAtkGfU4)
- [Trajectory Optimization for UAV Emergency Communication with Limited User Equipment Energy: A safe-DQN Approach](https://ieeexplore.ieee.org/abstract/document/9385412?casa_token=6VQg4qiD46QAAAAA:XdBL7C5Jq9D3weiL_ma3YwmnWVfmK2mdAOJyGL0ZHfr4-BEttnvWscyzE4DrSak76IbNKfEWrnw)
- [Optimal energy management strategies for energy Internet via deep reinforcement learning approach](https://www.sciencedirect.com/science/article/pii/S0306261919301746?casa_token=5Djio1TTOAYAAAAA:mO5QLb9pggLLk9iwop_PL5v0JnDtC0jo-dZKh7N-xLSc7wn1HqDCbelUBP4Jwybad4wLwC09pSzu)
## Combine with other methods:
- [Provably Safe Model-Based Meta Reinforcement Learning: An Abstraction-Based Approach](https://arxiv.org/abs/2109.01255)
- [Context-Aware Safe Reinforcement Learning for Non-Stationary Environments](https://arxiv.org/pdf/2101.00531.pdf), 2021 arxiv, no code, meta-learning
- [Learning to be Safe: Deep RL with a Safety Critic](https://arxiv.org/pdf/2010.14603.pdf), 2020 arxiv, no code, transfer learning
- [Safe exploration of nonlinear dynamical systems: A predictive safety filter for reinforcement learning](https://www.researchgate.net/profile/Kim-Wabersich/publication/329641554_Safe_exploration_of_nonlinear_dynamical_systems_A_predictive_safety_filter_for_reinforcement_learning/links/5ede2aab299bf1d20bd87981/Safe-exploration-of-nonlinear-dynamical-systems-A-predictive-safety-filter-for-reinforcement-learning.pdf),
no code, arxiv
- [REINFORCEMENT LEARNING WITH SAFE EXPLORATION FOR NETWORK SECURITY](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8682983&casa_token=GeE4bpTF4HgAAAAA:NJQIciHEQmPecuZFKdeutprlK17SIkwZcf70ac-qI-LWbe_qeHIVTzAAo2ZEzxDf41DQsrFE4w&tag=1),
no code
- [Continuous Safe Learning Based on First Principles and Constraints for
Autonomous Driving](http://ceur-ws.org/Vol-2560/paper29.pdf), no code
- [Blind Spot Detection for Safe Sim-to-Real Transfer](https://www.jair.org/index.php/jair/article/view/11436), [code](https://github.com/ramya-ram/discovering-blind-spots)
- [UAV-aided cellular communications with deep reinforcement learning against jamming](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9170648&casa_token=mubTX-ORnokAAAAA:cW2QQrilGVmGgf_tKBiwmBBBIvqeGNM30ujTlZYrgShSx2_l0Id-r-Dtaz7Oium2WksPYeLefQ), no code

- [Safe Imitation Learning via Fast Bayesian Reward Inference from Preferences](http://proceedings.mlr.press/v119/brown20a/brown20a.pdf), ICML 2020, [code](https://github.com/dsbrown1331/bayesianrex)

- [Safe policy improvement with baseline bootstrapping](http://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf), ICML 2019
- [Safe policy improvement with baseline bootstrapping in factored environments](https://ojs.aaai.org/index.php/AAAI/article/view/4427), aaai 2020
## Surveys
- [A Comprehensive Survey on Safe Reinforcement Learning](http://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf)
- [Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning](https://paperswithcode.com/paper/safe-learning-in-robotics-from-learning-based), 2021, [code](https://github.com/utiasDSL/safe-control-gym)
## Benchmarks
- [Benchmarking Safe Exploration in Deep Reinforcement Learning](https://d4mucfpksywv.cloudfront.net/safexp-short.pdf)

## Thesis
[SAFE REINFORCEMENT LEARNING](https://people.cs.umass.edu/~pthomas/papers/Thomas2015c.pdf)

## Lectures
[Safe Reinforcement Learning](https://web.stanford.edu/class/cs234/slides/2017/cs234_guest_lecture_safe_rl.pdf)

## Classical paper 
- Sui, Y., Gotovos, A., Burdick, J. W., and Krause, A. Safe
exploration for optimization with Gaussian processes. In
International Conference on Machine Learning (ICML), 2015.
- Turchetta, M., Berkenkamp, F., and Krause, A. Safe exploration
in finite Markov decision processes with Gaussian
processes. In Neural Information Processing Systems
(NeurIPS), 2016. [code](https://github.com/befelix/SafeMDP)
- Wachi et al. "Safe Exploration and Optimization of
Constrained MDPs using Gaussian Processes." AAAI 2018. no code
## Theory
### Robust control
- S. Bansal, M. Chen, S. Herbert, and C. J. Tomlin, “Hamilton-jacobi
reachability: A brief overview and recent advances”, in Conference on
Decision and Control (CDC), 2017.
- S. Li and O. Bastani, “Robust model predictive shielding for safe
reinforcement learning with stochastic dynamics”, in Proc. IEEE Int.
Conf. Robotics and Automation (ICRA), 2020.
- J. F. Fisac, A. K. Akametalu, M. N. Zeilinger, S. Kaynama, J.
Gillula, and C. J. Tomlin, “A general safety framework for learningbased
control in uncertain robotic systems”, in IEEE Transactions on
Automatic Control, 2018.
- J. H. Gillula and C. J. Tomlin, “Guaranteed safe online learning via
reachability: Tracking a ground target using a quadrotor”, in Proc.
IEEE Int. Conf. Robotics and Automation (ICRA), 2012.
- E. Altman, Constrained Markov Decision Processes.1999, p. 260.
