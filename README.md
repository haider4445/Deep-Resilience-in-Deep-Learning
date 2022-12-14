# Deep-Resilience-in-Deep-Learning
### ACADIA: Efficient and Robust Adversarial Attacks Against Deep Reinforcement Learning
Haider Ali, Mohannad Al Ameedi, Ananthram Swami, Rui Ning, Jiang Li, Hongyi Wu, Jin-Hee Cho

### Abstract:
Existing adversarial algorithms for Deep Reinforcement Learning (DRL) have largely focused on identifying an
optimal time to attack a DRL agent. However, little work has
been explored in injecting efficient adversarial perturbations
in DRL environments. We propose a suite of novel DRL ad-
versarial attacks, called ACADIA, representing AttaCks Against
Deep reInforcement leArning. ACADIA provides a set of efficient
and robust perturbation-based adversarial attacks to disturb the
DRL agent’s decision-making based on novel combinations of
techniques utilizing momentum, ADAM optimizer (i.e., Root Mean
Square Propagation, or RMSProp), and initial randomization.
These kinds of DRL attacks with novel integration of such
techniques have not been studied in the existing Deep Neural
Networks (DNNs) and DRL research. We consider two well-known
DRL algorithms, Deep-Q Learning Network (DQN) and Proximal
Policy Optimization (PPO), under Atari games and MuJoCo
where both targeted and non-targeted attacks are considered with
or without the state-of-the-art defenses in DRL (i.e., RADIAL
and ATLA). Our results demonstrate that the proposed ACADIA
outperforms existing gradient-based counterparts under a wide
range of experimental settings. ACADIA is nine times faster than
the state-of-the-art Carlini & Wagner (CW) method with better
performance under defenses of DRL.


**[ACADIA](https://github.com/haider4445/Deep-Resilience-in-Deep-Learning/blob/main/ACADIA_IEEE_CNS.pdf) got accepted to IEEE Conference on Communications and Network Security (CNS) 2022.**

### BibTex citation:

@article{ali2022acadia,
  title={ACADIA: Efficient and Robust Adversarial Attacks Against Deep Reinforcement Learning},
  author={Ali, Haider and Al Ameedi, Mohannad and Swami, Ananthram and Ning, Rui and Li, Jiang and Wu, Hongyi and Cho, Jin-Hee },
  journal={2022 IEEE Conference on Communications and Network Security (CNS)},
  year={2022},
  organization={IEEE}
}


