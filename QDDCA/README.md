# Q-DDCA: Decentralized Dynamic Congestion  Avoid Routing in Large-Scale Quantum Networks

This is the minized prototype codes for implementing the Q-DDCA protocol. This code requires the [SimQN](https://github.com/QNLab-USTC/SimQN) Platform.

## How to Run

1. ``python3 exp1.py``
2. ``python3 exp2.py``
3. ``python3 exp3.py``

NOTE: They will create files to log the results.

## Details

run *exp1.py*, *exp2.py* or *exp3.py* for simulation, and modify the parameters to collect the complete set of results

- *exp1.py*

  Simulate the network's throughput, drop rate, and coefficient of variation in both single-request and multiple-request scenarios under different sending window sizes and specific maximum retry attempts.

- *exp2.py*

  Simulate the network's throughput and drop rate in a single-request scenario across different maximum retry attempts and multiple specific sending window sizes.

- *exp3.py*

  Simulate the memory occupancy of req1 and req2 at node u3 over time, under the given network topology.

## Note and Citation

- Please cite:"Chen L, Xue K, Li J, et al. Q-DDCA: Decentralized dynamic congestion avoid routing in large-scale quantum networks[J]. IEEE/ACM Transactions on Networking, 2023, 32(1): 368-381."[link](https://ieeexplore.ieee.org/abstract/document/10158747)
  
- Please add the following citation in your work if you use our open-source code.
```
@article{chen2023q,
  title={Q-DDCA: Decentralized dynamic congestion avoid routing in large-scale quantum networks},
  author={Chen, Lutong and Xue, Kaiping and Li, Jian and Li, Ruidong and Yu, Nenghai and Sun, Qibin and Lu, Jun},
  journal={IEEE/ACM Transactions on Networking},
  volume={32},
  number={1},
  pages={368--381},
  year={2023},
  publisher={IEEE}
}
```
