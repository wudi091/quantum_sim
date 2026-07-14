from qns.simulator.simulator import Simulator
from topo import Network
import random
from qns.simulator.simulator import Simulator
import qns.utils.log as log
import sys
import time

random.seed(0)

for i in range(0,100):
    a = time.time()
    net = Network(n=30, p = 0.04, reqs = 5)
    net.build()
    # net.route()
    b = time.time()
    print(i, b - a)

# n  p    i
# 20 0.05 21 6 25
# 30 0.05 2 0 5 14 18
# 30 0.04