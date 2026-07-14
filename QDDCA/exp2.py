from topo import Network
import random
from qns.simulator.simulator import Simulator
import qns.utils.log as log
import sys
from collections import Counter
# import numpy as np

random.seed(2)
# log.set_debug(True)
# random.seed(2)
randomstate = random.getstate()

f = open("output/exp1-4.csv","w", buffering=1)

for w in [10, 20, 30]:
    for reroute in [False, True]:
        for m in range(1, 11):
            random.setstate(randomstate)

            s = Simulator(0, 10, 1000)
            log.install(s)
            net = Network(n=50, p = 0.1, reqs = 1, memorySize=20, windowSize = w, queryTime= 0.5/m, send_max_try= m, rate = 1000, delay = 0.001, allow_reroute=reroute, random_memory=False)
            # net = Network(n=50, p = 0.1, reqs = 1, memorySize=20, windowSize = w, queryTime= 0.1, send_max_try= m, rate = 1000, delay = 0.1, allow_reroute=reroute)

            net.install(s)
            s.run()

            ans_list = []
            drop_list = []
            # print(net.s, net.d)
            for s in net.s:
                c = Counter([tuple(x.route) for x in s.sendedList])
                # print(f"result on {s}->{s.dest}: sended {len(s.sendedList)} drop {len(s.dropList)} sending {len(s.sendingList)}", end=" ")
                # print(c)
                ans_list.append(len(s.sendedList))
                drop_list.append(len(s.dropList))

            f.write(f"{w},{m},{reroute},{sum(ans_list)},{sum(drop_list)},{len(c)},\"{c}\"\n")

            print(reroute, w, m , sum(ans_list) , sep=",")
f.close()
