from topo import Network
import random
from qns.simulator.simulator import Simulator
import qns.utils.log as log
import sys
from collections import Counter
import numpy as np

random.seed(1225)
# random.seed(120)

s = Simulator(0, 10, 10000)
# log.set_debug(True)
log.install(s)

windowSize = int(sys.argv[1])
send_max_try = int(sys.argv[2])
reroute = True if sys.argv[3] == "1" else False

# windowSize = 10
# send_max_try = 100
# reroute = True

net = Network(n=5, p = 0.05, reqs = 1, memorySize=10, windowSize = windowSize, queryTime= 0.1, send_max_try= send_max_try, rate = 1000, delay = 0.1, allow_reroute=reroute)
net.install(s)

net.print_route_table()
print(net.route_table[net.get_node("n5")][net.get_node("n1")])
# net.s = [net.get_node("n22")]
# net.s = [net.get_node("n27")]
# net.get_node("n22").isSender = True
# net.get_node("n22").dest = net.get_node("n27")

s.run()

ans_list = []
# print(net.s, net.d)
for s in net.s:
    c = Counter([tuple(x.route) for x in s.sendedList])
    print(f"result on {s}->{s.dest}: sended {len(s.sendedList)} drop {len(s.dropList)} sending {len(s.sendingList)}", end=" ")
    print(c)
    ans_list.append(len(s.sendedList))

print(windowSize, sum(ans_list), np.average(ans_list), np.std(ans_list))


    # if len(s.dropList) > 0:
    #     log.info(f"drop route: {s.dropList[0].route}")

    # for q in s.sendingList:
    #     print(q.name, q.src, q.curr, q.dest, q.birth, q.route)
    
    # for q in s.sendedList:
    #     print(q.name, q.src, q.curr, q.dest, q.birth, q.route)

# for n in net.nodes:
#     ans = {}

#     for q in n.memory:
#         try:
#             ans[q.src] += 1
#         except:
#             ans[q.src] = 1

#     print(f"node {n} memory {n.currentSize}: {ans}")

