from topo import Network
import random
from qns.simulator.simulator import Simulator
import qns.utils.log as log
import sys
from collections import Counter
from qns.entity.timer.timer import Timer

from entity import QNNode, Link
import numpy as np
random.seed(1)
randomstate = random.getstate()

f = open("output/tmp_2.csv","w", buffering=1)

class FixTopoNetwork(Network):
    def build(self):
        self.n = 6
        self.nodes = []
        self.links = {}
        self.route_table = {}


        n1: QNNode = QNNode("n1", memorySize=self.memorySize, windowSize=self.windowSize,
                             queryTime=self.queryTime, send_max_try=self.send_max_try, allow_reroute=self.allow_reroute, random_memory= self.random_memory)
        n2: QNNode = QNNode("n2", memorySize=self.memorySize, windowSize=self.windowSize,
                             queryTime=self.queryTime, send_max_try=self.send_max_try, allow_reroute=self.allow_reroute, random_memory= self.random_memory)
        n3: QNNode = QNNode("n3", memorySize=self.memorySize, windowSize=self.windowSize,
                             queryTime=self.queryTime, send_max_try=self.send_max_try, allow_reroute=self.allow_reroute, random_memory= self.random_memory)
        n4: QNNode = QNNode("n4", memorySize=self.memorySize, windowSize=self.windowSize,
                             queryTime=self.queryTime, send_max_try=self.send_max_try, allow_reroute=self.allow_reroute, random_memory= self.random_memory)
        n5: QNNode = QNNode("n5", memorySize=self.memorySize, windowSize=self.windowSize,
                             queryTime=self.queryTime, send_max_try=self.send_max_try, allow_reroute=self.allow_reroute, random_memory= self.random_memory)
        n6: QNNode = QNNode("n6", memorySize=self.memorySize, windowSize=self.windowSize,
                             queryTime=self.queryTime, send_max_try=self.send_max_try, allow_reroute=self.allow_reroute, random_memory= self.random_memory)
        n7: QNNode = QNNode("n7", memorySize=self.memorySize, windowSize=self.windowSize,
                     queryTime=self.queryTime, send_max_try=self.send_max_try, allow_reroute=self.allow_reroute, random_memory= self.random_memory)
        # n8: QNode = QNode("n8", memorySize=self.memorySize, windowSize=self.windowSize,
        #                      queryTime=self.queryTime, send_max_try=self.send_max_try, allow_reroute=self.allow_reroute, random_memory= self.random_memory)
   
        # self.nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
        self.nodes = [n1, n2, n3, n4, n5, n6, n7]
        n1.set_net(self)
        n2.set_net(self)
        n3.set_net(self)
        n4.set_net(self)
        n5.set_net(self)
        n6.set_net(self)
        n7.set_net(self)
        # n8.set_net(self)
        self.links[n1] = []
        self.links[n2] = []
        self.links[n3] = []
        self.links[n4] = []
        self.links[n5] = []
        self.links[n6] = []
        self.links[n7] = []
        # self.links[n8] = []

        l1 = Link(name=n1.name+"-"+n3.name, nodes=[n1, n3], rate=self.rate, delay=self.delay, buffer=self.buffer)
        self.links[n1].append((n3, l1))
        self.links[n3].append((n1, l1))
        l2 = Link(name=n2.name+"-"+n7.name, nodes=[n2, n7], rate=self.rate, delay=self.delay, buffer=self.buffer)
        self.links[n2].append((n7, l2))
        self.links[n7].append((n2, l2))
        l3 = Link(name=n3.name+"-"+n4.name, nodes=[n3, n4], rate=self.rate, delay=self.delay, buffer=self.buffer)
        self.links[n3].append((n4, l3))
        self.links[n4].append((n3, l3))
        l4 = Link(name=n4.name+"-"+n5.name, nodes=[n4, n5], rate=self.rate, delay=self.delay, buffer=self.buffer)
        self.links[n4].append((n5, l4))
        self.links[n5].append((n4, l4))
        l5 = Link(name=n4.name+"-"+n6.name, nodes=[n4, n6], rate=self.rate, delay=self.delay, buffer=self.buffer)
        self.links[n4].append((n6, l5))
        self.links[n6].append((n4, l5))
        l6 = Link(name=n7.name+"-"+n3.name, nodes=[n7, n3], rate=self.rate, delay=self.delay, buffer=self.buffer)
        self.links[n3].append((n7, l6))
        self.links[n7].append((n3, l6))
        # l7 = Link(name=n7.name+"-"+n8.name, nodes=[n7, n8], rate=self.rate, delay=self.delay, buffer=self.buffer)
        # self.links[n7].append((n8, l7))
        # self.links[n8].append((n7, l7))
        # l8 = Link(name=n8.name+"-"+n6.name, nodes=[n8, n6], rate=self.rate, delay=self.delay, buffer=self.buffer)
        # self.links[n8].append((n6, l8))
        # self.links[n6].append((n8, l8))

        self.route()

    def get_requests(self):
        self.s = [self.get_node("n1"), self.get_node("n2")]
        self.d = [self.get_node("n5"), self.get_node("n6")]

        self.get_node("n1").isSender = True
        self.get_node("n1").dest = self.get_node("n5")

        self.get_node("n2").isSender = True
        self.get_node("n2").dest = self.get_node("n6")
        self.get_node("n2").start_time = 10
        self.get_node("n2").end_time = 20


# class SelfTimer(Timer):
#     def __init__(self, network, start_time = 0, end_time=30, step_time=0.01, alloc_time=1, name="t1"):
#         super().__init__(start_time, end_time, step_time, alloc_time, name)
#         self.network: FixTopoNetwork = network
#
#         self.last_write = 0
#         self.l1 = []
#         self.l2 = []
#
#
#     def run(self, simulator):
#         n3m = self.network.get_node("n3").memory
#         n1 = self.network.get_node("n1")
#         n2 = self.network.get_node("n2")
#         n1s = [q for q in n3m if q.src == n1]
#         n2s = [q for q in n3m if q.src == n2]
#
#         self.l1.append(len(n1s))
#         self.l2.append(len(n2s))
#         self.last_write += 1
#         print(n1s,n2s)
#
#
#         if self.last_write % 10 == 0:
#             f.write(f"{simulator.current_time},{np.mean(self.l1)},{np.mean(self.l2)}\n")
#             self.l1 = []
#             self.l2 = []


class SelfTimer(Timer):
    def __init__(self, network, start_time=0, end_time=30, step_time=0.01, alloc_time=1, name="t1"):
        # 正确调用父类构造函数
        super().__init__(name, start_time, end_time, step_time, trigger_func=None)
        self.network: FixTopoNetwork = network
        self.alloc_time = alloc_time
        self.last_write = 0
        self.l1 = []
        self.l2 = []

    def trigger(self):
        """定时器触发时自动调用"""
        n3m = self.network.get_node("n3").memory
        n1 = self.network.get_node("n1")
        n2 = self.network.get_node("n2")
        n1s = [q for q in n3m if q.src == n1]
        n2s = [q for q in n3m if q.src == n2]

        self.l1.append(len(n1s))
        self.l2.append(len(n2s))
        self.last_write += 1
        # print(f"n1的量子比特: {n1s}, n2的量子比特: {n2s}")

        if self.last_write % 10 == 0:
            # 使用安装时设置的 _simulator
            f.write(f"{self._simulator.current_time},{np.mean(self.l1)},{np.mean(self.l2)}\n")
            self.l1 = []
            self.l2 = []

w = 10
m = 10
reroute = True
# for reroute in [False, True]:

random.setstate(randomstate)

s = Simulator(0, 30, 1000)
log.install(s)
net = FixTopoNetwork(n=50, p = 0.1, reqs = 1, memorySize=20, windowSize = w, queryTime= 0.5/m, send_max_try= m, rate = 1000, delay = 0.001, allow_reroute=reroute, random_memory=False)
# net = Network(n=50, p = 0.1, reqs = 1, memorySize=20, windowSize = w, queryTime= 0.1, send_max_try= m, rate = 1000, delay = 0.1, allow_reroute=reroute)
t = SelfTimer(net)

net.install(s)
net.get_node("n2").windowSize = 5
t.install(s)

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
    print(f"{s}, {w},{m},{reroute},{len(s.sendedList)},{len(s.dropList)}, \"{c}\"\n")

# print(reroute, w, m , sum(ans_list) , sep=",")
f.close()
