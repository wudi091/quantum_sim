from qns.entity.entity import Entity
from qns.simulator.simulator import Simulator
from entity import QNNode, Link
import random

INF = 9999999  # 表示无穷大的常量，用于路由计算


class Network(Entity):
    """量子网络类：管理网络拓扑、路由表和通信请求"""

    def __init__(self, n=10, p=0.5, reqs=3, memorySize=30, windowSize=100, queryTime=0.02, send_max_try=10,
                 allow_reroute=False, rate=3, delay=0.2, buffer=1, random_memory=False):
        # 网络基本属性
        self.nodes = []  # 节点列表
        self.links = {}  # 邻接表表示的链路关系
        self.route_table = {}  # 路由表

        # 拓扑参数
        self.n = n  # 节点数量
        self.p = p  # 链路生成概率
        self.reqs = reqs  # 通信请求数量

        # 通信请求
        self.s = []  # 源节点列表
        self.d = []  # 目标节点列表

        # 节点配置参数
        self.memorySize = memorySize
        self.random_memory = random_memory
        self.windowSize = windowSize
        self.queryTime = queryTime
        self.send_max_try = send_max_try
        self.allow_reroute = allow_reroute

        # 链路配置参数
        self.rate = rate
        self.buffer = buffer
        self.delay = delay

    def install(self, simulator: Simulator):
        """安装网络到模拟器：构建拓扑、生成请求并初始化所有组件"""
        self.build()  # 构建网络拓扑
        self.get_requests()  # 生成通信请求

        # 安装所有链路
        for rl in self.links.values():
            for _, l in rl:
                l.install(simulator)

        # 安装所有节点
        for n in self.nodes:
            n.install(simulator)

    def build(self):
        """构建网络拓扑：创建节点、随机生成链路并确保网络连通性"""
        self.nodes = []
        self.links = {}
        self.route_table = {}

        # 创建所有节点
        for i in range(self.n):
            n: QNNode = QNNode("n" + str(i + 1), memorySize=self.memorySize, windowSize=self.windowSize,
                               queryTime=self.queryTime, send_max_try=self.send_max_try,
                               allow_reroute=self.allow_reroute, random_memory=self.random_memory)
            n.set_net(self)  # 设置节点所属网络
            self.nodes.append(n)
            self.links[n] = []  # 初始化节点的邻接表

        # 随机生成链路
        for i1 in range(self.n):
            for i2 in range(i1 + 1, self.n):
                n1 = self.nodes[i1]
                n2 = self.nodes[i2]

                if random.random() < self.p:  # 以概率p创建链路
                    l = Link(name=n1.name + "-" + n2.name,
                             nodes=[n1, n2], rate=self.rate, delay=self.delay, buffer=self.buffer)
                    self.links[n1].append((n2, l))
                    self.links[n2].append((n1, l))

        # 确保网络连通性：添加必要链路直到所有节点可达
        while True:
            self.route()  # 计算路由表
            tmplist = []
            flag = False

            # 检查是否存在不可达的节点对
            for n1, vl in self.route_table.items():
                for n2, metric in vl.items():
                    if metric[0] == INF:  # 如果距离为无穷大，说明不可达
                        tmplist.append((n1, n2))
                        flag = True

            if flag:
                # 随机选择一个不可达的节点对添加链路
                idx = random.randint(0, len(tmplist) - 1)
                n1, n2 = tmplist[idx]
                l = Link(name=n1.name + "-" + n2.name,
                         nodes=[n1, n2], rate=self.rate, delay=self.delay, buffer=self.buffer)
                self.links[n1].append((n2, l))
                self.links[n2].append((n1, l))
            else:
                break  # 所有节点都连通，退出循环

    def route(self):
        """计算路由表：使用Dijkstra算法计算所有节点对之间的最短路径"""
        for n in self.nodes:
            selected = []  # 已选择节点集合
            unselected = self.nodes.copy()  # 未选择节点集合
            d = {}  # 距离字典：节点 -> [最短距离, 路径]

            # 初始化距离
            for nn in self.nodes:
                if nn == n:
                    d[nn] = [0, []]  # 到自身的距离为0
                else:
                    d[nn] = [INF, [nn]]  # 到其他节点的初始距离为无穷大

            # Dijkstra算法主循环
            while len(unselected) != 0:
                # 选择当前距离最小的节点
                ms = unselected[0]
                mi = d[ms][0]
                for s in unselected:
                    if d[s][0] < mi:
                        ms = s
                        mi = d[s][0]

                # 将当前节点标记为已处理
                selected.append(ms)
                unselected.remove(ms)

                # 更新邻居节点的距离
                for (s, l) in self.links[ms]:
                    if s in unselected and d[s][0] > d[ms][0] + l.metric:
                        # 找到更短路径，更新距离和路径
                        d[s] = [d[ms][0] + l.metric, [ms] + d[ms][1]]

            # 完善路径信息（添加目标节点）
            for nn in self.nodes:
                d[nn][1] = [nn] + d[nn][1]

            self.route_table[n] = d  # 存储该节点的路由信息

    def print_route_table(self):
        """打印路由表：以矩阵形式显示所有节点对之间的距离"""
        for i1 in range(self.n):
            for i2 in range(self.n):
                print(self.route_table[self.nodes[i1]][self.nodes[i2]][0], end="\t")
            print()

    def query_route(self, src, dest):
        """查询路由：获取从源节点到目标节点的所有可能下一跳及其度量"""
        assert (src in self.nodes and dest in self.nodes)

        ret = []

        # 遍历源节点的所有邻居
        for neighbour, l in self.links[src]:
            # 计算通过该邻居到目标节点的总度量
            total_metric = l.metric + self.route_table[dest][neighbour][0]
            ret.append((neighbour, l, total_metric))

        # 按总度量排序返回
        return sorted(ret, key=lambda x: x[2])

    def get_requests(self):
        """生成通信请求：随机选择源-目标节点对并配置发送节点"""
        self.s = []
        self.d = []

        # 重置所有节点的发送状态
        for n in self.nodes:
            n.isSender = False

        # 生成指定数量的通信请求
        for _ in range(self.reqs):
            # 选择源节点（确保不重复）
            s = None
            while s is None or s in self.s:
                si = random.randint(0, self.n - 1)
                s = self.nodes[si]
            self.s.append(s)

            # 选择目标节点（确保不是源节点自身）
            d = None
            while d is None or d == s:
                di = random.randint(0, self.n - 1)
                d = self.nodes[di]
            self.d.append(d)

        # 配置发送节点
        for i in range(self.reqs):
            s = self.s[i]
            d = self.d[i]
            s.isSender = True
            s.dest = d

    def get_node(self, name):
        """根据节点名称查找节点对象"""
        for n in self.nodes:
            if n.name == name:
                return n
        return None