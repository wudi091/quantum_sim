from functools import singledispatch
from typing import List, Optional, Tuple, Any
from qns.simulator.simulator import Simulator
from qns.simulator.event import Event
from qns.entity.node.node import QNode
from qns.entity.qchannel.qchannel import QuantumChannel
from qns.simulator.ts import Time
from qubit import Qubit
import qns.utils.log as log
import random

# 处理步骤时间间隔
handle_step_time = 0.01000


class QNodeHandleEvent(Event):
    """处理节点事件：用于触发节点处理逻辑的通用事件"""

    def __init__(self, refresh: bool, t: Optional[Time] = None, name: Optional[str] = None, by: Optional[Any] = None):
        super().__init__(t=t, name=name, by=by)
        self.refresh = refresh  # 是否刷新发送窗口

    def invoke(self) -> None:
        """事件触发时调用节点的handle方法"""
        if self.by and hasattr(self.by, 'handle'):
            self.by.handle(self.by.simulator, None, self.by, self)


class QNodeQueryBeforeEvent(Event):
    """查询前事件：在发送量子比特前进行路由查询和资源检查"""

    def __init__(self, qubit: Qubit, t: Optional[Time] = None, name: Optional[str] = None, by: Optional[Any] = None):
        super().__init__(t=t, name=name, by=by)
        self.qubit = qubit  # 待发送的量子比特

    def invoke(self) -> None:
        """触发节点的before_send_attempt处理逻辑"""
        if self.by and hasattr(self.by, 'handle'):
            self.by.handle(self.by.simulator, self.qubit, None, self)


class QNodeQueryAfterEvent(Event):
    """查询后事件：在量子比特成功发送到下一跳后处理"""

    def __init__(self, qubit: Qubit, currhop, nexthop, t: Optional[Time] = None, name: Optional[str] = None,
                 by: Optional[Any] = None):
        super().__init__(t=t, name=name, by=by)
        self.qubit = qubit  # 传输中的量子比特
        self.currhop = currhop  # 当前节点
        self.nexthop = nexthop  # 下一跳节点

    def invoke(self) -> None:
        """触发节点的after_send_attempt处理逻辑"""
        if self.by and hasattr(self.by, 'handle'):
            self.by.handle(self.by.simulator, self.qubit, None, self)


class QNNode(QNode):
    """量子网络节点：扩展的基础量子节点，支持量子比特的路由和传输管理"""

    def __init__(self, name: str, isSender=False, dest=None, memorySize=10, windowSize=10, queryTime=0.02,
                 start_time: float = 0, end_time: float = None, send_max_try=100, allow_reroute=False,
                 random_memory=False):
        self.name = name

        # 发送器配置
        self.isSender = isSender  # 是否为发送节点
        self.dest = dest  # 目标节点
        self.send_max_try = send_max_try  # 最大尝试次数
        self.queryTime = queryTime  # 查询时间间隔
        self.start_time = start_time  # 开始时间
        self.end_time = end_time  # 结束时间

        # 内存管理
        self.memorySize = memorySize  # 内存容量
        self.currentSize = 0  # 当前使用量
        self.memory = []  # 存储的量子比特列表
        self.random_memory = random_memory  # 是否使用随机内存大小

        # 窗口管理
        self.windowSize = windowSize  # 发送窗口大小
        self.minhop = 0  # 最小跳数
        self.currentWindowsSize = 0  # 当前窗口大小
        self.sendingList = []  # 发送中的量子比特列表
        self.sendedList = []  # 已发送的量子比特列表
        self.dropList = []  # 丢弃的量子比特列表
        self.allow_reroute = allow_reroute  # 是否允许重路由

        # 查询统计
        self.query_ans = []  # 查询结果历史
        self.query_ans_max_len = 10  # 历史记录最大长度
        self.query_delta = 0.001  # 查询时间随机延迟

        # 网络状态
        self.query_list = {}  # 各节点的查询记录
        self.net = None  # 所属网络

        if self.random_memory:
            self.memorySize = random.randint(1, self.memorySize)

    def set_net(self, net):
        """设置节点所属的网络拓扑"""
        self.net = net

    def install(self, simulator: Simulator):
        """安装节点到模拟器：初始化时间参数并启动第一个处理事件"""
        self.simulator = simulator
        self.query_time = simulator.time(sec=self.queryTime)
        self.start_time_obj = simulator.time(sec=self.start_time)
        if self.end_time is not None:
            self.end_time_obj = simulator.time(sec=self.end_time)
        else:
            self.end_time_obj = simulator.te

        self.step_time = simulator.time(sec=handle_step_time)

        # 初始化所有节点的查询记录
        for n in self.net.nodes:
            self.query_list[n] = []

        # 添加启动事件
        event = QNodeHandleEvent(refresh=True, t=self.start_time_obj, name=f"StartEvent_{self.name}", by=self)
        simulator.add_event(event)

    def handle(self, simulator: Simulator, msg: object, source=None, event: Event = None):
        """事件处理分发器：根据事件类型调用相应的处理逻辑"""
        if not self.isSender:
            return
        if isinstance(event, QNodeHandleEvent):
            if self.simulator.current_time > self.end_time_obj:
                return
            refresh = event.refresh
            self.send(refresh)
        elif isinstance(event, QNodeQueryAfterEvent):
            qubit = event.qubit
            nexthop = event.nexthop
            currhop = event.currhop
            self.after_send_attempt(qubit, currhop, nexthop)
        elif isinstance(event, QNodeQueryBeforeEvent):
            qubit = event.qubit
            self.before_send_attempt(qubit)

    def send(self, refresh=False):
        """发送量子比特：根据窗口大小创建并发送新的量子比特"""
        if self.minhop == 0:
            self.minhop = self.net.route_table[self][self.dest][0]

        # 如果发送列表未满，创建新的量子比特
        if len(self.sendingList) < self.windowSize * self.minhop:
            q = Qubit(src=self, dest=self.dest,
                      max_try_count=self.send_max_try, birthday=self.simulator.current_time)
            self.sendingList.append(q)
            log.debug(f"qubit {q} start to transmit from {self} to {self.dest}")

            # 添加随机延迟后触发查询前事件
            rt_delay = random.random() * self.query_delta
            rt = self.simulator.time(sec=rt_delay)
            event = QNodeQueryBeforeEvent(
                qubit=q,
                t=self.simulator.current_time + rt,
                name=f"QueryBefore_{q.name}",
                by=self
            )
            self.simulator.add_event(event)

        # 如果还有空间，安排下一次发送
        if len(self.sendingList) < self.windowSize * self.minhop:
            event = QNodeHandleEvent(
                refresh=True,
                t=self.simulator.current_time + self.step_time,
                name=f"NextSend_{self.name}",
                by=self
            )
            self.simulator.add_event(event)

    def before_send_attempt(self, qubit):
        """发送前尝试：检查路由和资源可用性，决定是否发送"""
        retq = qubit.attempt()  # 量子比特自身状态检查

        # 获取路由信息
        currhop, nexthop, nextlink = self.route(qubit)

        # 检查下一跳节点和链路的可用性
        if nexthop is None or nextlink is None:
            retn = False
            retl = False
            retq = False
        else:
            retn = nexthop.query(self.simulator)  # 检查节点内存
            retl, _ = nextlink.query(self.simulator)  # 检查链路带宽

        # 如果所有条件满足，发送量子比特
        if retq and retn and retl:
            self.update2(nexthop, True)  # 更新节点查询统计
            nexthop.use(qubit)  # 占用下一跳节点内存
            retll, st = nextlink.use(self.simulator)  # 占用链路资源
            assert (retll == True)
            log.debug(f"qubit {qubit.name} send from {currhop} to {nexthop} at {st + self.query_time}")

            # 安排查询后事件
            event = QNodeQueryAfterEvent(
                qubit=qubit,
                currhop=currhop,
                nexthop=nexthop,
                t=st + self.query_time,
                name=f"QueryAfter_{qubit.name}",
                by=self
            )
            self.simulator.add_event(event)
        else:
            # 发送失败处理
            if nexthop is not None:
                self.update2(nexthop, False)  # 更新失败统计

            if retq == False:
                # 量子比特超过最大尝试次数，丢弃
                self.sendingList.remove(qubit)
                self.dropList.append(qubit)
                if self != currhop:
                    currhop.release(qubit)
                log.debug(f"qubit {qubit.name} drop on {currhop} nexthop {nexthop} {retn} nextlink {nextlink} {retl}")

                # 触发处理事件
                event = QNodeHandleEvent(
                    refresh=False,
                    t=self.simulator.current_time + self.query_time,
                    name=f"HandleAfterDrop_{qubit.name}",
                    by=self
                )
                self.simulator.add_event(event)
            else:
                # 资源不足，稍后重试
                log.debug(f"qubit {qubit.name} retry on {currhop} nexthop {nexthop} {retn} nextlink {nextlink} {retl}")
                rt_delay = random.random() * self.query_delta
                rt = self.simulator.time(sec=rt_delay)
                event = QNodeQueryBeforeEvent(
                    qubit=qubit,
                    t=self.simulator.current_time + self.query_time + rt,
                    name=f"RetryQuery_{qubit.name}",
                    by=self
                )
                self.simulator.add_event(event)

    def route(self, qubit):
        """路由选择：根据路由策略选择下一跳节点和链路"""
        currhop = qubit.curr
        rt = self.net.query_route(currhop, self.dest)

        # 如果不允许重路由，使用最短路径
        if not self.allow_reroute:
            nexthop: QNode = rt[0][0]
            nextlink: Link = rt[0][1]
            return currhop, nexthop, nextlink

        # 重路由逻辑：基于概率和路径质量选择最优路径
        m = qubit.try_count
        M = qubit.max_try_count
        if m > M:
            return currhop, None, None
        metric_drop = self.net.route_table[self.dest][self][0] * 2

        nexthop: QNode = rt[0][0]
        nextlink: Link = rt[0][1]
        INF = 999999

        min_mt = INF
        min_y = INF

        # 评估所有可能的邻居节点
        for neigh in rt:
            np = neigh[0]
            nl = neigh[1]
            mt = neigh[2]

            Lmax = max(self.net.route_table[self.dest][self][0], 5)
            pce = 1 - self.net.route_table[self.dest][self][0] / Lmax

            # 概率性路径探索
            if random.random() < pce:
                lmt = min_mt + 1
            else:
                lmt = min_mt

            if mt > lmt:
                continue
            if len(qubit.route) + mt > Lmax:
                continue

            # 计算路径评估指标
            p = self.stat2(np)
            y = (1 - (1 - p) ** (M - m)) * mt + (1 - p) ** (M - m) * metric_drop
            if y < min_y:
                nexthop = np
                nextlink = nl
                min_mt = mt
                min_y = y

        delta = 1
        if min_y > metric_drop:
            return currhop, None, None

        if nexthop in qubit.route:
            return currhop, None, None

        return currhop, nexthop, nextlink

    def after_send_attempt(self, qubit: Qubit, currhop, nexthop):
        """发送后处理：量子比特成功到达下一跳后的处理逻辑"""
        log.debug(f"qubit {qubit.name} ({qubit.src}->{qubit.dest}) recved from {currhop} to {nexthop}")

        # 释放当前节点的资源
        if self != currhop:
            currhop.release(qubit)

        qubit.send(nexthop)  # 更新量子比特位置

        # 如果到达目的地，完成传输
        if self.dest == nexthop:
            self.sendingList.remove(qubit)
            self.sendedList.append(qubit)
            nexthop.release(qubit)
            log.debug(f"qubit {qubit.name} ({qubit.src}->{qubit.dest}) arrived")

            event = QNodeHandleEvent(
                refresh=False,
                t=self.simulator.current_time,
                name=f"HandleAfterArrival_{qubit.name}",
                by=self
            )
            self.simulator.add_event(event)
        else:
            # 继续传输到下一跳
            rt_delay = random.random() * self.query_delta
            rt = self.simulator.time(sec=rt_delay)
            event = QNodeQueryBeforeEvent(
                qubit=qubit,
                t=self.simulator.current_time + rt,
                name=f"QueryBeforeNextHop_{qubit.name}",
                by=self
            )
            self.simulator.add_event(event)

    def query(self, simulator: Simulator) -> Tuple[bool, int]:
        """查询节点内存状态：检查是否有可用内存空间"""
        ret = self.currentSize < self.memorySize
        return ret

    def use(self, qubit):
        """占用节点内存：存储量子比特到内存中"""
        self.currentSize += 1
        self.memory.append(qubit)

    def release(self, qubit):
        """释放节点内存：从内存中移除量子比特"""
        self.currentSize -= 1
        self.memory.remove(qubit)

    def stat(self):
        """统计历史查询成功率"""
        delta = 0.5
        nt = 0
        na = len(self.query_ans)
        for ans in self.query_ans:
            if ans:
                nt += 1
        return (nt + delta) / (na + delta)

    def update(self, ret):
        """更新查询结果历史"""
        self.query_ans.append(ret)
        if len(self.query_ans) > self.query_ans_max_len:
            del self.query_ans[0]

    def stat2(self, node):
        """统计特定节点的查询成功率"""
        delta = 0.5
        nt = 0
        na = len(self.query_list[node])
        for ans in self.query_list[node]:
            if ans:
                nt += 1
        return (nt + delta) / (na + delta)

    def update2(self, node, result):
        """更新特定节点的查询记录"""
        self.query_list[node].append(result)
        if len(self.query_list[node]) > self.query_ans_max_len:
            del self.query_list[node][0]

    def __repr__(self):
        """返回节点名称的字符串表示"""
        return self.name


class Link(QuantumChannel):
    """量子链路：管理两个量子节点之间的连接和带宽资源"""

    def __init__(self, name: str, nodes: List[QNode], metric=1, rate=1, delay=0.2, buffer=None):
        self.name = name  # 链路名称
        self.nodes = nodes  # 连接的节点
        self.rate = rate  # 传输速率
        self.delay = delay  # 传输延迟
        self.buffer = buffer  # 缓冲区大小
        self.metric = metric  # 链路度量值

        self.current_send_time = None  # 当前发送时间

    def install(self, simulator: Simulator):
        """安装链路到模拟器：初始化时间参数"""
        self.current_send_time = simulator.time(sec=simulator.ts.sec)
        send_interval_sec = 1.0 / self.rate
        self.step_send_time = simulator.time(sec=send_interval_sec)  # 发送间隔
        self.delay_time = simulator.time(sec=self.delay)  # 传输延迟

    def query(self, simulator: Simulator) -> Tuple[bool, Optional[int]]:
        """查询链路可用性：检查链路是否可立即使用"""
        if self.current_send_time < simulator.current_time:
            send_time_slice = self.current_send_time + self.delay_time
            return True, send_time_slice

        if self.buffer is None:
            send_time_slice = self.current_send_time + self.delay_time
            return True, send_time_slice

        # 检查缓冲区限制
        buffer_time_slots = self.step_send_time.time_slot * self.buffer
        buffer_time = simulator.time(time_slot=buffer_time_slots)

        if self.current_send_time > simulator.current_time + buffer_time:
            return False, None
        else:
            send_time_slice = self.current_send_time + self.delay_time
            return True, send_time_slice

    def use(self, simulator: Simulator):
        """占用链路资源：分配传输时间并返回发送时间片"""
        if self.current_send_time < simulator.current_time:
            send_time_slice = simulator.current_time + self.delay_time
            self.current_send_time = simulator.current_time + self.step_send_time
            return True, send_time_slice

        if self.buffer is None:
            send_time_slice = self.current_send_time + self.delay_time
            self.current_send_time += self.step_send_time
            return True, send_time_slice

        # 检查缓冲区限制
        buffer_time_slots = self.step_send_time.time_slot * self.buffer
        buffer_time = simulator.time(time_slot=buffer_time_slots)

        if self.current_send_time > simulator.current_time + buffer_time:
            return False, None
        else:
            send_time_slice = self.current_send_time + self.delay_time
            self.current_send_time += self.step_send_time
            return True, send_time_slice

    def __repr__(self):
        """返回链路名称的字符串表示"""
        return self.name