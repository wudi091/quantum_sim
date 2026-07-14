import uuid

count = 0


def get_qubit_name():
    """生成唯一的量子比特名称：使用全局计数器创建顺序标识符"""
    global count
    count += 1
    return "e" + str(count)


class Qubit(object):
    """量子比特类：表示在量子网络中传输的量子信息单元"""

    def __init__(self, name=None, src=None, dest=None, max_try_count=None, birthday=None):
        """初始化量子比特：设置基本属性和传输参数"""
        if name is None:
            self.name = get_qubit_name()  # 自动生成唯一名称
        else:
            self.name = name

        self.birth = birthday  # 创建时间戳

        # 传输端点
        self.src = src  # 源节点
        self.dest = dest  # 目标节点

        # 当前状态
        self.curr = src  # 当前所在节点
        self.route = [self.src]  # 已经过的路径记录

        # 传输控制
        self.try_count = 0  # 当前尝试次数
        self.max_try_count = max_try_count  # 最大允许尝试次数

    def send(self, n):
        """发送量子比特到新节点：更新位置并重置尝试计数器"""
        self.try_count = 0  # 重置尝试计数
        self.route.append(n)  # 记录路径
        self.curr = n  # 更新当前位置

        # 检查是否到达目的地
        if n == self.dest:
            return True
        return False

    def attempt(self):
        """尝试传输：增加尝试计数并检查是否超过最大限制"""
        self.try_count += 1

        # 检查是否超出最大尝试次数
        if self.try_count > self.max_try_count:
            return False
        return True

    def __repr__(self) -> str:
        """返回量子比特名称的字符串表示"""
        return self.name