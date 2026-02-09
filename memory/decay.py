# -*- coding: utf-8 -*-
"""
时间衰减管理器
- 周期性对全局共现矩阵施加指数衰减
- 近期事件权重高，远期事件自然淡化
- 符合认知科学中的艾宾浩斯遗忘曲线特性
"""


class DecayManager:
    """
    时间衰减管理器。

    原理：每隔 decay_interval 条文档，对全局矩阵乘以衰减系数 (1 - decay_rate)。
    这样近期的共现关系权重最高，远期的逐步衰减。

    参数:
        decay_rate: float - 每次衰减的比率，默认 0.005（即每次保留 99.5%）
        decay_interval: int - 每多少条文档执行一次衰减
    """

    def __init__(self, decay_rate=0.005, decay_interval=500):
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval
        self.last_decay_doc = 0
        self.total_decay_steps = 0

    def maybe_decay(self, cooccurrence_obj):
        """
        检查是否需要执行衰减，如果到达间隔则执行。

        参数:
            cooccurrence_obj: IncrementalCooccurrence 实例
        返回:
            bool - 是否执行了衰减
        """
        current_doc = cooccurrence_obj.total_docs

        if current_doc - self.last_decay_doc >= self.decay_interval:
            steps = (current_doc - self.last_decay_doc) // self.decay_interval
            factor = (1 - self.decay_rate) ** steps

            # 直接对 DOK 矩阵的数据进行衰减
            # DOK 矩阵支持逐元素操作
            csr = cooccurrence_obj.matrix.tocsr()
            csr.data *= factor
            # 衰减后，非常小的值去掉，避免矩阵膨胀
            csr.data[csr.data < 0.5] = 0
            csr.eliminate_zeros()
            cooccurrence_obj.matrix = csr.todok()

            self.last_decay_doc = current_doc
            self.total_decay_steps += steps
            return True

        return False

    def get_info(self):
        """返回衰减管理器的状态信息"""
        return {
            "decay_rate": self.decay_rate,
            "decay_interval": self.decay_interval,
            "last_decay_doc": self.last_decay_doc,
            "total_decay_steps": self.total_decay_steps,
            "effective_retention": (1 - self.decay_rate) ** self.total_decay_steps
                if self.total_decay_steps > 0 else 1.0
        }
