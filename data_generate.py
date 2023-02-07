# coding=utf-8
import numpy as np


class Demand:
    def __init__(self, demand_size, supply_size):
        self.demand_size = demand_size
        self.supply_size = supply_size
        self.demand_amount_mat = None  # demand_j
        self.demand_cost_target = None  # C_j

        # init
        self.__rand_init()

    def __rand_init(self):
        demand_low = self.supply_size * 1.0 / self.demand_size * 0.1
        demand_high = self.supply_size * 1.0 / self.demand_size * 1.8
        np.random.seed(0)
        self.demand_amount_mat = np.random.randint(demand_low, demand_high, size=self.demand_size)
        self.demand_cost_target = np.ones(shape=(self.demand_size,)) * 0.5


class Supply:
    def __init__(self, demand_size, supply_size):
        # zero dim
        self.demand_size = demand_size
        self.supply_size = supply_size
        # two dim
        self.ctr_mat = None
        self.ctr_fake_mat = None
        self.connect_mat = None
        self.cost_mat = None
        self.cost_zero_mat =None

        # init
        self.__rand_init()

    def __rand_init(self):
        size = (self.supply_size, self.demand_size)
        np.random.seed(1)
        self.connect_mat = np.random.random(size=size)
        self.connect_mat = (self.connect_mat < 0.05) * 1.0
        self.ctr_mat = np.random.uniform(low=0.02, high=0.08, size=size)
        self.cost_mat = np.random.uniform(low=0.0, high=2.0, size=(self.supply_size,))
        self.cost_mat = np.repeat(self.cost_mat, self.demand_size, axis=0).reshape(size)
        # self.cost_zero_mat = np.zeros(shape=size)

        np.random.seed(2)
        self.ctr_fake_mat = np.random.uniform(low=0.02, high=0.08, size=size)


def load_data(demand_size=100, supply_size=100000):
    demand_data = Demand(demand_size, supply_size)
    supply_data = Supply(demand_size, supply_size)
    return supply_data, demand_data


if __name__ == '__main__':
    demand = Demand(10, 1000)
    supply = Supply(10, 20)
    print(supply.cost_zero_mat)
