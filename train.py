# coding=utf-8
import numpy as np
from random import shuffle
import copy


# b_{i j}=V_j+\lambda_j * v_{i j}-\alpha_j-\pi_j ({cost}_i-C_j)

class Model(object):
    def __init__(self, model_name, supply, demand, lamb=10, mu=1.0):
        ##########
        ## init ##
        ##########
        self.model_name, self.supply, self.demand, self.lamb, self.mu = model_name, copy.deepcopy(
            supply), copy.deepcopy(demand), lamb, mu
        self.i, self.j = self.supply.supply_size, self.supply.demand_size
        # init dual vars
        self.dual_alpha_list = np.ones(shape=(self.j,), dtype=float) * 0.05
        self.dual_pi_list = np.ones(shape=(self.j,), dtype=float) * self.mu
        if self.model_name not in ("Rand", "PBD", "FACC"):
            print("Wrong model name:" + self.model_name)

        self.ctr_for_compute = self.supply.ctr_mat
        self.cost_for_compute = self.supply.cost_mat
        self.target_cost_for_compute = self.demand.demand_cost_target
        self.update_pi = False

        if self.model_name == "Rand":
            self.ctr_for_compute = self.supply.ctr_fake_mat
            self.cost_for_compute = self.cost_for_compute * 0.0
        elif self.model_name == "PBD":
            self.cost_for_compute = self.cost_for_compute * 0.0
        elif self.model_name == "FACC":
            self.update_pi = True

    def delivery(self, budget_ratio=1.0):
        alloc = np.zeros((self.i, self.j), dtype=np.float64)
        demand_allocate = np.zeros(self.j, dtype=np.float64)
        bid = (1.0 + self.lamb * self.ctr_for_compute - self.dual_alpha_list - self.dual_pi_list *
               (self.cost_for_compute - self.target_cost_for_compute)) * self.supply.connect_mat
        #        shuffle
        req_idx_list = list(range(self.i))
        shuffle(req_idx_list)

        over_allo_flag = demand_allocate < self.demand.demand_amount_mat * budget_ratio
        for i in req_idx_list:
            prob_list = bid[i]
            prob_list *= over_allo_flag
            j = np.argmax(prob_list, axis=0, keepdims=False)
            if prob_list[j] < 0.0000000001:  # all filtered
                continue
            alloc[i][j] = 1.0
            demand_allocate[j] += 1
            over_allo_flag = demand_allocate < self.demand.demand_amount_mat * budget_ratio

        delivery_pv = np.sum(alloc)
        dual_alpha_avg = round(np.mean(self.dual_alpha_list), 4)
        dual_pi_avg = round(np.mean(self.dual_pi_list), 4)
        ctr_avg = round(np.sum(self.supply.ctr_mat * alloc) / delivery_pv, 4)
        cost_avg = round(np.sum(self.supply.cost_mat * alloc) / delivery_pv, 4)

        cost_demand_avg = round(
            np.mean(np.sum(self.supply.cost_mat * alloc, axis=0) / (demand_allocate + 0.0000000001)), 4)
        # allocate_residual: over-alloc:negative; under-alloc:postive
        allocate_residual = self.demand.demand_amount_mat - demand_allocate
        over_allocate = -int(np.sum(allocate_residual.clip(max=0)))
        delivery_pv_valid = delivery_pv - over_allocate
        finish_rate = round(delivery_pv_valid / np.sum(self.demand.demand_amount_mat), 4)
        finish_rate_all = round(delivery_pv / np.sum(self.demand.demand_amount_mat), 4)

        eval_list = [finish_rate, ctr_avg, cost_avg, cost_demand_avg]

        print("Model:", self.model_name, ",finish_rate_real:", finish_rate, ",finish_rate_all:", finish_rate_all,
              ",alpha:", dual_alpha_avg,
              ",pi:", dual_pi_avg, ",ctr:", ctr_avg, ",cost:", cost_avg, ',cost_d:', cost_demand_avg)

        return alloc, demand_allocate, eval_list

    def train(self, t=80, lr_init=0.8, budget_ratio=1.5):
        for epoch in range(t):
            lr = lr_init / (epoch / 2.0 + 1.0)
            budget_ratio_t = budget_ratio
            if budget_ratio > 1.1:
                budget_ratio_t = budget_ratio - (budget_ratio - 1.1) / t * epoch
            alloc, demand_allocate, _ = self.delivery(budget_ratio)

            # update alpha
            ogd_alpha = (self.demand.demand_amount_mat - demand_allocate) / self.demand.demand_amount_mat
            print("    lr:", round(lr, 4), ",grad_alpha:", round(np.mean(np.absolute(ogd_alpha)), 4),
                  ",budget_rat:", round(budget_ratio_t, 4))
            ogd_alpha = ogd_alpha.clip(min=-0.2)
            ogd_alpha = ogd_alpha.clip(max=0.2)
            self.dual_alpha_list = self.dual_alpha_list - lr * ogd_alpha
            self.dual_alpha_list = self.dual_alpha_list.clip(min=0.0)
            # update pi
            if self.update_pi:
                ogd_pi = (self.target_cost_for_compute - np.sum(alloc * self.cost_for_compute,
                                                                axis=0)) / self.target_cost_for_compute
                ogd_pi = ogd_pi.clip(min=-0.2)
                ogd_pi = ogd_pi.clip(max=0.2)
                self.dual_pi_list = self.dual_pi_list - lr * ogd_pi
                self.dual_pi_list = self.dual_pi_list.clip(min=0.0)
                self.dual_pi_list = self.dual_pi_list.clip(max=self.mu)

            print()
