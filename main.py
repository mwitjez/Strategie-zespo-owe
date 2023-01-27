import numpy as np
import matplotlib.pyplot as plt
from firefly_algorytm import FireflyAlgorithm
from data import PO1, PO2, PO6, PO7, PO8


class KnapsackSolver:
    def __init__(self, capacity, weights, profits, optimal):
        self.capacity = capacity
        self.objects = list(map(lambda x, y: (x, y), profits, weights))
        self.optimal = optimal
        self.solution = None

    def maximize_profit(self, **kwargs):
        fa = FireflyAlgorithm(**kwargs, dimension=len(self.objects))
        self.solution, self.best_solutions = fa.run(self.function)

    def function(self, mask):
        profit_sum = 0
        weight_sum = 0
        for i, object in enumerate(self.objects):
            if weight_sum + object[1] * mask[i] > self.capacity:
                break
            weight_sum += object[1] * mask[i]
            profit_sum += object[0] * mask[i]
        return profit_sum

    def calculate_result(self, mask):
        profit_sum = 0
        weight_sum = 0
        for i, object in enumerate(self.objects):
            if weight_sum + object[1] * mask[i] > self.capacity:
                mask = self._fill_unused_with_zeros(mask, i)
                break
            weight_sum += object[1] * mask[i]
            profit_sum += object[0] * mask[i]
        return profit_sum, weight_sum, mask

    def _fill_unused_with_zeros(self, mask, start_index):
        for i in range(start_index, len(mask)):
            mask[i] = 0
        return mask

    def compare_results(self):
        print("--------------")
        p, w, m = self.calculate_result(self.optimal)
        print(f"OPTIMAL: {m} PROFIT: {p} WEIGHT: {w}")
        p, w, m = self.calculate_result(self.solution)
        print(f"FOUND: {m} PROFIT: {p} WEIGHT: {w}")

    def plot_result(self):
        best_profits = [self.calculate_result(mask)[0] for mask in self.best_solutions]
        best_weights = [self.calculate_result(mask)[1] for mask in self.best_solutions]
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(best_profits)
        ax1.set_xlabel("Epoka")
        ax1.set_ylabel("Zysk najlepszego")

        ax2.plot(best_weights)
        ax2.set_xlabel("Epoka")
        ax2.set_ylabel("Waga najlepszego")
        plt.show()


def test_solver():
    ks_1 = KnapsackSolver(**PO1)
    ks_1.maximize_profit(max_generation=10)
    ks_1.compare_results()
    ks_1.plot_result()

    ks_2 = KnapsackSolver(**PO2)
    ks_2.maximize_profit(max_generation=10)
    ks_2.compare_results()
    ks_2.plot_result()

    ks_3 = KnapsackSolver(**PO6)
    ks_3.maximize_profit(max_generation=10)
    ks_3.compare_results()
    ks_3.plot_result()

    ks_4 = KnapsackSolver(**PO7)
    ks_4.maximize_profit(population_size=20, beta_max=2000)
    ks_4.compare_results()
    ks_4.plot_result()

    ks_5 = KnapsackSolver(**PO8)
    ks_5.maximize_profit(population_size=150, beta_max=15000000)
    ks_5.compare_results()
    ks_5.plot_result()


if __name__ == "__main__":
    test_solver()
