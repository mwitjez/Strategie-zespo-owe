import numpy as np


class FireflyAlgorithm:
    def __init__(
        self,
        population_size=100,
        max_generation=100,
        absorption_coefficient=0.01,
        beta_max=500,
        dimension=None,
    ):
        self.population_size = population_size
        self.max_generation = max_generation
        self.absorption_coefficient = absorption_coefficient
        self.beta_max = beta_max
        self.dimension = dimension
        self.fireflies = self._generate_population()
        self.best_firefly = None

    def run(self, function):
        generation = 0
        light_intensities = self._generate_light_intensity(function)
        self.best_firefly = np.argmax(light_intensities)

        while generation <= self.max_generation:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if light_intensities[j] > light_intensities[i]:
                        distance = self._calculate_distance(i, j)
                        beta = self._calculate_attractiveness(distance)
                        self._update_firefly(i, j, beta)
                        light_intensities[i] = self._update_intensity(i, function)

            self.best_firefly = np.argmax(light_intensities)
            # self._print_res(light_intensities, function)
            generation += 1
        return np.rint(self.fireflies[self.best_firefly])

    def _generate_population(self):
        return np.random.randint(2, size=(self.population_size, self.dimension)).astype(
            "float64"
        )

    def _generate_light_intensity(self, function):
        intensities = np.apply_along_axis(function, 1, self.fireflies)
        return intensities

    def _calculate_distance(self, i, j):
        return np.sum(np.square(self.fireflies[i] - self.fireflies[j]), axis=-1)

    def _calculate_attractiveness(self, distance):
        return self.beta_max * np.exp(-self.absorption_coefficient * distance)

    def _update_firefly(self, i, j, beta):
        random_step = self._generate_random_step()
        self.fireflies[i] += (
            beta * (self.fireflies[j] - self.fireflies[i]) + random_step
        )
        self.fireflies[i] = np.clip(self.fireflies[i], 0, 1)

    def _update_intensity(self, i, function):
        return function(np.rint(self.fireflies[i]))

    def _move_best_firefly(self):
        random_step = self._generate_random_step()
        self.fireflies[self.best_firefly] += random_step
        self.fireflies[self.best_firefly] = np.clip(
            self.fireflies[self.best_firefly], 0, 1
        )

    def _generate_random_step(self):
        return np.random.uniform(low=-1, high=1, size=self.dimension)

    def _print_res(self, light_intensities, function):
        print("---------")
        print(f"BEST: {np.rint(self.fireflies[self.best_firefly])}")
        print((f"TRUE SCORE: {function(np.rint(self.fireflies[self.best_firefly]))}"))
        print(f"SCORE: {light_intensities[self.best_firefly]}")
