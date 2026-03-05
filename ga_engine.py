import random
import numpy as np

class GeneticOptimizer:
    def __init__(self, pop_size=100, mut_rate=0.15, generations=150, elite_percent=0.05, seed=None):
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.generations = generations
        self.elite_percent = elite_percent
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def initialise(self, num_features):
        population = []
        for _ in range(self.pop_size):
            genes = [random.randint(0, 1) for _ in range(num_features)]
            threshold = random.randint(2, max(2, num_features))
            population.append(genes + [threshold])
        return population

    def fitness(self, individual, data_np):
        genes = np.array(individual[:-1])
        threshold = individual[-1]

        if genes.sum() == 0:
            return 0.0

        active_counts = (data_np[:, :-1] * genes).sum(axis=1)
        positions = (active_counts >= threshold).astype(int)

        if positions.sum() == 0:
            return 0.0

        targets = data_np[:, -1]
        daily_pnl = positions[:-1] * (targets[:-1] * 2 - 1)

        if daily_pnl.std() == 0:
            return 0.0

        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)

        # complexity penalty: favour rules using fewer features
        complexity = genes.sum() / len(genes)
        penalty = 0.1 * complexity

        return float(np.clip(sharpe - penalty, -3.0, 3.0))

    def evolve(self, train_data):
        data_np = np.array(train_data)  # convert ONCE here
        num_features = data_np.shape[1] - 1
        population = self.initialise(num_features)

        best_rule = None
        best_fitness = -1

        for _ in range(self.generations):
            fits = [self.fitness(ind, data_np) for ind in population]
            
            # track best across all generations
            gen_best_idx = int(np.argmax(fits))
            if fits[gen_best_idx] > best_fitness:
                best_fitness = fits[gen_best_idx]
                best_rule = population[gen_best_idx]
            
            num_elites = max(1, int(self.pop_size * self.elite_percent))
            elites = [population[i] for i in np.argsort(fits)[-num_elites:]]
            
            new_pop = elites[:]
            while len(new_pop) < self.pop_size:
                p1, p2 = self.select(population, fits), self.select(population, fits)
                child1, child2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(child1, num_features))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(child2, num_features))
            population = new_pop

        return best_rule

    def select(self, pop, fits):
        idx = random.sample(range(len(pop)), 3)
        return pop[max(idx, key=lambda i: fits[i])]

    def crossover(self, p1, p2):
        mask = [random.random() < 0.5 for _ in range(len(p1))]
        c1 = [p1[i] if mask[i] else p2[i] for i in range(len(p1))]
        c2 = [p2[i] if mask[i] else p1[i] for i in range(len(p1))]
        return c1, c2

    def mutate(self, ind, num_features):
        new_ind = [1-g if random.random() < self.mut_rate else g for g in ind[:-1]]
        thresh = ind[-1]
        if random.random() < self.mut_rate:
            thresh = max(2, min(num_features, thresh + random.choice([-1, 1])))
        return new_ind + [thresh]