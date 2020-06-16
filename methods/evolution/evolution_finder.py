# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

import copy
import random

import numpy as np

from utils.converter import Converter
from utils.latency_predictor import LatencyPredictor
from utils.accuracy_predictor import AccuracyPredictor


class EvolutionFinder():
    def __init__(self, latency_predictor: LatencyPredictor, accuracy_predictor: AccuracyPredictor):
        self.latency_predictor = latency_predictor
        self.accuracy_preditcor = accuracy_predictor
        self.converter = Converter()

    def random_spec(self, constraint):
        while True:
            spec = self.converter.random_spec()
            if not self.converter.is_valid(spec):
                continue
            lat = self.latency_predictor.predict_lat(spec)
            if lat <= constraint:
                return spec, lat

    def mutate_spec(self, spec, constraint):
        while True:
            identity = []
            new_spec = copy.deepcopy(spec)
            block_mutation_prob = 0.1
            father = spec

            for i in range(21):
                depth = i % 4 + 1
                stg = i // 4
                if random.random() < block_mutation_prob:
                    self.converter.change_spec(new_spec, i)

                if depth > father['d'][stg]:
                    identity.append(1)
                else:
                    identity.append(0)
            bad = False
            for i in range(21):
                depth = i % 4 + 1
                stg = i // 4
                if depth == 3 and identity[i]:
                    if not identity[i + 1]:
                        bad = True
                if not identity[i]:
                    new_spec['d'][stg] = max(new_spec['d'][stg], depth)

            if not self.converter.is_valid(new_spec):
                continue
            lat = self.latency_predictor.predict_lat(new_spec)
            if not bad and lat <= constraint:
                return new_spec, lat

    def crossover_spec(self, spec1, spec2, constraint):
        while True:
            new_spec = copy.deepcopy(spec1)
            identity = []
            for i in range(21):
                depth = i % 4 + 1
                stg = i // 4
                father = copy.deepcopy(spec1) if random.random() < 0.5 else copy.deepcopy(spec2)

                new_spec['ks'][i] = father['ks'][i]
                new_spec['e'][i] = father['e'][i]
                for it in range(4):  # quantization policy
                    qname = self.converter.num2qname[it]
                    new_spec[qname][i] = father[qname][i]

                if depth > father['d'][stg]:
                    identity.append(1)
                else:
                    identity.append(0)
            bad = False
            for i in range(21):
                depth = i % 4 + 1
                stg = i // 4
                if depth == 3 and identity[i]:
                    if not identity[i + 1]:
                        bad = True
                if not identity[i]:
                    new_spec['d'][stg] = max(new_spec['d'][stg], depth)
            if not self.converter.is_valid(new_spec):
                continue
            lat = self.latency_predictor.predict_lat(new_spec)
            if not bad and lat <= constraint:
                return new_spec, lat

    def run_evolution_search(self, max_time_budget=1000,
                             population_size=100, mutation_numbers=50, constraint=120):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        times, best_valids, best_tests = [0.0], [-100], [-100]
        population = []  # (validation, spec, latency) tuples
        child_pool = []
        lat_pool = []
        best_info = None
        print('Generate random population...')
        for _ in range(population_size):
            spec, lat = self.random_spec(constraint)
            child_pool.append(spec)
            lat_pool.append(lat)

        accs = self.accuracy_preditcor.predict_accuracy(child_pool)
        for i in range(mutation_numbers):
            population.append((accs[i].item(), child_pool[i], lat_pool[i]))
        print('Start Evolution...')
        iter = 0
        # After the population is seeded, proceed with evolving the population.
        while True:
            parents_size = population_size // 4
            parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
            acc = parents[0][0]
            if iter > 0 and iter % 100 == 1:
                print('Iter: {} Acc: {}'.format(iter - 1, parents[0][0]))

            times.append(iter)
            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_info = parents[0]
            else:
                best_valids.append(best_valids[-1])
            if iter > max_time_budget:
                break
            # sample = random_combination(population, tournament_size)[::-1]
            # best_spec = sorted(sample, key=lambda i: i[0])[-1][1]
            population = parents
            child_pool = []
            lat_pool = []

            for i in range(mutation_numbers):
                par_spec = population[np.random.randint(parents_size)][1]
                # Mutate
                new_spec, lat = self.mutate_spec(par_spec, constraint)
                child_pool.append(new_spec)
                lat_pool.append(lat)

            for i in range(mutation_numbers):
                par_spec1 = population[np.random.randint(parents_size)][1]
                par_spec2 = population[np.random.randint(parents_size)][1]
                # Crossover
                new_spec, lat = self.crossover_spec(par_spec1, par_spec2, constraint)
                child_pool.append(new_spec)
                lat_pool.append(lat)

            accs = self.accuracy_preditcor.predict_accuracy(child_pool)
            for i in range(mutation_numbers):
                population.append((accs[i].item(), child_pool[i], lat_pool[i]))
            iter = iter + 1

        return times, best_valids, best_info
