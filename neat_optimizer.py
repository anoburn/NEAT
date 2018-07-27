import numpy as np
import population
import genome as genome_mod
import logging
import sys
import os
from copy import deepcopy

logger = logging.getLogger('Logger.NEAT')

class NEAT_optimizer:
    def __init__(self):
        self.population = None
        self.parental_population = None

    def initialize(self, n_in, n_out, pop_size=100, folder=None, delta_t=1., c1=1., c2=1., c3=0.5, desired_species=10,
                   min_species=2, p_weight_mut=0.7, p_weight_random=0.02, weight_mut_sigma=0.3, node_mut_rate=0.01,
                   edge_mut_rate=0.05, p_child_clone=0.25, p_mutate=0.8, p_inter_species=0.02, weight_amplitude=1.):
        self.generation = 1
        self.n_in = n_in
        self.n_out = n_out
        prototype = genome_mod.Genome(self.n_in, self.n_out)
        self.population = population.Population([deepcopy(prototype) for _ in range(pop_size)],
                                                delta_t=delta_t, c1=c1, c2=c2, c3=c3,
                                                desired_species=desired_species, min_species=min_species,
                                                p_weight_mut=p_weight_mut, p_weight_random=p_weight_random,
                                                weight_mut_sigma=weight_mut_sigma,
                                                node_mut_rate=node_mut_rate, edge_mut_rate=edge_mut_rate,
                                                p_child_clone=p_child_clone, p_mutate=p_mutate, p_inter_species=p_inter_species,
                                                weight_amplitude=weight_amplitude)
        self.population = self.population.generate_offspring()
        #self.population = population.Population([genome_mod.Genome(self.n_in, self.n_out) for _ in range(pop_size)])
        if folder is not None:
            if not os.path.exists(folder):
                os.makedirs(folder)
        else:
            folder = ""
        self.folder = folder + "/"
        self.best_genome = None

    def generate_offspring(self, save_old=False):
        """ Go fuck yourself """
        logger.debug('Generating generation {}'.format(self.generation + 1))
        if self.parental_population is not None and save_old:
            self.parental_population.phenotypes = None
            np.save(self.folder + 'generation{}.npy'.format(self.generation-1), self.parental_population)

        self.parental_population = self.population
        #logger.info("Population size before: {}".format(len(self.population.genomes)))
        self.population = self.population.generate_offspring()
        #logger.info("Population size after:  {}".format(len(self.population.genomes)))
        self.generation += 1

    def get_functions(self):
        self.population.build_phenotypes()
        return self.population.phenotypes

    def set_scores(self, scores):
        """ scores is a list of numerical scores """
        for i, score in enumerate(scores):
            self.population.genomes[i].score = score

    def get_best(self, generation=None):
        self.best_genome = self.population.get_best()
        return self.best_genome

    def draw_best(self):
        self.get_best()
        genome_mod.draw_genome_net(self.best_genome, show_disabled=False, show_innov=True, filename=self.folder + "best_gen{}".format(self.generation))
        #logging.info("Generation {}: Best score {}".format(self.generation, self.best_genome.score))

    def print_gen_info(self):
        logger.info("Generation {}: {} species".format(self.generation, len(self.population.all_species)))
        logger.info("    Best score: {}".format(self.get_best().score))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    genotype = genome_mod.Genome(3, 2)
    genotype2 = genome_mod.Genome(3, 2, genotype.connections)
    genotype.add_random_node()
    genotype2.add_random_node()
    for i in range(2):
        #genotype.add_random_node()
        genotype2.add_random_node()
        #genotype2.random_connection()
        #print_edges(genotype2)
        #draw_genome_net(genotype2, show_innov=True, show_disabled=True, filename="genome2_run%i"%i)

    #genome_mod.draw_genome_net(genotype, show_weights=True, show_disabled=True, show_innov=False, filename="genome1")
    #genome_mod.draw_genome_net(genotype2,show_weights=True, show_disabled=True, show_innov=False, filename="genome2")

    #funktion = genotype2.build_phenotype()
    #print(funktion([1,1,1]))
    pop = population.Population([genotype, genotype2])
    print(len(pop.genomes))
    #print(pop.all_species.values())
    pop = pop.generate_offspring()
    print(len(pop.genomes))
    #child = pop.create_child(genotype, genotype2)

    #genome_mod.print_edges(genotype)
    #genome_mod.print_edges(genotype2)
    #genome_mod.print_edges(child)
    #genome_mod.draw_genome_net(child, show_weights=True, show_disabled=False, show_innov=False, filename="child before")
    #pop.mutate(child)
    #draw_genome_net(child, show_innov=True, show_disabled=True, filename="child after")

    #f = genotype.build_phenotype()
    #x = np.array([1,2,3])
    #print(f(x))

