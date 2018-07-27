import random
import numpy as np
import genome as genome_mod
import logging

logger = logging.getLogger('Logger.NEAT.population')

class Population:
    def __init__(self, genomes, species_keys=list([]), delta_t=1., c1=1., c2=1., c3=0.4, desired_species=10, min_species=2,
                 p_weight_mut = 0.8, p_weight_random=0.02, weight_mut_sigma=0.1,
                 node_mut_rate=1.01, edge_mut_rate=1.05,
                 p_child_clone=0.25, p_mutate=1.8, p_inter_species=0.02,
                 weight_amplitude=1):
        self.delta_t = delta_t
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.p_child_clone = p_child_clone       # Prob for child to be a copy of its parent
        self.p_mutate = p_mutate       # Prob for child ot be mutated copy of its parent
        self.p_inter_species = p_inter_species
        self.n_in, self.n_out = genomes[0].n_in, genomes[0].n_out
        self.p_weight_mut = p_weight_mut
        self.p_weight_random = p_weight_random
        self.weight_mut_sigma = weight_mut_sigma
        self.weight_amplitude = weight_amplitude
        self.node_mut_rate   =   node_mut_rate
        self.edge_mut_rate   =   edge_mut_rate
        self.pop_size = len(genomes)
        self.desired_species = desired_species
        self.min_species = min_species
        self.genomes = genomes
        self.speciate()
        #for core in species_keys:
        #    self.all_species[deepcopy(core)] = []

    def distance(self, genome1, genome2):
        gene1_index = 0
        gene2_index = 0
        N1 = len(genome1.connections)
        #print("N1: ", N1)
        N2 = len(genome2.connections)
        N = max(N1, N2)
        W = 0       # Sum of weight differences
        E = D = S = 0   # Counters for excess, disjoint, and same genes
        while gene1_index < N1 and gene2_index < N2:
            #print("index 1: ", gene1_index)
            #print("index 2: ", gene2_index)
            gene1 = genome1.connections[gene1_index]
            gene2 = genome2.connections[gene2_index]
            if gene1.innov == gene2.innov:
                gene1_index += 1
                gene2_index += 1
                S += 1
                W += abs(gene1.weight - gene2.weight)
            elif gene1.innov > gene2.innov:
                #if gene2_index == N2 - 1:
                    #E += 1
                    #gene1_index += 1
                #else:
                D += 1
                gene2_index += 1
            else:           # gene2.innov > gene1.innov
                #if gene1_index == N1 - 1:
                #    E += 1
                #    gene2_index += 1
                #else:
                D += 1
                gene1_index += 1
        if gene1_index < N1:
            E += len(genome1.connections[gene1_index:])
        elif gene2_index < N2:
            E += len(genome2.connections[gene2_index:])
        #logger.info((E, D, S, W))
        #print("E, D, S, W:", E, D, S, W)
        E_eff = self.c1 * E / N
        D_eff = self.c2 * D / N
        if S == 0:
            W_eff = 0
        else:
            W_eff = self.c3 * W / S
        return E_eff + D_eff + W_eff

    def speciate(self, increased=False):
        #logger.info("Starting speciate with {}".format(self.delta_t))
        #if delta_t is None:
        #    delta_t = self.delta_t
        self.all_species = {}
        for i, genome in enumerate(self.genomes):
            for core in self.all_species.keys():
                if self.distance(genome, core) <= self.delta_t:
                    self.all_species[core].append(i)
                    break
            else:
                self.all_species[genome] = [i]
        if len(self.all_species.keys()) > self.desired_species:
            #logger.info("Too many species ({})".format(len(self.all_species.keys())))
            self.delta_t += 0.2
            self.speciate(increased = True)
        elif len(self.all_species.keys()) < self.min_species and self.delta_t > 0.3 and not increased:
            #logger.info("Too few species ({})".format(len(self.all_species.keys())))
            self.delta_t -= 0.2
            self.speciate()
        #logger.info("Finished speciate")

    def create_child(self, genome1, genome2):
        if genome1.score > genome2.score:
            fitter, unfitter = genome1, genome2
        else:
            fitter, unfitter = genome2, genome1

        child = []
        fit_index = unfit_index = 0
        while fit_index < len(fitter.connections):
            gene_fit = fitter.connections[fit_index]
            gene_unfit = unfitter.connections[unfit_index]

            if gene_fit.innov == gene_unfit.innov:
                if not gene_fit.enabled or not gene_unfit.enabled:
                    child.append(gene_fit)
                else:
                    try:
                        p_fit = fitter.score / (fitter.score + unfitter.score)
                    except ZeroDivisionError:
                        p_fit = 0.5
                    prob = random.random()
                    if prob < p_fit:
                        child.append(gene_fit)
                    else:
                        child.append(gene_unfit)
                    #child.append(random.choice([gene_fit, gene_unfit]))
                fit_index += 1
                if unfit_index + 1 < len(unfitter.connections):
                    unfit_index += 1
            elif gene_fit.innov < gene_unfit.innov:
                child.append(gene_fit)
                fit_index += 1
            else:
                if unfit_index + 1 < len(unfitter.connections):
                    unfit_index += 1
                else:
                    child.append(gene_fit)
                    fit_index += 1
            #print("Child list:")
            #for gene in child:
            #    print("In:", gene.node_in, "  Out:", gene.node_out)
            #print("Done")
        child_genome = genome_mod.Genome(self.n_in, self.n_out, child)
        return child_genome

    def mutate(self, genome):
        #logger.debug("Mutating genome")
        #no_cycle = False
        #while not no_cycle:
        #genome_copy = genome.copy()
        prob = random.random()
        if prob < self.p_weight_mut:
            for edge in genome.connections:
                prob = random.random()
                if prob <= self.p_weight_random:
                    edge.weight = random.uniform(-self.weight_amplitude, self.weight_amplitude)
                else:
                    edge.weight += random.gauss(0, self.weight_mut_sigma)
        prob = random.random()
        if prob <= self.node_mut_rate:
            #logger.debug("Adding node!")
            genome.add_random_node(self.weight_amplitude)
        prob = random.random()
        if prob <= self.edge_mut_rate:
            #logger.debug("Adding connection!")
            genome.random_connection(self.weight_amplitude)
        #no_cycle = not genome_copy.has_cycle()
        #if no_cycle:
        #    break
        #if not no_cycle:
        #logger.debug("Cycle found. Rejecting")
            #genome_mod.print_edges(genome_copy)
        #genome_mod.draw_genome_net(genome_copy, show_disabled=True, show_innov=True)
            #time.sleep(0.5)

        return genome

    def generate_offspring(self):
        children = []
        species_scores = {}
        species_sorted = {}
        # Sort each species by score and calculate total score
        for key, value in self.all_species.items():
            species_scores[key] = 0
            species_sorted[key] = []
            if len(value) == 0:
                continue
            for genome_id in value:
                genome = self.genomes[genome_id]
                species_scores[key] += genome.score
                for i in range(len(species_sorted[key])):
                    if genome.score > species_sorted[key][i].score:
                        species_sorted[key].insert(i, genome)
                        break
                else:
                    species_sorted[key].append(genome)
            species_scores[key] /= len(value)**2        # **2 experimental, if they meant average instead of sum

        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        # Create children for each species
        total_score = sum([score for _, score in species_scores.items()])
        for key, value in self.all_species.items():
            if total_score == 0:
                n_offspring = len(value)
            else:
                n_offspring = int(round(self.pop_size * species_scores[key] / total_score))
            #logger.info("    n_offspring: {}".format(n_offspring))
            #print("sorted species ", species_sorted[key])
            #print([instance.score for instance in species_sorted[key]])
            reproducers = species_sorted[key][: int(round((len(species_sorted[key]) * 0.9)))]    # remove worst 10%
            #print([instance.score for instance in reproducers])
            p = np.array([2 * gaussian(i, 0, len(reproducers)/4) for i in range(len(reproducers))])
            p = p / sum(p)  # normalize so probs sum up to 1
            #print(len(species_sorted[key]), int(round((len(species_sorted[key]) * 0.9))))
            for i in range(n_offspring):
                prob = random.random()
                parent1 = np.random.choice(reproducers, p=p)
                if prob < self.p_child_clone:
                    child = parent1.copy()
                elif prob < self.p_child_clone + self.p_inter_species:
                    parent2 = random.choice(self.genomes)
                    child = self.create_child(parent1, parent2)
                else:
                    parent2 = np.random.choice(reproducers, p=p)
                    child = self.create_child(parent1, parent2)
                #genome_mod.draw_genome_net(child, show_weights=False, show_disabled=True, show_innov=True, filename="child-before")
                    #print_edges(child)
                prob = random.random()
                if prob < self.p_mutate:
                    child = self.mutate(child)
                #genome_mod.draw_genome_net(child, show_weights=False, show_disabled=True, show_innov=True, filename="child-after")
                children.append(child)

        # select species-cores for offspring
        new_cores = []
        for key, value in self.all_species.items():
            if len(value) == 0:
                continue
            new_core = self.genomes[random.choice(value)]
            new_cores.append(new_core)

        offspring = Population(children, new_cores, delta_t=self.delta_t, c1=self.c1, c2=self.c2, c3=self.c3,
                        desired_species=self.desired_species, min_species=self.min_species,
                        p_weight_mut=self.p_weight_mut, p_weight_random=self.p_weight_random, weight_mut_sigma=self.weight_mut_sigma,
                        node_mut_rate=self.node_mut_rate, edge_mut_rate=self.edge_mut_rate,
                        p_child_clone=self.p_child_clone, p_mutate=self.p_mutate, p_inter_species=self.p_inter_species)

        return offspring

    def get_best(self):
        best_genome, best_score = self.genomes[0], self.genomes[0].score
        for genome in self.genomes:
            if genome.score > best_score:
                best_genome, best_score = genome, genome.score
        return best_genome

    def build_phenotypes(self):
        self.phenotypes = [genome.build_phenotype() for genome in self.genomes]
        return self.phenotypes