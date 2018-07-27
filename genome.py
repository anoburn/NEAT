import random
import graphviz
import numpy as np
from copy import deepcopy, copy
import logging

logger = logging.getLogger('Logger.NEAT.population.genome')

innov = 0

class ConnectGene:
    def __init__(self, node_in, node_out, weight, innov, enabled=True):
        self.node_in    = node_in
        self.node_out   = node_out
        self.weight     = weight
        self.enabled    = enabled
        self.innov      = innov


class Genome:
    def __init__(self, n_in, n_out, connections=None):
        self.score = 0
        self.n = n_in + n_out
        self.n_in = n_in
        self.n_out = n_out
        self.outputs = [i for i in range(n_out)]
        self.inputs = [i for i in range(n_out, n_out + n_in)]
        self.hidden = []
        self.connections = []
        if connections is None:
            for node_in in self.inputs:
                for node_out in self.outputs:
                    self.add_connection(node_in, node_out)
        else:
            for edge in connections:
                self.add_connection(edge.node_in, edge.node_out, edge.weight, edge.enabled, edge.innov)

    def copy(self):
        genome_copy = Genome(self.n_in, self.n_out, self.connections)
        return genome_copy

    def add_connection(self, node_in, node_out, weight=None, enabled=True, innov_used=None):
        if node_in not in (self.outputs + self.inputs + self.hidden):
            self.hidden.append(node_in)
            self.n += 1
        if node_out not in (self.outputs + self.inputs + self.hidden):
            self.hidden.append(node_out)
            self.n += 1
        if weight is None:
            weight = random.uniform(-1., 1.)
        if innov_used is None:
            global innov
            innov_used = innov
            innov += 1
        connection = ConnectGene(node_in, node_out, weight, innov_used, enabled)
        self.connections.append(connection)

    def random_connection(self, amplitude=1.):
        if len(self.connections) == self.n ** 2:
            print("Already all connections")
            return

        #print_edges(self)

        # compare this with looping over neurons
        inputs  = np.array(self.inputs + self.hidden)
        outputs = np.array(self.hidden + self.outputs)
        #print("Random connection: inputs:",  inputs)
        #print(self.inputs)
        #print(self.hidden)
        #print(self.outputs)
        #print("Random connection: outputs:", outputs)
        v_in, v_out = np.meshgrid(inputs, outputs)
        available_connections = list(np.stack((v_in.flatten(), v_out.flatten()), axis=1).tolist())
        #print("Available connections (initial): ", available_connections)

        elders = {i: set() for i in self.outputs + self.hidden + self.inputs}   # save all nodes from which a node receives (inherited) signal to avoid loops
        for edge in self.connections:
            elders[edge.node_out].add(edge.node_in)
            #nodes = [edge.node_in, edge.node_out]
            #print(nodes)
            available_connections.remove([edge.node_in, edge.node_out])
            #try:
            #    available_connections.remove([edge.node_out, edge.node_in])
            #except ValueError:
            #    pass
        def get_elders(node):
            all_elders = elders[node]
            for elder in elders[node]:
                all_elders = all_elders | get_elders(elder)
            return all_elders

        for node in self.hidden:
            for elder in get_elders(node):
                if elder not in self.inputs:
                    #logger.info('{}, {}'.format(node, elder))
                    available_connections.remove([node, elder])

        for node in self.hidden:
            available_connections.remove([node, node])
        if len(available_connections) == 0:
            return
        node_in, node_out = random.choice(available_connections)
        weight = random.uniform(-amplitude, amplitude)
        self.add_connection(node_in, node_out, weight)

    def add_random_node(self, amplitude=1.):
        #print("All edges:")
        #print_edges(self)
        #print("Without disabled:")
        #for edge in [edge for edge in self.connections if edge.enabled]:
        #    print(edge.node_in, " -> ", edge.node_out)
        old_edge = random.choice([edge for edge in self.connections if edge.enabled])
        #print("Old edge: ", old_edge.node_in, " -> ", old_edge.node_out)
        old_edge.enabled = False
        new_weight = random.uniform(-amplitude, amplitude)
        #print("Self.n = ", self.n)
        new_node = self.n
        self.add_connection(old_edge.node_in, new_node,     old_edge.weight)
        self.add_connection(new_node,         old_edge.node_out, new_weight)
        return old_edge.node_in, old_edge.node_out

    def has_cycle(self):
        #logger.debug("n_in: {}, n_out: {}, n: {}".format(self.n_in, self.n_out, self.n))
        #logger.debug("inputs: {}, outputs: {}, hidden: {}".format(self.inputs, self.outputs, self.hidden))

        M = np.zeros((self.n, self.n))
        for connection in self.connections:
            M[connection.node_out, connection.node_in] += connection.weight
        M_iter = deepcopy(M)
        M_eff = M_iter[: self.n_out, self.n_out: self.n_in + self.n_out]
        iteration_counter = 0
        while len(np.nonzero(M_iter)[0]) > 0:
            iteration_counter += 1
            if iteration_counter > len(self.connections):
                #print("Found cycle. Rejecting")
                return True
            M_iter = M @ M_iter
            M_eff += M_iter[: self.n_out, self.n_out: self.n_in + self.n_out]
        return False

    def build_phenotype_new(self):
        """ Start of new method for building the phenotype """
        sources = {}
        for edge in self.connections:
            sources[edge.node_out] = (edge.node_in, edge.weight)

    def build_phenotype(self):
        M = np.zeros((self.n, self.n))
        for connection in self.connections:
            if connection.enabled:
                M[connection.node_out, connection.node_in] += connection.weight
        M_iter = deepcopy(M)
        M_eff  = M_iter[: self.n_out, self.n_out : self.n_in + self.n_out]
        iteration_counter = 0
        # print(M_iter)
        #while np.max(np.nonzero(M_iter)[0], 0) >= self.n_out or np.max(np.nonzero(M_iter)[1], 0) >= self.n_in + self.n_out:
        while len(np.nonzero(M_iter)[0]) > 0:
            #print('First: ', np.max(np.nonzero(M_iter)[0], 0))
            #print('Second:', np.max(np.nonzero(M_iter)[1], 0))

            iteration_counter += 1
            if iteration_counter > len(self.connections):
                print("Found loop. Breaking")
                M_eff = np.zeros((self.n_out, self.n_in))
                break
            # print("looping")
            M_iter = M @ M_iter
            M_eff += M_iter[: self.n_out, self.n_out: self.n_in + self.n_out]

        #print(M_eff.shape)
        phenotype = lambda x: M_eff @ x
        return phenotype


def draw_genome_net(genome, show_disabled=False, filename=None, show_weights=False, show_innov=False, view=True):
    node_attrs = {'fontsize': '9', 'height': '0.2', 'width': '0.2'}
    dot = graphviz.Digraph(format='png', node_attr=node_attrs)
    in_graph = graphviz.Digraph('input subgraph')
    in_graph.graph_attr.update(rank='min')
    out_graph = graphviz.Digraph('output subgraph')
    out_graph.graph_attr.update(rank='max')

    for node in genome.inputs:
        input_attrs = {'style': 'filled', 'shape': 'circle', 'fillcolor': 'lightgrey'}
        name = str(node)
        in_graph.node(str(node), label='in '+name, _attributes=input_attrs)

    for node in genome.outputs:
        output_attrs = {'style': 'filled', 'shape': 'circle', 'fillcolor': 'lightblue'}
        name = str(node)
        out_graph.node(str(node), label='out '+name, _attributes=output_attrs)

    for node in genome.hidden:
        hidden_attrs = {'style': 'filled', 'shape': 'circle', 'fillcolor': 'white'}
        dot.node(str(node), label=str(node), _attributes=hidden_attrs)

    # invisible connections for right order
    for i, node in enumerate(genome.inputs):
        if i < len(genome.inputs) - 1:
            dot.edge(str(node), str(genome.inputs[i+1]), style='invis')

    for i, node in enumerate(genome.outputs):
        if i < len(genome.outputs) - 1:
            dot.edge(str(node), str(genome.outputs[i+1]), style='invis')

    for edge in genome.connections:
        node_in, node_out = edge.node_in, edge.node_out
        style = 'solid' if edge.enabled else 'dotted'
        color = 'red' if edge.weight > 0 else 'blue'
        width = str(0.1 + abs(edge.weight / 1.0))
        if show_weights:
            label = str(round(edge.weight,2))
        elif show_innov:
            label = str(edge.innov)
        else:
            label = ''
        if edge.enabled or show_disabled:
            dot.edge(str(node_in), str(node_out), label=label, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.subgraph(in_graph)
    dot.subgraph(out_graph)
    #dot.view()
    if filename is None:
        filename = 'network'
    assert type(filename) == str
    dot.render(filename, view=view)
    return dot


def print_edges(genome):
    print("Printing edges")
    for edge in genome.connections:
        print(edge.node_in, " -> ", edge.node_out, "      innov: ", edge.innov)
    print("Done")


if __name__ == "__main__":
    genotype = Genome(3, 1)
    genotype2 = deepcopy(genotype)
    draw_genome_net(genotype, show_disabled=True, filename="initial")
    print(genotype.has_cycle())
    #genotype.add_connection(2, 4)
    #genotype.add_connection(3, 4)
    #genotype.add_connection(4, 0)
    genotype.add_random_node()
    genotype.add_random_node()
    genotype.random_connection()
    draw_genome_net(genotype, show_disabled=True, filename="after")
    print(genotype.has_cycle())

    draw_genome_net(genotype2, show_disabled=True, filename="copy")

    connection = ConnectGene(3, 1, 1.5, 0)
    connection2 = copy(connection)
    connection.weight = 20
    print(connection2.weight)