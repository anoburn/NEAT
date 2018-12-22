import random
import neat_optimizer
import logging
import matplotlib.pyplot as plt
import numpy as np
from genome import draw_genome_net


logger = logging.getLogger('Logger')

def test_function(f):
    inputs = [[0,0], [0,1], [1,0], [1,1]]
    outputs = [0, 1, 1, 0]

    correct = 0
    num_tests = 100
    score = 0
    for i in range(num_tests):
        index = random.randint(0, 3)
        input = inputs[index]
        output = outputs[index]
        result = f([*input, 1])[0]
        if result >= 0.5 and output == 1:
            correct += 1
        elif result < 0.5 and output == 0:
            correct += 1
        score += np.exp(-5 * abs(output - result))
    accuracy = correct / num_tests
    return score, accuracy



if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    optimizer = neat_optimizer.NEAT_optimizer()
    optimizer.initialize(3, 1, 150, "xor")
    max_generations = 50
    #optimizer.draw_best()
    best_accuracys = []
    average_accuracys = []
    generations = []
    plt.show()
    axes = plt.gca()
    axes.set_xlim(0, max_generations)
    axes.set_ylim(0, 1)
    axes.axhline(0.5)
    line1, = axes.plot(generations, best_accuracys, 'r-')
    line2, = axes.plot(generations, average_accuracys, 'b-')

    for j in range(max_generations):
        #print("Starting generation {}".format(j))
        functions = optimizer.get_functions()
        scores = []
        accuracys = []
        for f in functions:
            score, accuracy = test_function(f)
            #print("Score: ", score)
            scores.append(score)
            accuracys.append(accuracy)
        #print("Scores: ", scores)
        #print("Mean score: ", sum(scores)/len(scores))
        optimizer.set_scores(scores)
        optimizer.print_gen_info()
        logger.info("    Highest accuracy: {}, Average accuracy: {}".format(max(accuracys), sum(accuracys)/len(accuracys)))
        if j % 5 == 0:
            pass
            #optimizer.draw_best()
        optimizer.generate_offspring()

        #print(best_scores)
        #print(generations)
        best_accuracys.append(max(accuracys))
        average_accuracys.append(sum(accuracys)/len(accuracys))
        generations.append(j)
        line1.set_xdata(generations)
        line1.set_ydata(best_accuracys)
        line2.set_xdata(generations)
        line2.set_ydata(average_accuracys)
        plt.draw()
        plt.pause(1e-17)

    plt.show()

    #for i, genome in enumerate(optimizer.population.genomes):
    #for i, genome in enumerate(optimizer.population.generate_offspring().genomes):
        #draw_genome_net(genome, filename="test/element%i"%i, view=False)