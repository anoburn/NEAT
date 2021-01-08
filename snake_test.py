import random
import neat_optimizer
import logging
import matplotlib.pyplot as plt
import numpy as np
import SnakeEngine
import pygame

logger = logging.getLogger('Logger')

def get_input(Snake):
    Input = np.zeros(6)
    xHead = Snake.snake[0][0]
    yHead = Snake.snake[0][1]
    for index in range(np.shape(Snake.snake)[0]):
        part = Snake.snake[index]
        if index == 0:
            MinLeft  = xHead
            MinUp    = yHead
            MinRight = Snake.FieldSizeX-(xHead+1)
            MinDown  = Snake.FieldSizeY-(yHead+1)
        else:
            if xHead == part[0]:
                if part[1]-yHead < 0:
                    MinUp = min([MinUp , np.absolute(part[1]-yHead)])
                else:
                    MinDown = min([MinDown , np.absolute(part[1]-yHead)])

            elif yHead == part[1]:
                if part[0]-xHead < 0:
                    MinLeft = min([MinLeft , np.absolute(xHead-part[0])])
                else:
                    MinRight = min([MinRight , np.absolute(xHead-part[0])])


    Input[0] = MinLeft
    Input[1] = MinUp
    Input[2] = MinRight
    Input[3] = MinDown
    Input[4] = Snake.snake[0][0]-Snake.food[0]
    Input[5] = Snake.snake[0][1]-Snake.food[1]
    return Input

def test_function(f):
    snake = SnakeEngine.SnakeQ(9, 9)
    snake.start()
    score = 0
    timeout = 0
    while snake.alive and timeout < 25:
        timeout += 1
        snake_input = get_input(snake)
        outputs = f(snake_input)
        key = np.argmax(outputs)
        snake.move(key)
        if snake.alive and snake.score_old < snake.score:
            # Reward is 1 if our score increased
            r = 200
            timeout = 0
        elif snake.alive:
            # nothing happens score is decreased for punishing long ways
            r = 0.1
        else:
            # Score decreased if the snake died
            r = -100
        score += r
    # pygame.quit()

    return score

if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    N_population = 400
    optimizer = neat_optimizer.NEAT_optimizer()
    optimizer.initialize(6, 4, N_population, "snake")

    max_generations = 400

    best_scores = []
    average_scores = []

    generations = []
    species_sizes = []
    plt.show()
    f, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set_xlim(0, max_generations)
    # ax1.set_ylim(0, 1)
    # ax1.axhline(0.5)
    line1, = ax1.plot(generations, best_scores, 'r-')
    line2, = ax1.plot(generations, average_scores, 'b-')

    for j in range(max_generations):
        # print("Starting generation {}".format(j))
        functions = optimizer.get_functions()
        scores = []
        for f in functions:
            score = test_function(f)
            #print("Score: ", score)
            scores.append(score)

        np.save(f'snake/genomes/best_genome_{j}', optimizer.population.genomes[np.argmax(scores)])

        ax3.cla()
        ax3.hist(scores, bins=20)
        # print("Scores: ", scores)
        #print("Mean score: ", sum(scores)/len(scores))
        optimizer.set_scores(scores)
        optimizer.print_gen_info()
        logger.info("    Highest score:    {}, Average score:    {}".format(max(scores), sum(scores)/len(scores)))
        if j % 20 == 0:
            # pass
            optimizer.draw_best()
        optimizer.generate_offspring()

        #print(best_scores)
        #print(generations)
        best_scores.append(max(scores))
        average_scores.append(sum(scores)/len(scores))
        generations.append(j)
        species_sizes.append(optimizer.species_sizes())
        max_n_species = max([len(sizes) for sizes in species_sizes])
        species_sizes_padded = np.zeros((max_n_species, j + 1))
        for i, gen_sizes in enumerate(species_sizes):
            for k, size in enumerate(gen_sizes):
                species_sizes_padded[k, i] = size
        line1.set_xdata(generations)
        line1.set_ydata(best_scores)
        line2.set_xdata(generations)
        line2.set_ydata(average_scores)
        ax1.relim()
        ax1.autoscale()

        ax2.clear()
        ax2.stackplot(np.array(generations), species_sizes_padded)
        ax2.set_xlim(0, max_generations)
        plt.draw()
        plt.pause(1e-17)

    plt.show()

    #for i, genome in enumerate(optimizer.population.genomes):
    #for i, genome in enumerate(optimizer.population.generate_offspring().genomes):
        #draw_genome_net(genome, filename="test/element%i"%i, view=False)