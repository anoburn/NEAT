import genome as genome_mod
import numpy as np
from SnakeEngine import SnakeQ
from snake_test import get_input
import pygame

# FILE = 'best_genome_399.npy'
FILE = 'snake/genomes/best_genome_305.npy'
genome = np.load(FILE, allow_pickle=True).item()
print(genome)
f = genome.build_phenotype()

snake = SnakeQ(9, 9)
snake.show = True

while True:
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
    print(f'Score = {score}')

pygame.quit()