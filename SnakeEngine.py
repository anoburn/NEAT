from random import randint
import numpy as np
import pygame
import time

class Rectangle:
    def __init__(self, _width, _height, _line):
        self.w = _width
        self.h = _height
        self.l = _line

RGB = {
    0: (255, 255, 255),  # 0 = white
    1: (  0,   0,   0),  # 1 = black
    2: (255,   0,   0),  # 2 = red
    3: (  0,   0, 255)   # 3 = blue
}

directions = {
    0:    (0, -1),      # left
    1:    (1,  0),      # up
    2:    (0,  1),      # right
    3:    (-1, 0)}      # down




def get_RGB(grid, _row, _column):
        return RGB[grid[_row][_column]]


def get_Key():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 4
                if event.key == pygame.K_UP:
                    return 5







def leftStart(FieldSizeX,FieldSizeY):

    xHead = randint(0,FieldSizeX-1)
    yHead = randint(1,FieldSizeY-4)
    snake = [[xHead,yHead], [xHead,yHead+1], [xHead,yHead+2]]                                     # Initial snake coordinates

    food = [randint(0, FieldSizeX-1), randint(0, FieldSizeY-1)]                                                       # Initial food coordinates
    while food in snake:
        food = [randint(0, FieldSizeX-1), randint(0, FieldSizeY-1)]                                                             # Draw food
    key = 0                                                                               # Start direction snake
    return food,snake,key

def upStart(FieldSizeX,FieldSizeY):
    xHead = randint(1,FieldSizeX-4)
    yHead = randint(0,FieldSizeX-1)
    snake = [[xHead,yHead], [xHead+1,yHead], [xHead+2,yHead]]                                     # Initial snake coordinates

    food = [randint(0, FieldSizeX-1), randint(0, FieldSizeY-1)]                                                       # Initial food coordinates
    while food in snake:
        food = [randint(0, FieldSizeX-1), randint(0, FieldSizeY-1)]                                                             # Draw food
    key = 1                                                                                 # Start direction snake
    return food,snake,key

def rightStart(FieldSizeX,FieldSizeY):
    xHead = randint(3,FieldSizeX-2)
    yHead = randint(0,FieldSizeY-1)
    snake = [[xHead,yHead], [xHead,yHead-1], [xHead,yHead-2]]                                     # Initial snake coordinates

    food = [randint(0, FieldSizeX-1), randint(0, FieldSizeY-1)]                                                       # Initial food coordinates
    while food in snake:
        food = [randint(0, FieldSizeX-1), randint(0, FieldSizeY-1)]                                                            # Draw food
    key = 2                                                                             # Start direction snake
    return food,snake,key

def downStart(FieldSizeX,FieldSizeY):
    xHead = randint(0,FieldSizeX-1)
    yHead = randint(3,FieldSizeY-2)
    snake = [[xHead,yHead], [xHead-1,yHead], [xHead-2,yHead]]                                     # Initial snake coordinates

    food = [randint(1, FieldSizeX-2), randint(1, FieldSizeY-2)]                                                       # Initial food coordinates
    while food in snake:
        food = [randint(1, FieldSizeX-2), randint(1, FieldSizeY-2)]                                                             # Draw food
    key = 3                                                                                # Start direction snake
    return food,snake,key


def switch_start(argument,FieldSizeX,FieldSizeY):
    switcher = {
        1: rightStart,
        2: downStart,
        3: leftStart,
        4: upStart,
    }

    startIni = switcher.get((argument))
    food,snake,key = startIni(FieldSizeX,FieldSizeY)
    #print('start',key,snake)
    return food,snake





class SnakeQ(object):

    def __init__(self, FieldSizeX, FieldSizeY):
        self.FieldSizeX = FieldSizeX
        self.FieldSizeY = FieldSizeY
        self.all_score = []
        self.WindowSizeX = FieldSizeX*11 + 20
        self.WindowSizeY = FieldSizeY*11 + 20
        self.grid = np.zeros((FieldSizeX, FieldSizeY))
        self.show = False

        self.FPS=20
        #self.FPS=0.5
        self.size = (self.WindowSizeX, self.WindowSizeY)

    def start(self):
        startDirection  = randint(1, 4)                                            # random start direction 1: right, 2: down, 3: left, 4: up
        food,snake      = switch_start(startDirection,self.FieldSizeX,self.FieldSizeY)
        self.food       = food
        self.snake      = snake
        self.alive      = True
        self.score      = 0
        self.score_old  = 0
        self.rec = Rectangle(10, 10, 1)
        self.grid = np.zeros((self.FieldSizeX, self.FieldSizeY))
        # Update grid. body: 1    food: 2
        for part in self.snake:
            self.grid[part[0]][part[1]] = 1
        self.grid[self.food[0]][self.food[1]] = 2
        pygame.init()

        self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()



    def move(self, key):
        self.grid = np.zeros((self.FieldSizeX, self.FieldSizeY))
        #print(key)
        #print(self.snake)
        # Calculates the new coordinates of the head of the snake
        self.snake.insert(0, [self.snake[0][0] + (key == 3 and 1) + (key == 1 and -1), self.snake[0][1] + (key == 0 and -1) + (key == 2 and 1)])

        # If snake crosses the boundaries kill it
        if self.snake[0][0] == -1: self.alive = False
        if self.snake[0][1] == -1: self.alive = False
        if self.snake[0][0] == self.FieldSizeX:  self.alive = False
        if self.snake[0][1] == self.FieldSizeY: self.alive = False
        # If snake runs into itself kill it
        if self.snake[0] in self.snake[1:]:   self.alive = False

        self.score_old = self.score

        # When snake eats the food
        if self.snake[0] == self.food:
            self.food = []
            self.score += 1
            while self.food == []:
                self.food = [randint(0, self.FieldSizeX-1), randint(0, self.FieldSizeY-1)]                 # Calculating next food's coordinates
                if self.food in self.snake: self.food = []
        else:
            self.snake.pop()


        if self.alive:
             # Update grid
            for part in self.snake:
                self.grid[part[0]][part[1]] = 1
            self.grid[self.food[0]][self.food[1]] = 2
            self.grid[self.snake[0][0]][self.snake[0][1]] = 3   # head



        if self.show:
            # Update window
            for row in range(self.FieldSizeX):
                for column in range(self.FieldSizeY):
                    color = get_RGB(self.grid, row, column)
                    pygame.draw.rect(self.screen, color, [(self.rec.l + self.rec.w) * column + self.rec.l + 10,
                                                     (self.rec.l + self.rec.h) * row + self.rec.l + 10, self.rec.w, self.rec.h])
            self.clock.tick(self.FPS)
            pygame.display.flip()
















