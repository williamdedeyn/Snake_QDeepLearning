import pygame
import random
import numpy

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_COLOR = (144,238,144)
FOOD_COLOR = (255,0,0)
WIDTH_SNAKE = 20
N_ROWS = SCREEN_HEIGHT // WIDTH_SNAKE
N_COLUMNS = SCREEN_WIDTH // WIDTH_SNAKE


class Snake:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen.fill((144, 238, 144))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.snake_speed = 60
        self.direction = (1, 0)
        self.X = 400
        self.Y = 300
        self.color = (0,0,0)
        self.snake = []
        self.total = 1
        pygame.draw.rect(self.screen, self.color, pygame.Rect(self.X, self.Y, WIDTH_SNAKE, WIDTH_SNAKE), 0)
        self.spawn_food()
        self.iteration = 0
        self.score = 0

    def take_step(self,direction):
        reward = 0
        game_over = False
        self.iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
        self.direction = direction
        self.update()
        if self.check_death(self.X,self.Y):
            reward = -10
            game_over = True
            return reward, game_over, self.score
        if self.iteration >= 500:
            reward = -20
            game_over = True
            return reward, game_over, self.score
        if self.eat_food():
            reward = 10
        self.draw()
        pygame.display.update()
        self.clock.tick(self.snake_speed)

        return reward, game_over, self.score

    def reset(self):
        self.direction = (1, 0)
        self.X = 400
        self.Y = 300
        self.snake = []
        self.total = 1
        pygame.draw.rect(self.screen, self.color, pygame.Rect(self.X, self.Y, WIDTH_SNAKE, WIDTH_SNAKE), 0)
        self.spawn_food()
        self.iteration = 0
        self.score = 0

    def update(self):
        self.X = self.X + self.direction[0]*WIDTH_SNAKE
        self.Y = self.Y + self.direction[1]*WIDTH_SNAKE
        self.snake.append((self.X,self.Y))
        if len(self.snake) > self.total:
            del self.snake[0]

    def draw(self):
        self.screen.fill(SCREEN_COLOR)
        for coor in self.snake:
            pygame.draw.rect(self.screen, self.color,pygame.Rect(coor[0],coor[1],WIDTH_SNAKE,WIDTH_SNAKE), 0)
        pygame.draw.rect(self.screen, FOOD_COLOR, pygame.Rect(self.foodX, self.foodY, WIDTH_SNAKE, WIDTH_SNAKE), 0)

    def check_death(self,X,Y):
        for coor in self.snake[:-1]:
            if coor ==(X,Y):
                return True
        if X + WIDTH_SNAKE > SCREEN_WIDTH or \
            Y + WIDTH_SNAKE > SCREEN_HEIGHT or \
            X < 0 or Y < 0:
            return True
        return False

    def eat_food(self):
        if self.X == self.foodX and self.Y == self.foodY:
            self.spawn_food()
            self.score += 1
            self.total += 1
            self.iteration = 0
            return True
        return False

    def spawn_food(self):
        self.foodX = (random.randrange(0,SCREEN_WIDTH//WIDTH_SNAKE))*WIDTH_SNAKE
        self.foodY = (random.randrange(0, SCREEN_HEIGHT//WIDTH_SNAKE))*WIDTH_SNAKE
        if (self.foodX,self.foodY) in self.snake:
            self.spawn_food()

    #
    # def check_direction(self,new_direction):
    #     if new_direction[0] + self.direction[0] == 0 and new_direction[0] != 0:
    #         return False
    #     if new_direction[1] + self.direction[1] == 0 and new_direction[1] != 0:
    #         return False
    #     return True



