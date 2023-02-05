from main import *
from model import *
from collections import deque
from helper import plot

MAX_MEMORY = 1000_000
BATCH_SIZE = 10_000
LR = 0.001

class Agent:

    def __init__(self,game):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.game = game
        self.gamma= 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11,256,3)
        #self.model.load()
        self.trainer = Qtrainer(LR,self.gamma,self.model)

    def get_state(self):
        state = [
        self.check_danger_straight(self.game.direction),
        self.check_danger_right(self.game.direction),
        self.check_danger_left(self.game.direction),

        self.game.direction == (-1, 0),
        self.game.direction == (1, 0),
        self.game.direction == (0, -1),
        self.game.direction == (0, 1),

        self.game.foodX < self.game.X,
        self.game.foodX > self.game.X,
        self.game.foodY < self.game.Y,
        self.game.foodY > self.game.Y
        ]

        return numpy.array(state,dtype=int)

    def train(self,state, new_state, action, reward, game_over):
        self.trainer.train_step(state, new_state, action, reward, game_over)

    def remember(self, state, action, reward, new_state, game_over):
        self.memory.append(
            (state, action, reward, new_state, game_over)
        )  # popleft() on overflow

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, new_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, new_states, actions, rewards, game_overs)


    def get_action(self,state):
        self.epsilon = 100 - self.n_games
        move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            index = random.randint(0, 2)
        else:
            action = self.trainer.predict(state)
            index = torch.argmax(action).item()
        move[index] = 1
        return move

    def action_to_direction(self,move,direction):
        index = move.index(1)
        if index == 0:
            return self.get_left(direction)
        elif index == 1:
            return direction
        elif index == 2:
            return self.get_right(direction)

    def get_right(self, direction):
        return (-1 * direction[1], direction[0])

    def get_left(self, direction):
        return (direction[1], -1 * direction[0])

    def check_danger_straight(self,direction):
        newX, newY = self.game.X + direction[0]*WIDTH_SNAKE, self.game.Y + direction[1]*WIDTH_SNAKE
        return self.game.check_death(newX,newY)

    def check_danger_right(self, direction):
        new_direction = self.get_right(direction)
        newX, newY = self.game.X + new_direction[0] * WIDTH_SNAKE, self.game.Y + new_direction[1] * WIDTH_SNAKE
        return self.game.check_death(newX, newY)

    def check_danger_left(self, direction):
        new_direction = self.get_left(direction)
        newX, newY = self.game.X + new_direction[0] * WIDTH_SNAKE, self.game.Y + new_direction[1] * WIDTH_SNAKE
        return self.game.check_death(newX, newY)


def play_agent():
    game = Snake()
    agent = Agent(game)
    record = 0
    total_score = 0
    plot_scores = []
    plot_mean_scores = []

    
    while True:
        state = agent.get_state()
        action = agent.get_action(state)
        direction = agent.action_to_direction(action,game.direction)
        reward, game_over, score = game.take_step(direction)
        new_state = agent.get_state()
        agent.train(state, new_state, action, reward, game_over)
        agent.remember(state, action, reward, new_state, game_over)
        if game_over:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
                print("New high score:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    play_agent()