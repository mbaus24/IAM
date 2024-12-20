# coding:utf-8

import random
import setup
import qlearn
import sarsa
import config as cfg
from queue import Queue
import ExpectedSarsa
from IPython.display import clear_output
import matplotlib.pyplot as plt

import os 
import sys 

original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

# reload(setup)
# reload(qlearn)


def pick_random_location():
    while 1:
        x = random.randrange(world.width)
        y = random.randrange(world.height)
        cell = world.get_cell(x, y)
        if not (cell.wall or len(cell.agents) > 0):
            return cell


class Cat(setup.Agent):
    def __init__(self, filename):
        self.cell = None
        self.catWin = 0
        self.color = cfg.cat_color
        with open(filename) as f:
            lines = f.readlines()
        lines = [x.rstrip() for x in lines]
        self.fh = len(lines)
        self.fw = max([len(x) for x in lines])
        self.grid_list = [[1] * self.fw for _ in range(self.fh)]
        self.move = [(0, -1), (1, -1), (
                1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

        for y in range(self.fh):
            line = lines[y]
            for x in range(min(self.fw, len(line))):
                t = 1 if (line[x] == 'X') else 0
                self.grid_list[y][x] = t

        print('cat init success......')

    # using BFS algorithm to move quickly to target.
    def bfs_move(self, target):
        if self.cell == target:
            return

        for n in self.cell.neighbors:
            if n == target:
                self.cell = target  # if next move can go towards target
                return

        best_move = None
        q = Queue()
        start = (self.cell.y, self.cell.x)
        end = (target.y, target.x)
        q.put(start)
        step = 1
        V = {}
        preV = {}
        V[(start[0], start[1])] = 1

        #print('begin BFS......')
        while not q.empty():
            grid = q.get()

            for i in range(8):
                ny, nx = grid[0] + self.move[i][0], grid[1] + self.move[i][1]
                if nx < 0 or ny < 0 or nx > (self.fw-1) or ny > (self.fh-1):
                    continue
                if self.get_value(V, (ny, nx)) or self.grid_list[ny][nx] == 1:  # has visit or is wall.
                    continue

                preV[(ny, nx)] = self.get_value(V, (grid[0], grid[1]))
                if ny == end[0] and nx == end[1]:
                    V[(ny, nx)] = step + 1
                    seq = []
                    last = V[(ny, nx)]
                    while last > 1:
                        k = [key for key in V if V[key] == last]
                        seq.append(k[0])
                        assert len(k) == 1
                        last = preV[(k[0][0], k[0][1])]
                    seq.reverse()
                    #print(seq)

                    best_move = world.grid[seq[0][0]][seq[0][1]]

                q.put((ny, nx))
                step += 1
                V[(ny, nx)] = step

        if best_move is not None:
            self.cell = best_move

        else:
            dir = random.randrange(cfg.directions)
            self.go_direction(dir)
            #print("!!!!!!!!!!!!!!!!!!")

    def get_value(self, mdict, key):
        try:
            return mdict[key]
        except KeyError:
            return 0

    def update(self):
       # print('cat update begin..')
        if self.cell != mouse.cell:
            self.bfs_move(mouse.cell)
            #print('cat move..')


class Cheese(setup.Agent):
    def __init__(self):
        self.color = cfg.cheese_color

    def update(self):
        #print('cheese update...')
        pass


class Mouse(setup.Agent):
    def __init__(self):
        self.ai = None
        if cfg.algorithm == 'qlearn':
            self.ai = qlearn.QLearn(actions=range(cfg.directions), alpha=0.1, gamma=0.9, epsilon=0.1)
        elif cfg.algorithm == 'sarsa':
            self.ai = sarsa.Sarsa(actions=range(cfg.directions), alpha=0.1, gamma=0.9, epsilon=0.1)
        elif cfg.algorithm == 'expected_sarsa':
            self.ai = ExpectedSarsa.ExpectedSarsa(actions=range(cfg.directions), alpha=0.1, gamma=0.9, epsilon=0.1)
        else:
            raise ValueError(f"Unknown algorithm: {cfg.algorithm}")
        self.catWin = 0
        self.mouseWin = 0
        self.lastState = None
        self.lastAction = None
        self.color = cfg.mouse_color

        print('mouse init...')

    def update(self):
       # print('mouse update begin...')
        state = self.calculate_state()
        reward = cfg.MOVE_REWARD

        if self.cell == cat.cell:
            #print('eaten by cat...')
            self.catWin += 1
            reward = cfg.EATEN_BY_CAT
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, state, reward)
                #print('mouse learn...')
            self.lastState = None
            self.cell = pick_random_location()
            #print('mouse random generate..')
            return

        if self.cell == cheese.cell:
            self.mouseWin += 1
            reward = cfg.EAT_CHEESE
            cheese.cell = pick_random_location()

        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, state, reward)

        # choose a new action and execute it
        action = self.ai.choose_action(state)
        self.lastState = state
        self.lastAction = action
        self.go_direction(action)

    def calculate_state(self):
        def cell_value(cell):
            if cat.cell is not None and (cell.x == cat.cell.x and cell.y == cat.cell.y):
                return 3
            elif cheese.cell is not None and (cell.x == cheese.cell.x and cell.y == cheese.cell.y):
                return 2
            else:
                return 1 if cell.wall else 0

        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return tuple([cell_value(world.get_relative_cell(self.cell.x + dir[0], self.cell.y + dir[1])) for dir in dirs])


if __name__ == '__main__':
    mouse = Mouse()
    cat = Cat(filename='TP/apprentissage renforcement/QLearningMouse_py3/resources/world.txt')
    cheese = Cheese()
    world = setup.World(filename='TP/apprentissage renforcement/QLearningMouse_py3/resources/world.txt')

    world.add_agent(mouse)
    world.add_agent(cheese, cell=pick_random_location())
    world.add_agent(cat, cell=pick_random_location())

    # world.display.activate()
    world.display.speed = cfg.speed

    plt.ion()
    fig, ax = plt.subplots()
    mouse_wins = []
    cat_wins = []
    steps = []

    while True:
        world.update(mouse.mouseWin, mouse.catWin)
        mouse_wins.append(mouse.mouseWin)
        cat_wins.append(mouse.catWin)
        steps.append(len(steps) + 1)

        if len(steps) % 20000 == 0:  # Update plot every 100 steps to reduce overhead
            ax.clear()
            ax.plot(steps, [m / ((c if c != 0 else 1) + m) for m, c in zip(mouse_wins, cat_wins)], label='mouse_wins/cat_wins')
            ax.legend()
            ax.set_xlabel('Steps')
            ax.set_ylabel('Wins')
            ax.set_title('Mouse Wins vs Cat Wins Algorithm: ' + cfg.algorithm)  

            clear_output(wait=True)
            plt.pause(0.0001)
