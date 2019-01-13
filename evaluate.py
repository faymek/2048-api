from game2048.game import Game
from game2048.displays import Display

from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent
import numpy as np
import keras
import time

ntest = 50
model = keras.models.load_model('best/model.h5')

OUT_SHAPE = (4,4)
CAND = 16
map_table = {2**i : i for i in range(1,CAND)}
map_table[0] = 0

def grid_one(arr):
    ret = np.zeros(shape=OUT_SHAPE+(CAND,),dtype=bool)  # shape = (4,4,16)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,arr[r,c]] = 1
    return ret

class MyAgent(Agent):

    def __init__(self, game, display=None):
        super().__init__(game, display)
        self.testgame = Game(4, random=False)
        self.testgame.enable_rewrite_board = True

    def step(self):
        piece = [map_table[k] for k in self.game.board.astype(int).flatten().tolist()]
        x0 = np.array([ grid_one(np.array(piece).reshape(4,4)) ])
        preds = list(model.predict(x0))
        direction = np.argmax(preds[0])
        return direction


def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50

    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=MyAgent)
        scores.append(score)

    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
