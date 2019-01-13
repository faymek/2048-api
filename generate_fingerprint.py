import json
import numpy as np
from game2048.game import Game

from game2048.displays import Display, IPythonDisplay
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


def generate_fingerprint(AgentClass, **kwargs):
    with open("board_cases.json") as f:
        board_json = json.load(f)

    game = Game(size=4, enable_rewrite_board=True)
    agent = AgentClass(game=game, **kwargs)

    trace = []
    for board in board_json:
        game.board = np.array(board)
        direction = agent.step()
        trace.append(direction)
    fingerprint = "".join(str(i) for i in trace)
    return fingerprint


if __name__ == '__main__':
    from collections import Counter


    fingerprint = generate_fingerprint(MyAgent)

    with open("EE369_fingerprint.json", 'w') as f:        
        pack = dict()
        pack['fingerprint'] = fingerprint
        pack['statstics'] = dict(Counter(fingerprint))
        f.write(json.dumps(pack, indent=4))
