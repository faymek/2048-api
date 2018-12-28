from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent
import numpy as np
import keras

model1 = keras.models.load_model('dev/model.h5')
model2 = keras.models.load_model('best/model.h5')


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
        if game.score >= 512:
            preds = list(model2.predict(x0))
        else:
            preds = list(model1.predict(x0))
        direction = np.argmax(preds[0])
        return direction

scores = []
for i in range(50):
    game = Game(4, random=False)
    agent = MyAgent(game, display=None)
    agent.play()
    scores.append(game.score)
print(scores)
print("Average scores: @50 times", sum(scores) / len(scores))
