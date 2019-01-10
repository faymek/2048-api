import keras
from keras.models import Model
import numpy as np

import random
from collections import namedtuple
from game2048.game import Game
from game2048.expectimax import board_to_move

OUT_SHAPE = (4,4)
CAND = 16
map_table = {2**i : i for i in range(1,CAND)}
map_table[0] = 0
vmap = np.vectorize(lambda x: map_table[x])

def grid_one(arr):
    ret = np.zeros(shape=OUT_SHAPE+(CAND,),dtype=bool)  # shape = (4,4,16)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,arr[r,c]] = 1
    return ret


Guide = namedtuple('Guides', ('state', 'action'))

class Guides:
    
    def __init__(self, cap):
        self.cap = cap
        self.mem = []
        self.pos = 0
        
    def push(self, *args):
        if len(self.mem) < self.cap:
            self.mem.append(None)
        self.mem[self.pos] = Guide(*args)
        self.pos = (self.pos + 1) % self.cap
        
    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)
    
    def ready(self,batch_size):
        return len(self.mem) >= batch_size
    
    def __len__(self):
        return len(self.mem)
    
    
class ModelWrapper:
    
    def __init__(self, model, cap):
        self.model = model
        self.mem = Guides(cap)
        self.trainning_step = 0
        
    def predict(self, board):
        return model.predict(np.expand_dims(board,axis=0))
    
    def move(self, game):
        ohe_board = grid_one(vmap(game.board))        
        self.mem.push(ohe_board, board_to_move(game.board))
        game.move(self.predict(ohe_board).argmax())
        
    def train(self, batch):
        if self.mem.ready(batch):
            guides = self.mem.sample(batch)
            X = []
            Y = []
            for guide in guides:
                X.append(guide.state)
                ohe_action = [0]*4
                ohe_action[guide.action] = 1
                Y.append(ohe_action)
            loss, acc = self.model.train_on_batch(np.array(X), np.array(Y))
            print('#%d \t loss:%.3f \t acc:%.3f'%(self.trainning_step, float(loss), float(acc)))
            self.trainning_step += 1

MEMORY = 32768
BATCH = 1024
ARCHIEVE = 1000

model = keras.models.load_model('modelOL.h5')
mw = ModelWrapper(model,MEMORY)

while True:
    game = Game(4, random=False)
    while not game.end:
        mw.move(game)
    print('score:',game.score, end='\t')
    mw.train(BATCH)
    if(mw.trainning_step%10==0):
        model.save('modelOL.h5')
        if(mw.trainning_step%ARCHIEVE==0):
            model.save('modelOL_%d.h5'%mw.trainning_step)
