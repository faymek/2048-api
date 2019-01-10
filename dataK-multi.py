
# coding: utf-8

# In[5]:


from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent


# In[6]:


import numpy as np
import datetime
import csv
import random

map_table = {2**i : i for i in range(1,16)}
map_table[0] = 0

class BigDataAgent(ExpectiMaxAgent):
    def auto_log(self, data_dir="./data/", max_iter=1000, acc=1):
        filename = data_dir + datetime.datetime.now().strftime('%y%m%d_%H%M%S_%f') + ".csv"
        print("文件保存到：",filename)
        acc_th = (4*acc-1)/3  # 模拟当前正确率
        with open(filename,"w") as csvfile: 
            writer = csv.writer(csvfile)
            n_iter = 0
            n_run = 0
            while (n_iter < max_iter):
                if self.game.end:
                    n_run += 1
                    #print("局数：",n_run,"目前数据量：",n_iter)
                    self.game = Game(4, score_to_win=2048, random=False)
                direction = self.step()
                bd = list(self.game.board.flatten())
                bd = [int(s) for s in bd]
                bd = [map_table[i] for i in bd]
                bd.append(direction)
                writer.writerow(bd)
                
                # 模拟当前正确率 0.72 = x + (1-x)/4 => x = 0.63
                if(random.random()>acc_th):
                    direction = random.randrange(4)
                self.game.move(direction)
                n_iter += 1
        #print("数据量：",n_iter)


# In[11]:


THREAD_NUM = 8
LOGFILE_NUM = 10
LOGFILE_STEP = 1000000
AGENT_ACC = 0.6
SCORE = 2048

import os
if not os.path.exists("./multi/"):
    os.mkdir("./multi/")
for i in range(THREAD_NUM):
    path = "./multi/data%d/"%i
    if not os.path.exists(path):
        os.mkdir(path)


# In[12]:


def gen(data_dir = "./multi/data", number = 10):
    game = Game(4, score_to_win=SCORE, random=False)
    agent = BigDataAgent(game, display=None)
    for i in range(number): 
        agent.auto_log(data_dir=data_dir,max_iter=LOGFILE_STEP, acc=AGENT_ACC)


# In[13]:


import _thread
for i in range(THREAD_NUM):
    _thread.start_new_thread( gen, ("./multi/data%d/"%i, LOGFILE_NUM) )


