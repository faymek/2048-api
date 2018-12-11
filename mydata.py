# coding: utf-8

# In[1]:


from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent
# from game2048.displayer import Displayer
display1 = Display()
display2 = IPythonDisplay()


# In[2]:


import numpy as np
import datetime
import csv
import math

class DataAgent(ExpectiMaxAgent):
    def auto_log(self, data_dir="./data/", max_iter=np.inf):
        filename = data_dir + datetime.datetime.now().strftime('%y%m%d_%H%M%S_%f') + ".csv"
        print("文件保存到：",filename,end='\t')
        with open(filename,"w") as csvfile: 
            writer = csv.writer(csvfile)
            n_iter = 0
            while (n_iter < max_iter) and (not self.game.end):
                direction = self.step()
                bd = list(self.game.board.flatten())
                bd = [int(s) for s in bd]
                bd = [int(math.log(i,2)) if i>0 else i for i in bd]
                bd.append(direction)
                writer.writerow(bd)
                self.game.move(direction)
                n_iter += 1
        print("数据量：",n_iter)
        


# In[3]:


# 生成数据，一局一个csv文件，无header无index，每行数据为 [log2后的棋盘, 下一步的方向]
for i in range(10):
    game = Game(4, score_to_win=2048, random=False)
    agent = DataAgent(game, display=None)
    print("#%d\t"%i,end=' ')
    agent.auto_log()


# In[ ]:





# In[4]:


#聚合path下所有csv
# 有时候可能目录下会有.ipynb_checkpoints，请手动删除
import os
path = './data/'
files = os.listdir(path)
outputfile = "./" + "all_%d_%s.csv"%(len(files),datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
print("文件保存到：", outputfile)  # 文件默认不保存到data目录

target = open(outputfile,"w")
writer = csv.writer(target)
for file in files: 
    with open(path+file, "r") as f:
        for line in f:
            target.write(line)
target.close()


# In[ ]:




