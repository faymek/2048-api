from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent

display1 = Display()
display2 = IPythonDisplay()

for i in range(10):
    game = Game(4, score_to_win=256, random=False)
    display2.display(game)
    agent = ExpectiMaxAgent(game, display=display2)
    agent.play(verbose=True)

