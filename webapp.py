from flask import Flask, jsonify, request

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


def get_flask_app(game, agent):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return app.send_static_file('board.html')

    @app.route("/board", methods=['GET', 'POST'])
    def get_board():
        direction = -1
        control = "USER"
        if request.method == "POST":
            direction = request.json
            if direction == -1:
                direction = agent.step()
                control = 'AGENT'
            game.move(direction)
        return jsonify({"board": game.board.tolist(),
                        "score": game.score,
                        "end": game.end,
                        "direction": direction,
                        "control": control})

    return app


if __name__ == "__main__":
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    APP_PORT = 5005
    APP_HOST = "0.0.0.0"

    from game2048.game import Game
    game = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
    agent = MyAgent(game,display=None)
    print("Run the webapp at http://<any address for your local host>:%s/" % APP_PORT)    
    
    app = get_flask_app(game, agent)
    app.run(port=APP_PORT, threaded=False, host=APP_HOST)  # IMPORTANT: `threaded=False` to ensure correct behavior
    
    
