import random
import time

import requests
import keras as ks
import json
import util
import numpy as np
import gym


def get_model():
    r = requests.get("http://localhost:1337/model")

    m_params = r.json()
    model = ks.models.model_from_json(json.dumps(m_params['layout']))
    model.set_weights(util.base64_to_weights(m_params['weights']))
    return model


def get_move(state, model, eps):
    if random.random() < eps:
        return random.choice([0, 1])
    else:
        return int(np.argmax(model.predict(np.expand_dims(state, axis=0))[0]))



print("Fetching model")
m = get_model()
print("Done.")
eps = 0.0
eps_decay = 0.99995
env = gym.make("CartPole-v0")
report_every = 100
experience = []
while True:
    done = False
    s = env.reset()
    score = 0

    while not done:
        a = get_move(s, m, eps)

        sp, r, done, _ = env.step(a)
        env.render()
        time.sleep(1/60)
        score += r
        experience.append((s, a, r/100, sp, done))
        s = sp

        if len(experience) > report_every:
            print("Fetching model")
            m = get_model()
            print("Done")
            experience = []
        eps *= eps_decay
    print("Achieved a score of: \t", score, ", eps: \t", eps)
