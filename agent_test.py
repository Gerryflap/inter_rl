import random
import keras as ks
import util
import numpy as np
import gym


def get_move(state, model, eps):
    if random.random() < eps:
        return random.choice([0, 1])
    else:
        vs = model.predict(np.expand_dims(state, axis=0))[0]
        a = int(np.argmax(vs))
        print(a, vs)
        return a


print("Fetching model")
m = util.get_model(ks)
print("Done.")
eps = 1.0
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
        score += r
        experience.append((s, a, r/10, sp, done))
        s = sp

        if len(experience) > report_every:
            print("Reporting experience: ")
            util.report_experience(experience)
            print("Done")
            print("Fetching model")
            m = util.get_model(ks)
            print("Done")
            experience = []
        eps *= eps_decay
    print("Achieved a score of: \t", score, ", eps: \t", eps)
