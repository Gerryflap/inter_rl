import threading

import server
from trainers.q_trainer import QTrainer
import keras as ks

model = ks.models.Sequential()
model.add(ks.layers.Dense(100, input_shape=(4,), activation='tanh'))
model.add(ks.layers.Dense(100, activation='tanh'))
model.add(ks.layers.Dense(2, activation='linear'))
model.compile(optimizer=ks.optimizers.Adam(0.0001), loss='mse')

trainer = QTrainer(model, batches_per_update=1, fixed_length=100, gamma=0.99)

app = server.get_server(trainer)

threading.Thread(target=lambda: app.run(port='1337')).start()
trainer.train_loop()
