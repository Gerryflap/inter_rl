import threading

import server
from trainers.q_trainer import QTrainer
import keras as ks

model = ks.models.Sequential()
model.add(ks.layers.Dense(30, input_shape=(4,), activation='selu'))
model.add(ks.layers.Dense(20, activation='selu'))
model.add(ks.layers.Dense(6, activation='linear'))
model.compile(optimizer=ks.optimizers.Adam(0.0001), loss='mse')

trainer = QTrainer(model, batches_per_update=1, fixed_length=100, gamma=0.95, batch_size=32, replay_size=100000)

app = server.get_server(trainer)

threading.Thread(target=lambda: app.run(host='130.89.170.164', port='1337')).start()
trainer.train_loop()
