import random
import numpy as np
from trainers.trainer import Trainer
import util
import json
import keras as ks


class QTrainer(Trainer):
    def __init__(self, model: ks.models.Model, gamma=0.99, fixed_length=1000, batches_per_update=100, batch_size=320,
                 minibatch_size=32, replay_size=10000):
        """
        The QTrainer used to train a model using deepQ learning
        :param replay_size:
        :param model: Keras model
        :param gamma: Discount Factor
        :param fixed_length: Number of updates before the fixed network is updated
        :param batches_per_update: Controls the amount of processed batches until the model params are updated to the outside
        :param batch_size: Size of the total batch (epoch) per update step
        :param minibatch_size: The minibatch size
        """
        self.replay = []

        # Init model
        model.predict(np.zeros((1,) + model.layers[0].input_shape[1:]))
        self.model = model

        self.state_shape = model.layers[0].input_shape[1:]

        # Make fixed model
        self.fixed_model = ks.models.model_from_json(self.model.to_json())
        self.fixed_model.set_weights(self.model.get_weights())

        self.gamma = gamma
        self.fixed_length = fixed_length

        self.model_layout = json.loads(self.model.to_json())
        self.model_weights = self.model.get_weights()
        self.batches_per_update = batches_per_update
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.replay_size = replay_size

    def add_experiences(self, experiences):
        self.replay += experiences
        print("Received experiences, replay size is ", len(self.replay))

    def params_to_json(self):
        out = dict()
        out["layout"] = self.model_layout
        out["weights"] = util.weights_to_base64(self.model_weights)
        return out

    def sample_batch(self):
        batch = []
        for i in range(self.batch_size):
            s, a, r, sp, term = random.choice(self.replay[-self.replay_size:])
            batch.append((s, a, r, sp, term))

        states = np.stack([exp[0] for exp in batch], axis=0)
        targets = self.model.predict(states)

        next_states = np.stack([exp[3] for exp in batch], axis=0)
        vps = self.fixed_model.predict(next_states)

        # Get the max for each next state to get V(sp)
        vps = np.max(vps, axis=1)

        terms = np.stack([exp[4] for exp in batch], axis=0)
        vps[terms] = 0

        # Add rewards
        vps = np.stack([exp[2] for exp in batch], axis=0) + self.gamma * vps

        for i, exp in enumerate(batch):
            # Set the target to the new Q(s, a) value for the taken action index (exp[1])
            targets[i, exp[1]] = vps[i]
        return states, targets

    def train_loop(self):
        """
        The loop that trains the model
        """
        i = 0
        sum_loss = 0

        try:
            while True:
                if len(self.replay) > 0:

                    # Sample new training set
                    X, Y = self.sample_batch()

                    # Fit model and record loss
                    h = self.model.fit(X, Y, verbose=False, batch_size=self.minibatch_size)

                    # Update statistics for loss
                    loss = np.mean(h.history['loss'])
                    sum_loss += loss
                    i += 1

                    if i % self.fixed_length == 0:
                        # Update fixed model
                        self.fixed_model.set_weights(self.model.get_weights())

                    if i % self.batches_per_update == 0:
                        # Display log info
                        vs = self.model.predict(np.expand_dims(self.replay[0][0], axis=0))[0]
                        print("Loss average: ", sum_loss/self.fixed_length, ", initial state values: ", vs)
                        sum_loss = 0

                        # Update model weights
                        self.model_weights = self.model.get_weights()
        except KeyboardInterrupt:

            # Save backup of the model
            with open("quicksave.json", "w") as f:
                json.dump(self.params_to_json(), f)