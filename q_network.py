import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import sys

def build_q_network(input_shape=(75, 75, 4)):
    model = models.Sequential([
        tf.keras.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3),
                      activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3),
                      activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3),
                      activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        # Not having activation function because I want to output real Q values.

        layers.Dense(3, activation=None)  
    ])

    # Compile and return model
    return model


def train_step(model, target_model, batch, optimizer, gamma=0.99):
    states, actions, rewards, next_states, dones = zip(*batch)  
    # Convert to NumPy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    print("State batch shape:", states.shape)
    dones = np.array(dones, dtype=np.float32)
    # TensorFlow training block
    with tf.GradientTape() as tape:
        #  Predict Q-values for current state

        q_values = model(states, training=True)

        # Predict Q-values for next state using target network
        q_next = target_model(next_states)
        max_q_next = tf.reduce_max(q_next, axis=1)

        # Calculate target Q-values
        targets = rewards + (1.0 - dones) * gamma * max_q_next

        # Clipping values so that exploding values are ignored
        # targets = tf.clip_by_value(targets, -10.0, 10.0)
        # Gather Q-values of actions taken
        indices = tf.range(len(actions))
        q_action = tf.gather_nd(q_values, tf.stack([indices, actions], axis=1))


        # Compute loss
        loss_fn = tf.keras.losses.Huber()
        loss = loss_fn(targets, q_action)

    # 6. Apply gradients
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return {"loss" : float(loss.numpy())}
