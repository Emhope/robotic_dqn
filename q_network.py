import tensorflow as tf
import numpy as np


def create_q_model(
    lidar_frames_num,
    lidar_num,
    num_actions,
    hidden_dences,
    dropout_rate
) -> tf.keras.Model:
    inputs = tf.keras.Input((lidar_frames_num*lidar_num+2+2*(lidar_frames_num), 1))
    flatten_pos = tf.keras.layers.Flatten()(inputs[:, -2-2*(lidar_frames_num):, :])
    convs = []
    for i in range(lidar_frames_num):
        n_input = inputs[:, i*lidar_num: (i+1)*lidar_num, :]
        conv_l = tf.keras.layers.Conv1D(
            3,
            (5,),
            activation='relu',
            padding='valid',
            )(n_input)
        conv_drp = tf.keras.layers.Dropout(dropout_rate)(conv_l)
        pool = tf.keras.layers.MaxPool1D()(conv_drp)
        conv_flatten = tf.keras.layers.Flatten()(pool)
        convs.append(conv_flatten)

    merge_convs = tf.keras.layers.concatenate(convs)
    flatten_convs = tf.keras.layers.Flatten()(merge_convs)
    final_merge = tf.keras.layers.concatenate([flatten_convs, flatten_pos])
    for i in range(hidden_dences):
        if i == 0:
            d = tf.keras.layers.Dense(128, activation='relu')(final_merge)
        else:
            d = tf.keras.layers.Dense(128, activation='relu')(d)
    drop = tf.keras.layers.Dropout(dropout_rate)(d)
    out = tf.keras.layers.Dense(num_actions, activation="linear")(drop)
    m = tf.keras.Model(inputs, out)
    m.compile(
        optimizer="adam",
        loss='mse'
    )
    return m
