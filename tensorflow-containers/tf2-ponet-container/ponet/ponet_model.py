import tensorflow as tf

def ponet_generator():
    def model(inputs, is_training, dim, dropout):
        inputs = tf.keras.layers.Dense(dim)(inputs)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.dropout(inputs, dropout)
        inputs = tf.keras.layers.Dense(int(dim)/2)(inputs)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.dropout(inputs, dropout)
        inputs = tf.keras.layers.Dense(1)(inputs)
        output = tf.nn.sigmoid(inputs)
        return inputs

    return model
