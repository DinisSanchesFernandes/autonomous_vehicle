import tensorflow as tf
import tensorflow.keras.layers as layers


class Value(tf.keras.Model):
    def __init__(self, dim_state, dim_hidden=300, activation=tf.nn.leaky_relu, l2_reg=1e-3):
        super(Value, self).__init__()

        # Store parameters of the neural network
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden

        # Define the neural network hidden and output layer    
        self.value = tf.keras.models.Sequential([
            layers.Dense(self.dim_hidden, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)),
            layers.Dense(self.dim_hidden, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)),
            layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg))
        ])

        # Define the shape of the input layer
        self.value.build(input_shape=(None, self.dim_state))


    def call(self, states, **kwargs):

        # Input the variable states and get the output of the neural network
        value = self.value(states)

        # Return the output of the neural network
        return value