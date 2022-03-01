import tensorflow as tf 
from tensorflow.keras.layers import Layer  

class CombineCfacLayer(Layer):
   """
   Creates SIMUnet's combination layer.  
   """
   
   def __init__(self, ncfacs):
      # Initialize a Layer instance
      super().__init__()    
      # Create an initializer instance with mean=0. and std=.05 
      init_value = tf.random_normal_initializer() 
      # Initialize vector of trainable weights
      self.w = tf.Variable(
         initial_value=init_value(shape=(ncfacs,), dtype='float32'),
         trainable = True
         )
   
   def __call__(self, inputs, cfactor_values):
      """
      The operation that the SIMUnet layer performs on inputs.
      At this point we only include linear interference with the SMEFT. 
      Parameters
      ----------
         inputs: float 
            Represents the SM (c_factor_values = 0)
            theoretical prediction for a given observable. 
         cfactor_values: tf.Variable
            Set of trainable SMEFT C-factors that affect a given
            observable. 

      """
      return (1 + tf.reduce_sum(w[:, tf.newaxis] * cfactor_values, axis=0)) * inputs

                              




