from tensorflow.python.training import session_run_hook
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

class Learning_Rate_Hook(session_run_hook.SessionRunHook):
    def __init__(self, optimizer, graph):
        self.optimizer = optimizer
        self.graph = graph
    
    def before_run(self, run_context):
        with self.graph.as_default():
            current_lr = tf.keras.backend.get_value(self.optimizer.lr)
            new_lr = current_lr*0.5
            tf.keras.backend.set_value(self.optimizer.lr, new_lr)
            #current_lr2 = tf.keras.backend.get_value(self.optimizer.lr)
            #logger.info("current learning rate is %f" % current_lr2)
    



def input_fn():
  x = np.random.random((1024, 10))
  y = np.random.randint(2, size=(1024, 1))
  x = tf.cast(x, tf.float32)
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.repeat()
  dataset = dataset.batch(32)
  return dataset

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
optimizer = tf.keras.optimizers.Adam(lr= 0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer)
keras_estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                        config= tf.estimator.RunConfig(save_checkpoints_steps=1000))
training_hook = Learning_Rate_Hook(optimizer, tf.get_default_graph())
Exp = tf.contrib.learn.Experiment(keras_estimator,
                                 train_input_fn= input_fn,
                                 eval_input_fn= input_fn,
                                 train_steps= 20000,
                                 train_monitors=[training_hook],
                                 min_eval_frequency= 1000)
Exp.train_and_evaluate()


