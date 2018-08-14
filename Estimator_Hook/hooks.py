from tensorflow.python.training import session_run_hook
from tensorflow.python.training.training_util import get_global_step
from tensorflow.python.estimator.keras import _create_keras_model_fn, _save_first_checkpoint
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.model_fn import EstimatorSpec
import numpy as np
import logging

logger = logging.getLogger('tensorflow')
#logger.setLevel(logging.INFO)


def keras_model_to_estimator(model, run_config, training_hooks = None, evaluation_hooks=None):
    keras_weights = model.get_weights()
    keras_model_fn = _create_keras_model_fn(model)
    
    def add_hooks(features, labels, mode):
        spec = keras_model_fn(features, labels, mode)
        
        if training_hooks is not None:
            for hook in training_hooks:
                hook.loss_op = spec.loss
                hook.run_config = run_config
                hook.train_op = spec.train_op
           
        if evaluation_hooks is not None:
            for hook in evaluation_hooks:
                hook.loss_op = spec.loss
                hook.run_config = run_config
                hook.eval_metric_ops = spec.eval_metric_ops
                
        return EstimatorSpec( mode = spec.mode,
                              predictions = spec.predictions,
                              loss = spec.loss,
                              train_op = spec.train_op,
                              eval_metric_ops = spec.eval_metric_ops,
                              export_outputs = spec.export_outputs,
                              training_chief_hooks = spec.training_chief_hooks,
                              training_hooks = training_hooks,
                              scaffold = spec.scaffold,
                              evaluation_hooks = evaluation_hooks)
    
    estimator = Estimator(add_hooks, config = run_config)
    _save_first_checkpoint(model, estimator, custom_objects = None, keras_weights = keras_weights)
    return estimator

class HistoryHook(session_run_hook.SessionRunHook):
    def __init__(self, run_config = None, loss_op = None, eval_metric_ops = None):
        self.run_config = run_config
        self.loss_op = loss_op
        self.eval_metric_ops = eval_metric_ops
        
        self.eval_history = None
        self.current_eval_value = None
        self.current_step = None
        self.eval_step = None
        
    def before_run(self, run_context):
        return session_run_hook.SessionRunArgs(
                {'global_step': get_global_step(),
                 'current_loss': self.loss_op,
                 'eval_value': self.eval_metric_ops}
                )
    
    def after_run(self, run_context, run_values):
        self.current_step = run_values.results['global_step']
        self.current_eval_value = run_values.results['eval_value']
        
    def end(self, session):
        self.update_eval_history()
        for key in self.eval_history.keys():
            logger.info("HistoryHook: steps since the best: " + str(self.number_of_steps_since_best(key)))
    
    def update_eval_history(self):
        if self.eval_history is None:
            self.init_eval_history()
            return
        #update the global step array
        self.eval_step['global_step'] = np.append(self.eval_step['global_step'], self.current_step)
        for key in self.current_eval_value.keys():
            self.eval_history[key] = np.append(self.eval_history[key], self.current_eval_value[key][1])
        
    
    def init_eval_history(self):
        self.eval_history = dict()
        self.eval_step = dict()
        self.eval_step['global_step'] = np.array(self.current_step)
        logger.info('HistoryHook: Init eval history')
        for key in self.current_eval_value.keys():
            self.eval_history[key] = np.array(self.current_eval_value[key][1])
            
    def number_of_steps_since_best(self, metric_key):
        metric_array = self.eval_history[metric_key]
        return metric_array.size - (np.argmin(metric_array) + 1)
            

class EarlyStoppingHook(session_run_hook.SessionRunHook):
    def __init__(self, history_hook, patience, metric_name):
        self.history_hook = history_hook
        self.patience = patience
        self.metric_name = metric_name
    def before_run(self, run_context):
        if self.history_hook.eval_history is not None:
            if self.history_hook.number_of_steps_since_best(self.metric_name) >= self.patience:
                logger.info("EarlyStoppingHooks: %s did not improve for %d rounds, early stopping triggered" % (self.metric_name, self.patience))
                run_context.request_stop()
        pass