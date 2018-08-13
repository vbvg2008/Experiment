from tensorflow.python.training import session_run_hook
from tensorflow.python.training.training_util import get_global_step
import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class HistoryHook(session_run_hook.SessionRunHook):
    def __init__(self, run_config = None, loss_op = None, eval_metric_ops = None):
        self.run_config = run_config
        self.loss_op = loss_op
        self.eval_metric_ops = eval_metric_ops
        
        self.eval_history = None
        self.current_eval_value = None
        self.current_step = None
        
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
        self.eval_history['global_step'] = np.append(self.eval_history['global_step'], self.current_step)
        for key in self.current_eval_value.keys():
            self.eval_history[key] = np.append(self.eval_history[key], self.current_eval_value[key][1])
        
    
    def init_eval_history(self):
        self.eval_history = dict()
        self.eval_history['global_step'] = np.array(self.current_step)
        logger.info('HistoryHook: Init eval history')
        for key in self.current_eval_value.keys():
            self.eval_history[key] = np.array(self.current_eval_value[key][1])
            
    def number_of_steps_since_best(self, metric_key):
        metric_array = self.eval_history[metric_key]
        return metric_array.size - (np.argmin(metric_array) + 1)
            

class EarlyStoppingHook(session_run_hook.SessionRunHook):
    def __init__(self, history_hook, patience, metric_name, metric_best_value):
        self.history_hook = history_hook
        self.patience = patience
        self.metric_name = metric_name
        self.metric_best_value = metric_best_value
    def before_run(self, run_context):
        if self.history_hook.eval_history is not None:
            if self.history_hook.number_of_steps_since_best(self.metric_name) >= self.patience:
                logger.info("EarlyStoppingHooks: %s did not improve for %d rounds, early stopping triggered" % (self.metric_name, self.patience))
                run_context.request_stop()
        pass