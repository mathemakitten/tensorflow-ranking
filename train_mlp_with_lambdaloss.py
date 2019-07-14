'''
Train a simple multilayer perception neural neural network with TF-Ranking losses
Evaluate with learning-to-rank metrics such as mean reciprocal rank
'''

from utils import get_logger, get_data_path
import tensorflow_ranking as tfr
import tensorflow as tf
from ast import literal_eval
from datetime import datetime
import pandas as pd
import numpy as np
import time
import os

# tf.enable_eager_execution()
# tf.executing_eagerly()

logger = get_logger('train_model')
Filepath = get_data_path()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

CURRENT_TIME = str(time.time()).replace(".", "")
CURRENT_DATE = str(datetime.now())[:10].replace("-", "")

hyperparams = {'model_name': 'train_15epochs_bettershuffle',
               'num_documents': 25,
               'num_features': 12,#136,#11,
               'batch_size': 1024,
               'hidden_layer_sizes': ["1024", "512", "512", "256", "128"],
               'loss': 'approx_ndcg_loss',#'approx_ndcg_loss' ,#'sigmoid_cross_entropy_loss', #"approx_ndcg_loss",  # tensorflow_ranking.losses - "pairwise_logistic_loss" PAIRWISE_HINGE_LOSS PAIRWISE_SOFT_ZERO_ONE_LOSS
               # SOFTMAX_LOSS SIGMOID_CROSS_ENTROPY_LOSS MEAN_SQUARED_LOSS LIST_MLE_LOSS APPROX_NDCG_LOSS
               'learning_rate': 0.001,
               'num_epochs': 15,
               }

# assume the data is in the LibSVM format and that the content of each file is sorted by query ID.
_TRAIN_DATA_PATH = os.path.join(Filepath.gbm_cache_path, 'TRAIN_normalized.text')
_TEST_DATA_PATH = os.path.join(Filepath.gbm_cache_path, 'VALID_normalized.text')
MODEL_NAME = hyperparams['model_name']
MODEL_DIR = os.path.join(Filepath.model_path, 'mlp_lambdaloss_{}'.format(CURRENT_DATE), '{}_{}'.format(CURRENT_TIME, MODEL_NAME))

#assert os.path.isfile(_TRAIN_DATA_PATH)
#assert os.path.isfile(_TEST_DATA_PATH)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

logger.info("Saving model to: {}".format(MODEL_DIR))
logger.info(hyperparams)

# Hyperparameters
_LIST_SIZE=hyperparams['num_documents']  # list of documents (impressions), max 25 impressions. if impressions < 25, the group gets padded
_NUM_FEATURES=hyperparams['num_features']  # The total number of features per query-document pair.
_BATCH_SIZE=hyperparams['batch_size']
_HIDDEN_LAYER_DIMS=hyperparams['hidden_layer_sizes']
_NUM_EPOCHS = hyperparams['num_epochs']
_TRAINING_SIZE = 960882  # number of training elements (distinct query IDs) in training set
_VALIDATION_SIZE = 54175
n_steps = int((_NUM_EPOCHS * _TRAINING_SIZE) / _BATCH_SIZE)  # (num_epochs x training_size) / batch_size

logger.info("Training for {} steps".format(n_steps))

def input_fn(path, mode):

    # TODO HN? use this generator to get all the entries as np not gen
    train_dataset = tf.data.Dataset.from_generator(tfr.data.libsvm_generator(path, _NUM_FEATURES, _LIST_SIZE),
                                                   output_types=({str(k): tf.float32 for k in range(1,_NUM_FEATURES+1)},tf.float32),
                                                   output_shapes=({str(k): tf.TensorShape([_LIST_SIZE, 1]) for k in range(1,_NUM_FEATURES+1)},
                                                                  tf.TensorShape([_LIST_SIZE]))
                                                   )
    '''
      #Make dataset for training
        dataset_train = tf.data.Dataset.from_tensor_slices((file_ids_training,file_names_training))
        dataset_train = dataset_train.flat_map(lambda file_id,file_name: tf.data.Dataset.from_tensor_slices(
            tuple (tf.py_func(_get_data_for_dataset, [file_id,file_name], [tf.float32,tf.float32]))))
        dataset_train = dataset_train.cache()

        dataset_train= dataset_train.shuffle(buffer_size=train_buffer_size)
        dataset_train= dataset_train.batch(train_batch_size) #Make dataset, shuffle, and create batches
        dataset_train= dataset_train.repeat()
        dataset_train = dataset_train.prefetch(1)
        dataset_train_iterator = dataset_train.make_one_shot_iterator()
        get_train_batch = dataset_train_iterator.get_next()
    '''

    train_dataset = train_dataset.cache()

    if mode == 'TRAIN':
        BATCH_SIZE = _BATCH_SIZE
        train_dataset = train_dataset.shuffle(_TRAINING_SIZE).repeat().batch(BATCH_SIZE) # shuffle the number of elements from this dataset from which the new dataset will sample.
        train_dataset = train_dataset.prefetch(1)
    else:
        BATCH_SIZE = 10000
        train_dataset = train_dataset.repeat(1).batch(BATCH_SIZE)
        train_dataset = train_dataset.prefetch(1)
    return train_dataset.make_one_shot_iterator().get_next()


#TODO HN input for the test set. Note, the test set needs a dummy column for labels at the beginning


def example_feature_columns():
  """Returns the example feature columns."""
  feature_names = ["%d" % (i + 1) for i in range(0, _NUM_FEATURES)]
  return {name: tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0) for name in feature_names}


def make_score_fn():
    """Returns a scoring function to build `EstimatorSpec`."""

    def _score_fn(context_features, group_features, mode, params, config):
        """Defines the network to score a documents."""
        del params
        del config
        # Define input layer.
        example_input = [tf.layers.flatten(group_features[name]) for name in sorted(example_feature_columns())]
        input_layer = tf.concat(example_input, 1)

        cur_layer = input_layer
        for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
            cur_layer = tf.layers.dense(cur_layer, units=layer_width, activation="relu")

        logits = tf.layers.dense(cur_layer, units=1)

        print(logits.shape)
        print(input_layer.shape)

        return logits

    return _score_fn

def eval_metric_fns():
    """Returns a dict from name to metric functions.
    This can be customized as follows. Care must be taken when handling padded lists.

    def _auc(labels, predictions, features):
        is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
        clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
        clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
        return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)

    metric_fns["auc"] = _auc

    Returns: A dict mapping from metric name to a metric function with above signature.
    """

    metric_fns = {}
    metric_fns.update({"metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.NDCG,
                                                                                   topn=topn) for topn in [1, 3, 5, 10, 25]})
    metric_fns.update({"metric/mrr": tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.MRR)})
    return metric_fns


def get_estimator(hparams):
    """Create a ranking estimator.

    Args:
        hparams: (tf.contrib.training.HParams) a hyperparameters object.

    Returns: tf.learn `Estimator`.
    """

    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        return tf.contrib.layers.optimize_loss(loss=loss,
                                               global_step=tf.train.get_global_step(),
                                               learning_rate=hparams.learning_rate,
                                               optimizer="Adam")

    ranking_head = tfr.head.create_ranking_head(loss_fn=tfr.losses.make_loss_fn(hyperparams['loss'],#_LOSS,
                                                                                # TODO HN weights_feature_name for countering position bias
                                                                                ),
                                                eval_metric_fns=eval_metric_fns(),
                                                train_op_fn=_train_op_fn)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    run_config = tf.contrib.learn.RunConfig(session_config=config,
                                            log_step_count_steps=10,
                                            )

    return tf.estimator.Estimator(model_fn=tfr.model.make_groupwise_ranking_fn(group_score_fn=make_score_fn(),
                                                                               group_size=1, #TODO HN what is this???
                                                                               transform_fn=None,
                                                                               ranking_head=ranking_head),
                                  model_dir=MODEL_DIR,
                                  params=hparams,
                                  config=run_config
                                  )

logger.info("Loading data into tf.data.Dataset")

hparams = tf.contrib.training.HParams(learning_rate=hyperparams['learning_rate'])
ranker = get_estimator(hparams)

start_time = time.time()
#logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=10)

#TODO HN change this to train_and_evaluate to avoid calling tf.data again
# Loop over n epochs and output performance per epoch, where num_steps = training_size / batch_size
ranker.train(input_fn=lambda: input_fn(_TRAIN_DATA_PATH, mode='TRAIN'),
             #hooks=
             steps=n_steps)

#TODO HN Inverse Propensity Weighting [40] computed to counter position bias

logger.info("Finished model training")
logger.info(f'Model training took: {(time.time()-start_time)/60:.2f} mins')

logger.info("Evaluating performance on training dataset")
train_metrics = ranker.evaluate(input_fn=lambda: input_fn(_TRAIN_DATA_PATH, mode='VALID'))

logger.info("Evaluating performance on validation dataset")
valid_metrics = ranker.evaluate(input_fn=lambda: input_fn(_TEST_DATA_PATH, mode='VALID'))
logger.info(train_metrics)
logger.info(valid_metrics)

logger.info("TRAINING MRR: {} - VALID MRR: {}".format(train_metrics['metric/mrr'], valid_metrics['metric/mrr']))
logger.info("Completed training and evaluation")
logger.info(f'Total time took: {(time.time()-start_time)/60:.2f} mins')

