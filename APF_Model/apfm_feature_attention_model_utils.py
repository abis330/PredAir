from copy import deepcopy
import os
import logging
import tensorflow as tf
import argparse
import json
from apfm_feature_attention_model import Model


def params_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=72)
    parser.add_argument('--filename', type=str, default='traffic')
    parser.add_argument('--decay', type=int, default=0)
    parser.add_argument('--keep_prob', type=float, default=0.75)
    parser.add_argument('--file_output', type=int, default=1)
    # parser.add_argument('--highway', type=int, default=0)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--init_weight', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_gradient_norm', type=float, default=5.0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_dir', type=str, default='./models/checkpoints')
    parser.add_argument('--graph_dir', type=str, default='./models/graphs')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_units', type=int, default=128)

    para = parser.parse_args()

    para.logging_level = logging.INFO

    # creating directory
    try:
        os.makedirs(para.model_dir)
        os.makedirs(para.graph_dir)
    except os.error:
        pass

    json_path = para.model_dir + '/parameters.json'
    # json.dump(vars)
    json.dump(vars(para), open(json_path, 'w'), indent=4)
    return para


def logging_config_setup(para):
    if para.file_output == 0:
        logging.basicConfig(
            level=para.logging_level, format='%(levelname)-8s - %(message)s')
    else:
        logging.basicConfig(
            level=para.logging_level,
            format='%(levelname)-8s - %(message)s',
            filename=para.model_dir + '/progress.txt')
        logging.getLogger().addHandler(logging.StreamHandler())
    tf.logging.set_verbosity(tf.logging.ERROR)


def config_setup():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config


def create_data_generator(para):
    # if para.mts:
    from apfm_feature_attention_data_generator import TimeSeriesDataGenerator
    return TimeSeriesDataGenerator(para)
    # else:
    #     raise ValueError('data_set {} is unknown'.format(para.data_set))


def create_graph(para):
    graph = tf.Graph()
    with graph.as_default():
        initializer = tf.random_uniform_initializer(-para.init_weight,
                                                    para.init_weight)
        data_generator = create_data_generator(para)
        with tf.variable_scope('model', initializer=initializer):
            model = Model(para, data_generator)
    return graph, model, data_generator


def create_valid_graph(para):
    valid_para = deepcopy(para)
    valid_para.mode = 'validation'

    valid_graph, valid_model, valid_data_generator = create_graph(valid_para)
    return valid_para, valid_graph, valid_model, valid_data_generator


def load_weights(para, sess, model):
    ckpt = tf.train.get_checkpoint_state(para.model_dir)
    if ckpt:
        logging.info('Loading model from %s', ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.info('Loading model with fresh parameters')
        sess.run(tf.global_variables_initializer())


def save_model(para, sess, model):
    [global_step] = sess.run([model.global_step])
    checkpoint_path = os.path.join(para.model_dir, "model.ckpt")
    model.saver.save(sess, checkpoint_path, global_step=global_step)


def print_num_of_trainable_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    logging.info('# of trainable parameters: %d' % total_parameters)


def create_dir(path):
    try:
        os.makedirs(path)
    except os.error:
        pass
