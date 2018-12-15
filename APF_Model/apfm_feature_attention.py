import os
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import numpy as np
import apfm_feature_attention_model_utils as utils
import time


def train(para, sess, model, train_data_generator, writer):
    valid_para, valid_graph, valid_model, valid_data_generator = \
        utils.create_valid_graph(para)

    with tf.Session(config=utils.config_setup(), graph=valid_graph) as valid_sess:
        valid_sess.run(tf.global_variables_initializer())
        # code to create writer instance to write validation summary to TensorBoard
        valid_writer = tf.summary.FileWriter(para.graph_dir + '/validation', sess.graph)
        index_training = 0
        index_validation = 0
        for epoch in range(1, para.num_epochs + 1):
            logging.info("Epoch: %d" % epoch)
            sess.run(train_data_generator.iterator.initializer)

            start_time = time.time()
            train_loss = 0.0
            count = 0
            while True:
                try:
                    [loss, global_step, _, training_summary] = sess.run([model.loss, model.global_step,
                                                                         model.update, model.summary_ops])
                    train_loss += loss
                    # writing summary of chosen statistics like training loss, training accuracy to TensorBoard
                    writer.add_summary(training_summary, index_training)
                    index_training += 1
                    count += 1
                except tf.errors.OutOfRangeError:
                    logging.info("global step: %d, loss: %.3f, epoch time: %.3f", global_step, train_loss / count,
                                 time.time() - start_time)
                    utils.save_model(para, sess, model)
                    break

            # validation
            utils.load_weights(valid_para, valid_sess, valid_model)
            valid_sess.run(valid_data_generator.iterator.initializer)
            valid_loss = 0.0
            valid_rse = 0.0
            count = 0
            n_samples = 0
            all_outputs, all_labels = [], []
            while True:
                try:
                    [loss, outputs, labels, valid_summary] = valid_sess.run([valid_model.loss,
                                                                             valid_model.all_rnn_outputs,
                                                                             valid_model.labels,
                                                                             valid_model.summary_ops])
                    valid_rse += np.sum(((outputs - labels) * valid_data_generator.scale) ** 2)
                    # writing summary of chosen statistics like validation loss, validation accuracy to TensorBoard
                    valid_writer.add_summary(valid_summary, index_validation)
                    all_outputs.append(outputs)
                    all_labels.append(labels)
                    n_samples += np.prod(outputs.shape)
                    valid_loss += loss
                    count += 1
                    index_validation += 1
                except tf.errors.OutOfRangeError:
                    break
            all_outputs = np.concatenate(all_outputs)
            all_labels = np.concatenate(all_labels)
            sigma_outputs = all_outputs.std(axis=0)
            sigma_labels = all_labels.std(axis=0)
            mean_outputs = all_outputs.mean(axis=0)
            mean_labels = all_labels.mean(axis=0)
            idx = sigma_labels != 0
            valid_corr = ((all_outputs - mean_outputs) *
                          (all_labels - mean_labels)).mean(axis=0) / (sigma_outputs * sigma_labels)
            valid_corr = valid_corr[idx].mean()
            valid_rse = (np.sqrt(valid_rse / n_samples) / train_data_generator.rse)
            valid_loss /= count
            logging.info("validation loss: %.3f, validation rse: %.3f, validation corr: %.3f",
                         valid_loss, valid_rse, valid_corr)


def test(para, sess, model, test_data_generator):
    sess.run(test_data_generator.iterator.initializer)

    test_rse = 0.0
    count = 0
    n_samples = 0
    all_outputs, all_labels = [], []

    while True:
        try:
            outputs, labels = sess.run([model.all_rnn_outputs, model.labels])
            test_rse += np.sum(((outputs - labels) * test_data_generator.scale) ** 2)
            all_outputs.append(outputs)
            all_labels.append(labels)
            count += 1
            n_samples += np.prod(outputs.shape)
        except tf.errors.UnknownError:
            break
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    sigma_outputs = all_outputs.std(axis=0)
    sigma_labels = all_labels.std(axis=0)
    mean_outputs = all_outputs.mean(axis=0)
    mean_labels = all_labels.mean(axis=0)
    idx = sigma_labels != 0
    test_corr = ((all_outputs - mean_outputs) *
                 (all_labels - mean_labels)).mean(axis=0) / (sigma_outputs * sigma_labels)
    test_corr = test_corr[idx].mean()
    test_rse = np.sqrt(test_rse / n_samples) / test_data_generator.rse
    logging.info("test rse: %.5f, test corr: %.5f" % (test_rse, test_corr))
    # plotting actual vs truth
    predictions = all_outputs[-31 * 24:, 0]  # -31 * 24
    actual = all_labels[-31 * 24:, 0]
    plt.figure()
    plt.plot(predictions, color='orange', label='predicted')
    plt.plot(actual, color='blue', label='actual')
    plt.title("Predicted vs Actual (1 year)")
    plt.legend(loc="upper left")
    plt.savefig("APFM_tf_feature_attention_predicted_vs_actual_epochs_{}.png".format(para.num_epochs))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    params = utils.params_setup()
    utils.logging_config_setup(params)

    graph, apfm_model, data_generator = utils.create_graph(params)
    with tf.Session(config=utils.config_setup(), graph=graph) as session:
        session.run(tf.global_variables_initializer())
        utils.load_weights(params, session, apfm_model)
        utils.print_num_of_trainable_parameters()
        # writer instance to write training summary to TensorBoard
        train_writer = tf.summary.FileWriter(params.graph_dir + '/training', session.graph)
        try:
            if params.mode == 'train':
                train(params, session, apfm_model, data_generator, train_writer)
            elif params.mode == 'test':
                test(params, session, apfm_model, data_generator)

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            print('Finished {}ing!!!'.format(params.mode))
