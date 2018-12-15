from abc import ABCMeta, abstractmethod
import logging
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import tensorflow as tf

from apfm_feature_attention_model_utils import create_dir


class DataGenerator(metaclass=ABCMeta):
    def __init__(self, para):
        self.directory = para.model_dir + "/data"
        self.h = para.horizon
        self.data_path = os.path.join(self.directory, str(self.h))
        self.para = para
        self.iterator = None
        para.max_len = self.max_len = 16

    def inputs(self, mode, batch_size):
        """Reads input data num_epochs times.
        Args:
        mode: String for the corresponding tfrecords ('train', 'validation', 'test')
        batch_size: Number of examples per returned batch.
        Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, 28, 28]
        in the range [0.0, 1.0].
        * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
        """
        if mode != "train" and mode != "validation" and mode != "test":
            raise ValueError("mode: {} while mode should be ""'train', 'validation', or 'test'".format(mode))

        filename = self.data_path + "/" + mode + ".tfrecords"
        logging.info("Loading data from {}".format(filename))

        with tf.name_scope("input"):
            # TFRecordDataset opens a binary file and
            # reads one record at a time.
            # `filename` could also be a list of filenames,
            # which will be read in order.
            dataset = tf.data.TFRecordDataset(filename)

            # The map transformation takes a function and
            # applies it to every element
            # of the dataset.
            dataset = dataset.map(self._decode)
            for f in self._get_map_functions():
                dataset = dataset.map(f)

            # The shuffle transformation uses a finite-sized buffer to shuffle
            # elements in memory. The parameter is the number of elements in the
            # buffer. For completely uniform shuffling, set the parameter to be
            # the same as the number of elements in the dataset.
            if self.para.mode == "train":
                dataset = dataset.shuffle(1000 + 3 * batch_size)

            # dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(batch_size)

            self.iterator = dataset.make_initializable_iterator()
            return self.iterator.get_next()

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @abstractmethod
    def _decode(self, serialized_example):
        pass

    @abstractmethod
    def _get_map_functions(self):
        pass


class TimeSeriesDataGenerator(DataGenerator):
    def __init__(self, para):
        DataGenerator.__init__(self, para)
        create_dir(self.data_path)
        self.split = [0, 0.8, 0.98, 1]
        self.split_names = ["train", "validation", "test"]
        self._preprocess(para)
        del self.raw_dat, self.dat

    def _preprocess(self, para):

        # self.raw_dat = np.loadtxt(self.out_fn, delimiter=",")
        # start of my code
        df = pd.read_csv('pollution_copy.csv')
        df.fillna(0, inplace=True)
        # One-hot encode 'cbwd'
        # temp = pd.get_dummies(df['wnd_dir'], prefix='wnd_dir')
        # df = pd.concat([df, temp], axis=1)
        # del df['wnd_dir'], temp
        self.raw_dat = df.values.copy()
        label_encoder = LabelEncoder()
        self.raw_dat[:, 4] = label_encoder.fit_transform(self.raw_dat[:, 4])
        # end of my code
        para.input_size = self.INPUT_SIZE = self.raw_dat.shape[1]
        self.rse = self._compute_rse()

        para.max_len = self.MAX_LEN = 10  # self.para.highway
        # assert self.para.highway == self.para.attention_len
        para.output_size = self.OUTPUT_SIZE = self.raw_dat.shape[1]
        para.total_len = self.TOTAL_LEN = 1
        self.dat = np.zeros(self.raw_dat.shape)
        self.scale = np.ones(self.INPUT_SIZE)
        for i in range(self.INPUT_SIZE):
            mn = np.min(self.raw_dat[:, i])
            # if para.data_set == 'electricity':
            self.scale[i] = np.max(self.raw_dat[:, i]) - mn
            # else:
            #     self.scale[i] = np.max(self.raw_dat) - mn
            self.dat[:, i] = (self.raw_dat[:, i] - mn) / self.scale[i]
        logging.info('rse = {}'.format(self.rse))
        for i in range(len(self.split) - 1):
            self._convert_to_tfrecords(self.split[i], self.split[i + 1],
                                       self.split_names[i])

    def _compute_rse(self):
        st = int(self.raw_dat.shape[0] * self.split[2])
        ed = int(self.raw_dat.shape[0] * self.split[3])
        Y = np.zeros((ed - st, self.INPUT_SIZE))
        for target in range(st, ed):
            Y[target - st] = self.raw_dat[target]
        return np.std(Y)

    def _convert_to_tfrecords(self, st, ed, name):
        st = int(self.dat.shape[0] * st)
        ed = int(self.dat.shape[0] * ed)
        out_fn = os.path.join(self.data_path, name + ".tfrecords")
        if os.path.exists(out_fn):
            return
        with tf.python_io.TFRecordWriter(out_fn) as record_writer:
            for target in tqdm(range(st, ed)):
                end = target - self.h + 1
                beg = end - self.para.max_len
                if beg < 0:
                    continue
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "x":
                            self._float_list_feature(self.dat[beg:end].
                                                     flatten()),
                            "y":
                            self._float_list_feature(self.dat[target]),
                        }))
                record_writer.write(example.SerializeToString())

    def _get_map_functions(self):
        return []

    def _decode(self, serialized_example):
        example = tf.parse_single_example(
            serialized_example,
            features={
                "x":
                tf.FixedLenFeature([self.MAX_LEN, self.INPUT_SIZE],
                                   tf.float32),
                "y":
                tf.FixedLenFeature([self.OUTPUT_SIZE], tf.float32),
            },
        )
        rnn_input = tf.to_float(tf.reshape(example["x"], (self.MAX_LEN, self.INPUT_SIZE)))
        rnn_input_len = tf.constant(self.MAX_LEN, dtype=tf.int32)
        target_output = tf.expand_dims(tf.to_float(example["y"]), 0)
        target_output = tf.tile(target_output, [self.MAX_LEN, 1])
        return rnn_input, rnn_input_len, target_output
