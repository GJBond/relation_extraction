import tensorflow as tf
import json
import re
import os
import logging
import numpy as np
import collections
from typing import Dict
import sys
from sample import convert_single_example

from config import *

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, relation_label=None,subject_label=None):
        """
        Constructs sequence input example.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class InputTriple(object):
    def __init__(self,guid,triargu):
        """
        Constructs relation input example.
        Args:
            guid: Unique id for the example.
            triargu: predicate and object from input.
        """
        self.guid = guid
        self.triargu = triargu

def filed_based_convert_examples_to_features(examples,triargus,label_list,max_seq_length, tokenizer, output_file, mode=None):
    """
    Codes convert example to features
    Args:
        examples: sequence examples
        triargus: relation examples(triple arguments)
        label_list: label list for sequence("O","SEP","B-属性”......)
        max_seq_length: MAX_SEQ_LENGTH
    """

    writer = tf.io.TFRecordWriter(output_file)

    ex_index = 0

    for (example,triargu) in zip(examples,triargus):
        ex_index += 1
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # data pre-process     
        feature,relation_t,relation_l = convert_single_example(ex_index, example, triargu,label_list, max_seq_length, tokenizer,
                                                             mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))            
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)

        #convert realtion aruments and labels to feature
        relation_flat = np.array(relation_t)
        relation_flat = relation_flat.reshape(-1).tolist()
        re2 = np.array(relation_l)
        re2 = re2.reshape(-1).tolist()
        features["relation_triple"] = create_int_feature(relation_flat)
        features['relation_labels'] = create_int_feature(re2)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    # sentence token in each batch
    writer.close()

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()


    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_txt(cls, input_file):
        """Read a BIO data!"""
        rf = open(input_file, 'r', encoding="utf-8")
        lines = []
        words = []
        labels = []

        for line in rf:
            if len(line.strip()) == 0:
                lines.append((words, labels))
                words = []
                labels = []

            elif len(line.strip().split(' ')) == 1:
                label = line.strip().split(' ')[0]
                word = '\t'
                words.append(word)
                labels.append(label)

            elif len(line.strip().split(' ')) == 2:
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[1]
                words.append(word)
                labels.append(label)

            else:
                print('input format error')

        rf.close()
        return lines


    def _read_js(cls, input_file):
        with open(input_file,"r",encoding='utf-8') as f0:
            triargu = []
            for line in f0.readlines():
                data = json.loads(line)
                triargu.append(data)
        return triargu

class TriProcessor(DataProcessor):
    """
    read file with realtion arguments
    file type "json"
    """
    def _create_example(self,lines,set_type):
        examples = []
        for (i,line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputTriple(guid=guid,triargu=line))

        return(examples)


    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_js(os.path.join(data_dir, RELATION_FILE)), "train"
        )

class NerProcessor(DataProcessor):
    """
    read file with sequence inputs
    file type "txt"
    """
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_txt(os.path.join(data_dir, SEQUENCE_FILE)), "train"
        )

    def get_labels(self):
        return ["[PAD]", "X", "[CLS]", "[SEP]", "O"]+[j+'-'+i for i in ['属性key','属性value'] for j in ['B','I','E']]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text=line[0], label=line[1]))

        return examples


def file_based_input_fn_builder(input_file, seq_length, is_training):
    """
    decode features to train samples
    """

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "relation_triple": tf.io.VarLenFeature(tf.int64),
        "relation_labels": tf.io.VarLenFeature(tf.int64)
    }

    def _decode_record(record, name_to_features) -> Dict[str, tf.Tensor]:
        example = tf.io.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t,tf.int32)
            if name == 'relation_triple' or name == 'relation_labels':
                t = tf.sparse.to_dense(t)
            example[name] = t
        return example

    batch_size = TRAIN_BATCH_SIZE
    datare = tf.data.TFRecordDataset(input_file)
    if is_training:
        datare = datare.repeat()
        datare = datare.shuffle(buffer_size=100)

    datare = datare.map(lambda record: _decode_record(record, name_to_features))
    datare = datare.padded_batch(batch_size=batch_size)
    
    return datare