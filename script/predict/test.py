import os
from typing import Dict
from numpy.lib.twodim_base import tri, tril
import tensorflow as tf
from tensorflow.python.types.core import Value
import transformers
from transformers import BertTokenizer, TFBertModel, BertConfig
import logging
import collections
import pickle
import numpy as np
from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.text.crf import crf_log_likelihood
from typing import Dict
import math
from focal_loss import sparse_categorical_focal_loss
import json
import re
import random
# random.seed( 10 )
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


######### Config Setting #########
pretrained_path = '/home/gaojun/MacBertLarge'
data_dir = '/home/gaojun/REEXTRACT/PREPROCESS/data'
train_batch_size = 16
num_train_epochs = 3
warmup_proportion = 0.1
TFinput_path = '/home/gaojun/REEXTRACT/test_new_model/TESTSKIMRUN/DATATF'
max_seq_length = 256
middle_output = './'
output_path = './'

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, relation_label=None,subject_label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputTriple(object):
    def __init__(self,guid,triargu):
        self.guid = guid
        self.triargu = triargu

def filed_based_convert_examples_to_features(examples, label_list, triargus,max_seq_length, tokenizer, output_file, mode=None):
    writer = tf.io.TFRecordWriter(output_file)
    ex_index = 0
    for ( example,triargu) in zip(examples,triargus):
        ex_index += 1
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature, ntokens, label_ids,relation_t,relation_l = convert_single_example(ex_index, example, triargu,label_list, max_seq_length, tokenizer,
                                                             mode)


        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))            
            return f



        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
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
    # return batch_tokens,batch_labels

###### Model Definition ######
class TFMyBertModel(tf.keras.models.Model):

    def __init__(self, name: str, pretrain_model: str):
        super().__init__(name)

        self.bert_model = TFBertModel.from_pretrained(pretrain_model)

        self.classifiers = []

        self.classifiers = tf.keras.layers.Dense(11, name='ner_classifiers')

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        ### Metric ###
        self.metric_fn = tf.keras.metrics.Accuracy(name='ner_acc')

        self.relation_metric_fn = tf.keras.metrics.Accuracy(name='relation_acc')

        ### CRF ###
        self.crf = CRF(11,name = 'crf_layer')
        ### Denses ###
        self.linear = tf.keras.layers.Dense(11, activation=None)

        self.linear2 = tf.keras.layers.Dense(256,activation=tf.keras.activations.relu)

        self.softmax_linear = tf.keras.layers.Dense(11, activation=tf.keras.activations.softmax)

        self.linear_relation=tf.keras.layers.Dense(2,activation=tf.keras.activations.softmax)

    def compile(self, optimizer, optimizerner):
         super().compile()
         self.optimizer = optimizer
         self.optimizerner = optimizerner

    def _scale_l2(self,x, norm_length):
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keepdims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keepdims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit

    def train_step(self, data):
        input_mask = data['input_mask']
        label_ids = data['label_ids']
        relation_labels = data['relation_labels']
        with tf.GradientTape(persistent = True) as tape:

            loss,output_layer,predict_relation_loss,predict_relation_proba = self(data, training=True)  # Forward pass

            grad = tape.gradient(loss, output_layer)

            perturb = self._scale_l2(grad, 0.01)

            output_layer = tf.add(output_layer, perturb)

            logits = output_layer

            input_mask = tf.cast(input_mask, tf.float32)

            viterbi_decoded, potentials, sequence_length, chain_kernel = self.crf(
                logits, input_mask)

            loss_with_perturb = -crf_log_likelihood(potentials,
                                           label_ids, sequence_length, chain_kernel)[0]   

            loss_with_perturb = tf.reduce_mean(loss_with_perturb)

            #loss 
            loss_with_perturb += predict_relation_loss*10

        input_mask = tf.cast(input_mask, tf.int32)
        # print(logits.shape,input_mask.shape)
        viterbi_decoded *= input_mask
        
        trainable_vars = self.trainable_variables
        new_gradients = tape.gradient(loss_with_perturb, trainable_vars)

        final_layer_grad = new_gradients[-11:-6]
        final_layer_vars = trainable_vars[-11:-6]
    
        self.optimizerner.apply_gradients(zip(final_layer_grad, final_layer_vars))
        self.optimizer.apply_gradients(zip(new_gradients, trainable_vars))
        
        self.metric_fn.update_state(label_ids,viterbi_decoded)
        self.relation_metric_fn.update_state(relation_labels,predict_relation_proba)
        del tape
        return {'loss: ': loss_with_perturb,'re_loss': predict_relation_loss, ' ner: ': self.metric_fn.result(),' relation: ':self.relation_metric_fn.result()}

    def softmax_layer(self, logits, labels, num_labels, input_mask):
        logits = tf.reshape(logits, [-1, num_labels])
        labels = tf.reshape(labels, [-1])
        input_mask = tf.cast(input_mask, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        loss = tf.losses.categorical_crossentropy(one_hot_labels,logits,from_logits=True)
        loss *= tf.reshape(input_mask, [-1])
        loss = tf.reduce_sum(loss)
        total_size = tf.reduce_sum(input_mask)
        total_size += 1e-12  # to avoid division by 0 for all-0 weights
        loss /= total_size
        # predict not mask we could filtered it in the prediction part.
        probabilities = tf.math.softmax(logits, axis=-1)
        predict = tf.math.argmax(probabilities, axis=-1)
        return loss, predict


    ### Call method ###
    def call(self, inputs, **kwargs):
        features = inputs
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        token_type_ids = features['segment_ids']
        relation_triple = features['relation_triple']
        relation_labels = features['relation_labels']
        outputs = self.bert_model(
            {'input_ids': input_ids,
             'attention_mask': input_mask,
             'token_type_ids': token_type_ids},
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        self.sequence_output = outputs.last_hidden_state

        output_layer = tf.keras.layers.Dropout(rate=0.05)(self.sequence_output)

        ### NER Logits ###
        logits = self.linear(output_layer)
        
        logits = tf.reshape(logits, [-1, max_seq_length, 11])
 
        input_mask = tf.cast(input_mask, tf.float32)

        loss, predict = self.softmax_layer(logits, inputs['label_ids'], 11, input_mask)

        relation_triple = tf.reshape(relation_triple,[train_batch_size,-1,60])

        tri_mask = tf.cast(relation_triple != 0, tf.float32)

        mas_sum0 = tf.reduce_sum(tri_mask, axis=2 )

        is_not_zero = tf.cast(mas_sum0 != 0, tf.float32)

        #if spo mask is 0, add 1
        tri_mask = tf.expand_dims(tri_mask, axis=-1)

        mask_sum = tf.reduce_sum(tri_mask, axis=2 )

        is_zero = tf.cast(mask_sum == 0, tf.float32)

        mask_sum = mask_sum + is_zero

        res = tf.gather(self.sequence_output, relation_triple, axis=1, batch_dims=1)

        res = res * tri_mask        

        res = tf.reduce_sum(res, axis=2) / mask_sum

        predict_relation_proba = self.linear2(res)

        predict_relation_proba = self.linear_relation(predict_relation_proba)

        gold_relation_one_hot = tf.one_hot(relation_labels, depth=2, dtype=tf.float32)

        smoothing = 0.2
        gold_relation_one_hot -= smoothing * (gold_relation_one_hot - 1. / tf.cast(gold_relation_one_hot.shape[-1], gold_relation_one_hot.dtype))

        predict_relation_loss =  tf.nn.softmax_cross_entropy_with_logits(labels=gold_relation_one_hot,logits=predict_relation_proba )

        predict_relation_loss *= is_not_zero

        total_spo_numb = tf.reduce_sum(is_not_zero)
        total_spo_numb += 1e-12 


        predict_relation_loss = tf.reduce_sum(predict_relation_loss)
        predict_relation_loss /= total_spo_numb

        predict_relation_proba = tf.math.argmax(predict_relation_proba, axis=-1)
        
        predict_relation_proba = tf.cast(predict_relation_proba,dtype=tf.float32)
        predict_relation_proba *= is_not_zero
        # tf.print(relation_labels)
        # tf.print(predict_relation_proba)
        
        ### Total Loss ###
        loss += predict_relation_loss  + loss*10

        if 'pooler_output' in outputs:
            self.pooled_output = outputs.pooler_output
        else:
            # no pooled output, use mean of token embedding
            self.pooled_output = tf.reduce_mean(outputs.last_hidden_state, axis=1)

        return loss,output_layer,predict_relation_loss,predict_relation_proba




###### Data Load ######
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
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
    def _create_example(self,lines,set_type):
        examples = []
        for (i,line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputTriple(guid=guid,triargu=line))

        return(examples)


    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_js(os.path.join(data_dir, "train_re_skim.json")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_js(os.path.join(data_dir, "dev_re.json")), "dev"
        )

class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train_skim.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "test"
        )

    def get_labels(self):
        return ["[PAD]", "X", "[CLS]", "[SEP]", "O"]+[j+'-'+i for i in ['属性key','属性value'] for j in ['B','I','E']]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text=line[0], label=line[1]))

        return examples
######### Data PreProcess #########
def convert_single_example(ex_index, example, tria, label_list, max_seq_length, tokenizer, mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]

    """
    label_map = {}
    relation_triple = []
    
    # here start with zero this means that "[PAD]" is zero
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(middle_output + "/label2id.pkl", 'wb') as w:
        pickle.dump(label_map, w)
    textlist = example.text
    labellist = example.label
    trilist = tria.triargu
    
    
    merge_text = ''.join(textlist[1:])



    tokens = []
    labels = []
    relation_labels=[]

    max_tokens = 0
    
    for word in textlist:
        if word == '[SEP]':
            break
        max_tokens +=1



    for i, (word, label) in enumerate(zip(textlist, labellist)):
        if word.startswith('[') and word.endswith(']'):
            tokens.append(word)
            labels.append(label)
            continue
        
        word = word.lower()

        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i, _ in enumerate(token):
            if i == 0:
                labels.append(label)
            else:
                labels.append("X")


        if not token:
            tokens.append('[UNK]')
            labels.append(label)
            # relation_labels.append(int(r))
            # subject_labels.append(int(sr))

    preL = []
    objL = []
    w_trilist = []
    sub_word =''
    if trilist:
        if trilist[0] != 'NULL':
            for item in trilist:
                if item['pre'] not in preL:
                    preL.append(item['pre'])
                if item['obj'] not in objL:
                    objL.append(item['obj'])
                sub_word = item['sub']

            if len(preL) > 1:
                for pre_word in preL:
                    for obj_word in objL:
                        tri_dict  = {"sub":sub_word,"pre":pre_word,"obj":obj_word}
                        if tri_dict not in trilist and tri_dict not in w_trilist:
                            w_trilist.append(tri_dict)

    max_span_size = 20
    if not (trilist):
        relation_triple = []
        relation_labels=[]
    elif trilist[0] == 'NULL':
        relation_triple = []
        relation_labels=[]
    else:
        for item in trilist:
            if len(item['sub']) >= max_span_size or (item['pre'][1] - item['pre'][0]) >= max_span_size or (item['obj'][1] - item['obj'][0]) >= max_span_size or not item['pre'] or not item['obj']  or not item['sub']:
                continue
            re_temp = []
            re_temp_p = []
            re_temp_o = []
            re_temp_s = []

            sub_start = [m.start() for m in re.finditer(item['sub'], merge_text)]
            if len(sub_start) < 2:
                continue


            if len(sub_start) == 2:  #only 2 subjects, use the 1st one
                for ii in range(sub_start[0]+1,sub_start[0]+len(item['sub'])+1):
                    re_temp_s.append(ii)
                    re_temp.append(ii)
                while len(re_temp_s) < max_span_size:
                    re_temp_s.append(0)
                    re_temp.append(0)

            elif(len(sub_start) > 2):               #multi subjects, use the nearest one
                sub_start_re = [ (abs(x-(item['pre'][0]+1)) + abs(x-(item['obj'][0]+1)))  for x in sub_start[:-1]]
                f_sub_start = sub_start_re.index(min(sub_start_re))
                for ii in range(sub_start[f_sub_start]+1,sub_start[f_sub_start]+len(item['sub'])+1):
                    re_temp_s.append(ii)
                    re_temp.append(ii)
                while len(re_temp_s) < max_span_size:
                    re_temp_s.append(0)
                    re_temp.append(0)                


            for ii in range(item['pre'][0]+1,item['pre'][1]+1):
                re_temp_p.append(ii)
                re_temp.append(ii)
            while len(re_temp_p) < max_span_size:
                re_temp_p.append(0)
                re_temp.append(0)
            # print(item['obj'][0],item['obj'][1]-item['obj'][0])
            for ii in range(item['obj'][0]+1,item['obj'][1]+1):
                re_temp_o.append(ii)
                re_temp.append(ii)
            while len(re_temp_o) < max_span_size:
                re_temp_o.append(0)
                re_temp.append(0)

            
            # re_temp.append(item['obj'][0]+1)
            # re_temp.append(item['obj'][1]+1)
            relation_triple.append(re_temp)
            relation_labels.append(1)

        if w_trilist:
            # print(w_trilist)
            for item in w_trilist:
                if len(item['sub']) >= max_span_size or (item['pre'][1] - item['pre'][0]) >= max_span_size or (item['obj'][1] - item['obj'][0]) >= max_span_size or not item['pre'] or not item['obj']  or not item['sub']:
                    continue
                re_temp = []
                re_temp_p = []
                re_temp_o = []
                re_temp_s = []

                sub_start = [m.start() for m in re.finditer(item['sub'], merge_text)]
                if len(sub_start) < 2:
                    continue

                if len(sub_start) == 2:  #only 2 subjects, use the 1st one
                    for ii in range(sub_start[0]+1,sub_start[0]+len(item['sub'])+1):
                        re_temp_s.append(ii)
                        re_temp.append(ii)
                    while len(re_temp_s) < max_span_size:
                        re_temp_s.append(0)
                        re_temp.append(0)

                else:               #multi subjects, use the nearest one
                    sub_start_re = [ (abs(x-(item['pre'][0]+1)) + abs(x-(item['obj'][0]+1)))  for x in  sub_start[:-1]]
                    f_sub_start = sub_start_re.index(min(sub_start_re))
                    for ii in range(sub_start[f_sub_start]+1,sub_start[f_sub_start]+len(item['sub'])+1):
                        re_temp_s.append(ii)
                        re_temp.append(ii)
                    while len(re_temp_s) < max_span_size:
                        re_temp_s.append(0)
                        re_temp.append(0)                


                for ii in range(item['pre'][0]+1,item['pre'][1]+1):
                    re_temp_p.append(ii)
                    re_temp.append(ii)
                while len(re_temp_p) < max_span_size:
                    re_temp_p.append(0)
                    re_temp.append(0)

                for ii in range(item['obj'][0]+1,item['obj'][1]+1):
                    re_temp_o.append(ii)
                    re_temp.append(ii)
                while len(re_temp_o) < max_span_size:
                    re_temp_o.append(0)
                    re_temp.append(0)
                # re_temp.append(item['obj'][0]+1)
                # re_temp.append(item['obj'][1]+1)
                relation_triple.append(re_temp)
                relation_labels.append(0)

        # print(relation_triple)    
        neg_triple = relation_triple
        for ii in range(len(neg_triple)):
            # print(neg)
            neg_length1 = max_span_size - (neg_triple[ii][max_span_size:max_span_size*2].count(0))
            neg_length2 = max_span_size - (neg_triple[ii][max_span_size*2:max_span_size*3].count(0))
            neg_start1 = random.randint(1,max_tokens - neg_length1 - 1)
            neg_start2 = random.randint(1,max_tokens - neg_length2 - 1)
            while neg_start1 ==  neg_triple[ii][2]:
                neg_start1 = random.randint(1,max_tokens - neg_length1 - 1)
            while neg_start2 ==  neg_triple[ii][4]:
                neg_start2 = random.randint(1,max_tokens - neg_length2 - 1)
            

            lux = neg_triple[ii].copy()
            lux2 = neg_triple[ii].copy()
            # print( item['obj'][1] - item['obj'][0],neg_start2,neg_length2,neg_triple[ii][max_span_size*2:max_span_size*3])
            for kk in range(max_span_size,max_span_size+neg_length1):
                lux[kk] = neg_start1 + kk - max_span_size
            for jj in range(max_span_size*2,max_span_size*2+neg_length2):
                lux2[jj] = neg_start2 + jj - max_span_size*2     
            # lux[2] = neg_start1
            # lux[3] = neg_start1 + neg_length1
            # lux2[4] = neg_start2
            # lux2[5] = neg_start2 + neg_length2
            # neg_triple[ii][2] = neg_start
            # neg_triple[ii][3] = neg_triple[ii][2]+ neg_length
            relation_triple.append(lux)
            relation_triple.append(lux2)
            relation_labels.append(0)
            relation_labels.append(0)

            # for i in range(1, (max_tokens - size) + 1): 
            #     if "，" in tokens[i:i+size] or "[UNK]" in tokens[i:i+size] or "。" in tokens[i:i+size] or "," in tokens[i:i+size] or "." in tokens[i:i+size]:
            #         continue
            # print(tokens[i:i+size])

    # max_tokens = len(tokens)
    # print(tokens)
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
        # relation_labels = relation_labels[0:(max_seq_length - 1)]
        # subject_labels=subject_labels[0:(max_seq_length-1)]
    ntokens = []
    segment_ids = []
    label_ids = []
    #ntokens.append("[CLS]")
    #segment_ids.append(0)
    #label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.

    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    # relation_labels.append(0)
    # subject_labels.append(0)


    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    if input_ids.count(102) >= 2:
        for k in range(input_ids.index(102),
            max([index for index in range(len(input_ids)) if input_ids[index] == 102]) + 1):
            segment_ids[k] = 1

    # use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        # relation_labels.append(0)
        # subject_labels.append(0)
        ntokens.append("[PAD]")


    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length - 1] + [input_ids[-1]]
        input_mask = input_mask[:max_seq_length - 1] + [input_mask[-1]]
        segment_ids = segment_ids[:max_seq_length - 1] + [segment_ids[-1]]
        label_ids = label_ids[:max_seq_length - 1] + [label_ids[-1]]
        # relation_labels = relation_labels[:max_seq_length - 1] + [relation_labels[-1]]
        # subject_labels=subject_labels[:max_seq_length-1]+[subject_labels[-1]]
        ntokens = ntokens[:max_seq_length - 1] + [ntokens[-1]]

        print('bomb')

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length

    if ex_index < 3:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # logging.info('relation_labels: %s' % ' '.join([str(x) for x in relation_labels]))
        # logging.info('subject_labels:%s' % ''.join([str(x) for x in subject_labels]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # relation_triple = relation_triple
        # relation_labels=relation_labels,
        # subject_labels=subject_labels
    )
    # we need ntokens because if we do predict it can help us return to original token.

    return feature, ntokens, label_ids,relation_triple,relation_labels

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        # "input_ids": tf.Tensor([train_batch_size, seq_length], tf.int64),
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "relation_triple": tf.io.VarLenFeature(tf.int64),
        "relation_labels": tf.io.VarLenFeature(tf.int64)
        # "relation_triple":
        # 'relation_labels': tf.io.FixedLenFeature([seq_length], tf.int64),
        # 'subject_labels':tf.io.FixedLenFeature([seq_length],tf.int64)

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

    batch_size = train_batch_size
    datare = tf.data.TFRecordDataset(input_file)
    if is_training:
        datare = datare.repeat()
        datare = datare.shuffle(buffer_size=100)

    datare = datare.map(lambda record: _decode_record(record, name_to_features))
    datare = datare.padded_batch(batch_size=batch_size)
    
    return datare


processors = {'ner': NerProcessor, 'triple': TriProcessor, 'brand': NerProcessor}
processor = processors['ner']()
processor_tri = processors['triple']()

label_list = processor.get_labels()

train_examples = None
train_examples = processor.get_train_examples(data_dir)
tri_examples = None
tri_examples = processor_tri.get_train_examples(data_dir)


num_train_steps = None
num_warmup_steps = None
num_train_steps = int(
    len(train_examples) / train_batch_size * num_train_epochs)
print("train_examples = " + str(len(train_examples)))
print("num_train_steps = " + str(num_train_steps))

num_warmup_steps = int(num_train_steps * warmup_proportion)

config_path = os.path.join(pretrained_path, 'config.json')

vocab_path = os.path.join(pretrained_path)

tokenizer = BertTokenizer.from_pretrained(vocab_path)

train_file = os.path.join(TFinput_path, "train.tf_record")



######## TFRecord Writer #########
#filed_based_convert_examples_to_features(
#    train_examples, label_list, tri_examples, max_seq_length, tokenizer, train_file)
#logging.info("***** Running training *****")
#logging.info("  Num examples = %d", len(train_examples))
#logging.info("  Batch size = %d", train_batch_size)
#logging.info("  Num steps = %d", num_train_steps)

######### Inputs #########
train_input = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=max_seq_length,
            is_training=True,
            drop_remainder=True)

optimizer, lr_scheduler = transformers.optimization_tf.create_optimizer(
    init_lr = 5e-5,
    num_train_steps = num_train_steps,
    num_warmup_steps = num_warmup_steps,
)

optimizerner, lr_schedulener = transformers.optimization_tf.create_optimizer(
    init_lr = 3e-3,
    num_train_steps = num_train_steps,
    num_warmup_steps = num_warmup_steps,
)

# print(train_input)
# for ii in train_input:
#     nimg_data = np.array(ii['relation_triple'])
#     xx = nimg_data.reshape(32,-1,8)
#     raise ValueError(nimg_data.reshape(32,-1,8), xx.shape)
config = BertConfig.from_json_file(config_path)

model = TFMyBertModel('my_bert', pretrain_model=pretrained_path)

# ######### Callback#########
ckpath = os.path.join(output_path, 'CK')
outweightpath = os.path.join(output_path,'ckpt')

callbacks =  tf.keras.callbacks.ModelCheckpoint(ckpath, save_weights_only=True,verbose=1)
    # tf.keras.callbacks.TensorBoard(logdir),
model.compile(optimizer=optimizer,optimizerner=optimizerner)

model.fit(x=train_input,epochs=num_train_epochs,steps_per_epoch=(num_train_steps/num_train_epochs),callbacks=callbacks)

model.save_weights(outweightpath)


