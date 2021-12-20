import pickle
import logging
import random
import re
import sys
from config import *


__all__ = ['InputFeatures','convert_single_example']



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
    with open(MIDDLE_OUTPUT + "/label2id.pkl", 'wb') as w:
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


    max_span_size = 50

    relation_triple = []
    relation_labels=[]
    tal = ''.join(textlist)
    text = tal.split('[SEP]')[1]
    subj = text.split(':')[1]
    for wb in subj:
        if wb:
            wb = wb.lower()
            relation_triple.append(wb)
    subj_token = tokenizer.convert_tokens_to_ids(relation_triple)
    

    while len(subj_token) < max_span_size:
        subj_token.append(0)
    # print(subj_token)
    assert len(subj_token) == max_span_size

    # print(tokens)
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]

    ntokens = []
    segment_ids = []
    label_ids = []

    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.

    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)

    label_ids.append(label_map["[SEP]"])



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

    return feature, ntokens, label_ids,subj_token


