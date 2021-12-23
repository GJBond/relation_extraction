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

def convert_spo_gatherdata(sub_start,item,max_span_size):
    """
    convert spo to the form that can be used in tf.gather()
    example with maxspan = 8:
    Args:
        sub_start: list of subject start index
        item: predict and object info
    input: ["sub":(1,4),"pre":(6,7),"obj":(8,11)]
    output:[1,2,3,4,0,0,0,0,6,7,0,0,0,0,0,0,8,9,11,0,0,0,0,0]
    """
    re_temp = []
    re_temp_p = []
    re_temp_o = []
    re_temp_s = []

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
    for ii in range(item['obj'][0]+1,item['obj'][1]+1):
        re_temp_o.append(ii)
        re_temp.append(ii)
    while len(re_temp_o) < max_span_size:
        re_temp_o.append(0)
        re_temp.append(0)

    return(re_temp)




def convert_single_example(ex_index, example, tria, label_list, max_seq_length, tokenizer, mode):
    """
    Data pre-process
    Args:
        ex_index: example num
        example: sequence example
        tria: relation example(triple argument)
        label_list: all labels

    return: feature

    IN this part we should rebuild input sentences to the following format.
    example:[[CLS]木村拓哉的太太工藤静香[SEP]人名:木村拓哉]
    labels: [O,O,O,O,O,O,B-属性,E-属性,B-Value,I-Value,I-Value,E-Value,O,O,O,O,O,O,O,O]
    relation_trip: ["sub":(1,4),"pre":(6,7),"obj":(8,11)]

    """
    label_map = {}
    relation_triple = []
    
    ####################################################
    ###                sequence inputs               ### 

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
    
    # sequence tagging
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

    # sequence padding 
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("[PAD]")


    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length - 1] + [input_ids[-1]]
        input_mask = input_mask[:max_seq_length - 1] + [input_mask[-1]]
        segment_ids = segment_ids[:max_seq_length - 1] + [segment_ids[-1]]
        label_ids = label_ids[:max_seq_length - 1] + [label_ids[-1]]
        ntokens = ntokens[:max_seq_length - 1] + [ntokens[-1]]
        print('bomb')

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length

    ####################################################
    ###                relation inputs               ###                      
    preL = []
    objL = []
    w_trilist = [] # random combination of spo for negative samples
    sub_word =''
    if trilist:
        if trilist[0] != 'NULL':
            for item in trilist:
                if item['pre'] not in preL:
                    preL.append(item['pre'])
                if item['obj'] not in objL:
                    objL.append(item['obj'])
                sub_word = item['sub']
            # negative samples
            if len(preL) > 1:
                for pre_word in preL:
                    for obj_word in objL:
                        tri_dict  = {"sub":sub_word,"pre":pre_word,"obj":obj_word}
                        if tri_dict not in trilist and tri_dict not in w_trilist:
                            w_trilist.append(tri_dict)

    max_span_size = MAX_SPAN_SIZE

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

            # positive samples
            sub_start = [m.start() for m in re.finditer(item['sub'], merge_text)]
            if len(sub_start) < 2:
                continue
            re_temp = convert_spo_gatherdata(sub_start,item,max_span_size)
            relation_triple.append(re_temp)
            relation_labels.append(1)

        #random combination of spo for negative samples
        if w_trilist:
            for item in w_trilist:
                if len(item['sub']) >= max_span_size or (item['pre'][1] - item['pre'][0]) >= max_span_size or (item['obj'][1] - item['obj'][0]) >= max_span_size or not item['pre'] or not item['obj']  or not item['sub']:
                    continue
                re_temp = []


                sub_start = [m.start() for m in re.finditer(item['sub'], merge_text)]
                if len(sub_start) < 2:
                    continue
                
                re_temp = convert_spo_gatherdata(sub_start,item,max_span_size)
                relation_triple.append(re_temp)
                relation_labels.append(0)

        # random negative sampling and padding in the sentence
        neg_triple = relation_triple
        for ii in range(len(neg_triple)):
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

            for kk in range(max_span_size,max_span_size+neg_length1):
                lux[kk] = neg_start1 + kk - max_span_size
            for jj in range(max_span_size*2,max_span_size*2+neg_length2):
                lux2[jj] = neg_start2 + jj - max_span_size*2     

            relation_triple.append(lux)
            relation_triple.append(lux2)
            relation_labels.append(0)
            relation_labels.append(0)


    if ex_index < 3:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )

    return feature,relation_triple,relation_labels


