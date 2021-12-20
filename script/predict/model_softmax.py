import tensorflow as tf
from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.text.crf import crf_log_likelihood
import transformers
from transformers import BertTokenizer, TFBertModel, BertConfig
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from typing import Dict
import sys
# from .data_module import file_based_input_fn_builder,filed_based_convert_examples_to_features
from config import *


class TFMyBertModel_SM(tf.keras.models.Model):

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

        ### Denses ###
        self.linear = tf.keras.layers.Dense(11, activation=None)

        self.linear2 = tf.keras.layers.Dense(256,activation=tf.keras.activations.relu)

        self.softmax_linear = tf.keras.layers.Dense(11, activation=tf.keras.activations.softmax)

        self.linear_relation=tf.keras.layers.Dense(2,activation=None)
    
    def get_index(self, lst=None, item=''):
        return [index for (index,value) in enumerate(lst) if value == item]

    def convert_spo_gatherdata(self,sublist,item,max_span_size,len_sub):
        re_temp = []
        re_temp_p = []
        re_temp_o = []
        re_temp_s = []
        if len(sublist) == 2:  #only 2 subjects, use the 1st one
            for ii in range(sublist[0],sublist[0]+len_sub):
                re_temp_s.append(ii)
                re_temp.append(ii)
            while len(re_temp_s) < max_span_size:
                re_temp_s.append(0)
                re_temp.append(0)
        else:
            sub_start_re = [ (abs(x-(item['pre'][0])) + abs(x-(item['obj'][0])))  for x in sublist[:-1]]
            f_sub_start = sub_start_re.index(min(sub_start_re))
            for ii in range(sublist[f_sub_start],sublist[f_sub_start]+len_sub):
                re_temp_s.append(ii)
                re_temp.append(ii)
            while len(re_temp_s) < max_span_size:
                re_temp_s.append(0)
                re_temp.append(0)
                
        for ii in range(item['pre'][0],item['pre'][1]):
            re_temp_p.append(ii)
            re_temp.append(ii)
        while len(re_temp_p) < max_span_size:
            re_temp_p.append(0)
            re_temp.append(0)
        for ii in range(item['obj'][0],item['obj'][1]):
            re_temp_o.append(ii)
            re_temp.append(ii)
        while len(re_temp_o) < max_span_size:
            re_temp_o.append(0)
            re_temp.append(0)

        return(re_temp)


    ### Call method ###
    def call(self, inputs, **kwargs):
        features = inputs
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        token_type_ids = features['segment_ids']
        subj_token = features['subj_token']
        outputs = self.bert_model(
            {'input_ids': input_ids,
             'attention_mask': input_mask,
             'token_type_ids': token_type_ids},
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        self.sequence_output = outputs.last_hidden_state
        output_layer = self.sequence_output

        logits = self.linear(output_layer)
        # output_layer = tf.keras.layers.Dropout(rate=0.05)(self.sequence_output)

        ner_proba =   tf.nn.softmax(logits,axis=-1)
        ner_predict = tf.argmax(ner_proba, axis=-1)
        ner_predict = tf.cast(ner_predict, tf.int32)


        ner_input = []
        max_span_size = 20
        max_num_spo = 0
        all_trilist = []
        for pos,sub,token in zip(ner_predict,subj_token,input_ids):
            pos = pos.numpy().tolist()
            sub = sub.numpy().tolist()
            token = token.numpy().tolist()
            f_pre_start = self.get_index(pos,5)
            f_obj_start = self.get_index(pos,8)
            f_sub_start = self.get_index(token,sub[0])


            trilist = []

            for ii in f_pre_start:
                len_pre = 0
                if ii >= len(pos):
                    continue
                while pos[ii+len_pre] == 6 or pos[ii+len_pre] == 7 or pos[ii+len_pre] == 5:

                    len_pre = len_pre + 1
                    if ii + len_pre >= len(pos):
                        break
                f_pre_end = ii + len_pre
                for jj in f_obj_start:
                    len_obj = 0
                    if jj >= len(pos):
                        continue
                    while pos[jj+len_obj] == 9 or pos[jj+len_obj] == 10 or pos[jj+len_obj] == 8:
                        len_obj = len_obj + 1
                        if jj + len_obj >= len(pos):
                            break
                    f_obj_end = jj + len_obj
                    trilist.append({"pre":(ii,f_pre_end),"obj":(jj,f_obj_end)})
            all_trilist.append(trilist)
            sublist = [] 
            for ss in f_sub_start:
                len_sub = 0
                is_sub = 1
                while sub[len_sub] !=0:
                    if ss+len_sub >= len(token):
                        break
                    if token[ss+len_sub] != sub[len_sub]:
                        is_sub = 0
                    len_sub +=1
                if is_sub:
                    sublist.append(ss)


            relation_input = []
            if not trilist:
                relation_input = []
            else:
                for item in trilist:
                    if (item['pre'][1] - item['pre'][0]) >= max_span_size or (item['obj'][1] - item['obj'][0]) >= max_span_size or not item['pre'] or not item['obj'] :
                        continue
                    re_temp = []

                    if len(sublist) < 2:
                        continue

                    re_temp = self.convert_spo_gatherdata(sublist,item,max_span_size,len_sub)
            
                    relation_input.append(re_temp)
            if len(relation_input) > max_num_spo:
                max_num_spo = len(relation_input)
            relation_input = np.array(relation_input)
            relation_input = relation_input.reshape(-1).tolist()
            ner_input.append(relation_input)
       
        ner_input = pad_sequences(ner_input,maxlen=max_span_size*max_num_spo*3, padding='post',value=0)

        true_batch_size = tf.shape(output_layer)[0]
        ner_input = tf.reshape(ner_input,[true_batch_size,-1,60])

        # ner_input = tf.reshape(ner_input,[train_batch_size,-1,60])

        tri_mask = tf.cast(ner_input != 0, tf.float32)

        mas_sum0 = tf.reduce_sum(tri_mask, axis=2 )

        is_not_zero = tf.cast(mas_sum0 != 0, tf.float32)

        tri_mask = tf.expand_dims(tri_mask, axis=-1)

        mask_sum = tf.reduce_sum(tri_mask, axis=2 )

        is_zero = tf.cast(mask_sum == 0, tf.float32)

        mask_sum = mask_sum + is_zero
        
        res = tf.gather(self.sequence_output, ner_input, axis=1, batch_dims=1)

        res = res * tri_mask

        res = tf.reduce_sum(res, axis=2) / mask_sum

        predict_relation_proba = self.linear2(res)

        predict_relation_proba = self.linear_relation(predict_relation_proba)

        predict_relation_proba = tf.math.argmax(predict_relation_proba, axis=-1)

        predict_relation_proba = tf.cast(predict_relation_proba,dtype=tf.float32)

        predict_relation_proba *= is_not_zero

        predict_relation_proba_int = tf.cast(predict_relation_proba, tf.int32)


        output = []
        for pro,tri,prob in zip(predict_relation_proba_int,all_trilist,predict_relation_proba):
            elist = []
            check_overlap = []
            pro_overlap = []
            pro = pro.numpy().tolist()
            prob = prob.numpy().tolist()
            for ii in range(len(tri)):
                if ii >= len(pro):
                    continue
                if pro[ii]:
                    if tri[ii]['obj'] not in check_overlap:
                        check_overlap.append(tri[ii]['obj'])
                        pro_overlap.append(prob[ii])
                        elist.append(tri[ii])
                    else:
                        index_id = check_overlap.index(tri[ii]['obj'])
                        if prob[ii] > pro_overlap[index_id]:
                            check_overlap[index_id] = tri[ii]['obj']
                            pro_overlap[index_id] = prob[ii]
                            elist[index_id] = tri[ii]

                    # elist.append(tri[ii])
            output.append(elist)

        # print(output)
        res_dict : Dict[str, tf.Tensor] = {}
        res_dict['ner'] = ner_predict
        res_dict['output'] = output
        return res_dict