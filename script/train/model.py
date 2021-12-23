import tensorflow as tf
from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.text.crf import crf_log_likelihood
import transformers
from transformers import BertTokenizer, TFBertModel, BertConfig
import sys
# from .data_module import file_based_input_fn_builder,filed_based_convert_examples_to_features
from config import *


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

        self.linear_relation=tf.keras.layers.Dense(2,activation=None)

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
        
        logits = tf.reshape(logits, [-1, MAX_SEQ_LENGTH, 11])
 
        input_mask = tf.cast(input_mask, tf.float32)

        loss, predict = self.softmax_layer(logits, inputs['label_ids'], 11, input_mask)

        relation_triple = tf.reshape(relation_triple,[TRAIN_BATCH_SIZE,-1,60])

        # spo mask, there is 3 status in spo inputs, 1: spo have relation, 0: spo have no relation, null(also stored as 0): spo padding.
        tri_mask = tf.cast(relation_triple != 0, tf.float32)

        mas_sum0 = tf.reduce_sum(tri_mask, axis=2 )

        is_not_zero = tf.cast(mas_sum0 != 0, tf.float32)

        #if spo mask is 0, plus 1
        tri_mask = tf.expand_dims(tri_mask, axis=-1)
        # true spo length
        mask_sum = tf.reduce_sum(tri_mask, axis=2 )

        is_zero = tf.cast(mask_sum == 0, tf.float32)

        # if length = 0, plus 1 to avoid mask_sum = 0
        mask_sum = mask_sum + is_zero

        res = tf.gather(self.sequence_output, relation_triple, axis=1, batch_dims=1)

        res = res * tri_mask        

        res = tf.reduce_sum(res, axis=2) / mask_sum

        predict_relation_proba = self.linear2(res)

        predict_relation_proba = self.linear_relation(predict_relation_proba)

        gold_relation_one_hot = tf.one_hot(relation_labels, depth=2, dtype=tf.float32)

        # label smoothing
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

        
        ### Total Loss ###
        loss += predict_relation_loss  + loss*10

        if 'pooler_output' in outputs:
            self.pooled_output = outputs.pooler_output
        else:
            # no pooled output, use mean of token embedding
            self.pooled_output = tf.reduce_mean(outputs.last_hidden_state, axis=1)

        return loss,output_layer,predict_relation_loss,predict_relation_proba