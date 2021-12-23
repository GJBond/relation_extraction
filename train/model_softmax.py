import tensorflow as tf
from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.text.crf import crf_log_likelihood
import transformers
from transformers import BertTokenizer, TFBertModel, BertConfig
import sys
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

            gold_ner_one_hot = tf.one_hot(label_ids, depth=11, dtype=tf.float32)

            logits = self.linear(output_layer)

            # logits = self.softmax_linear(output_layer)

            # loss padding 
            input_mask = tf.cast(input_mask, tf.float32)
            loss_with_perturb =  tf.nn.softmax_cross_entropy_with_logits(labels=gold_ner_one_hot,logits=logits)
            # loss_with_perturb = sparse_categorical_focal_loss(label_ids,logits,gamma=2)

            loss_with_perturb = input_mask*loss_with_perturb 

            # loss weight
            loss_with_perturb = tf.reduce_sum(loss_with_perturb,-1)

            total_size = tf.reduce_sum(input_mask)
            total_size += 1e-12 
            loss_with_perturb = loss_with_perturb/total_size * 100

            loss_with_perturb += predict_relation_loss

        logits = tf.argmax(logits, axis=-1)

        input_mask = tf.cast(input_mask, tf.int64)
        logits *= input_mask

        trainable_vars = self.trainable_variables

        new_gradients = tape.gradient(loss_with_perturb, trainable_vars)

        self.optimizer.apply_gradients(zip(new_gradients, trainable_vars))

        self.metric_fn.update_state(label_ids,logits)
        self.relation_metric_fn.update_state(relation_labels,predict_relation_proba)
        del tape
        return {'loss: ': loss_with_perturb, ' ner: ': self.metric_fn.result(),' relation: ':self.relation_metric_fn.result()}


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
        loss += predict_relation_loss  + loss*100

        if 'pooler_output' in outputs:
            self.pooled_output = outputs.pooler_output
        else:
            # no pooled output, use mean of token embedding
            self.pooled_output = tf.reduce_mean(outputs.last_hidden_state, axis=1)

        return loss,output_layer,predict_relation_loss,predict_relation_proba

