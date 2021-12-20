import transformers
from transformers import BertTokenizer, TFBertModel, BertConfig
import tensorflow as tf
import os
import logging
import pickle
import json

import sys
sys.path.append('/home/gaojun/git_new_model/odie/script')

from predict.data_module import file_based_input_fn_builder,filed_based_convert_examples_to_features,NerProcessor,TriProcessor
from predict.config import *
from predict.model import *
from predict.model_softmax import *


os.environ["CUDA_VISIBLE_DEVICES"] = "5"


######## Data Preprocess ########
processors = {'ner': NerProcessor, 'triple': TriProcessor}
processor = processors['ner']()
processor_tri = processors['triple']()

label_list = processor.get_labels()

train_examples = None
train_examples = processor.get_train_examples(DATA_DIR)
tri_examples = None
tri_examples = processor_tri.get_train_examples(DATA_DIR)


num_train_steps = None
num_warmup_steps = None
num_train_steps = int(
    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
print("train_examples = " + str(len(train_examples)))
print("num_train_steps = " + str(num_train_steps))

num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

config_path = os.path.join(PRETRAINED_PATH, 'config.json')

vocab_path = os.path.join(PRETRAINED_PATH)

tokenizer = BertTokenizer.from_pretrained(vocab_path)

train_file = os.path.join(TFINPUT_PATH, "train.tf_record")

######## TFRecord Writer #########
processors = {'ner': NerProcessor, 'triple': TriProcessor, 'brand': NerProcessor}
processor = processors['ner']()
processor_tri = processors['triple']()

label_list = processor.get_labels()

train_examples = None
train_examples = processor.get_train_examples(DATA_DIR)
tri_examples = None
tri_examples = processor_tri.get_train_examples(DATA_DIR)



num_train_steps = None
num_warmup_steps = None
num_train_steps = int(
    len(train_examples) / TRAIN_BATCH_SIZE)
print("train_examples = " + str(len(train_examples)))
print("num_train_steps = " + str(num_train_steps))

num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

config_path = os.path.join(PRETRAINED_PATH, 'config.json')

vocab_path = os.path.join(PRETRAINED_PATH)

tokenizer = BertTokenizer.from_pretrained(vocab_path)

train_file = os.path.join(TFINPUT_PATH, "train.tf_record")



# ######## TFRecord Writer #########
filed_based_convert_examples_to_features(
   train_examples, label_list, tri_examples, MAX_SEQ_LENGTH, tokenizer, train_file)
logging.info("***** Running training *****")
logging.info("  Num examples = %d", len(train_examples))
logging.info("  Batch size = %d", TRAIN_BATCH_SIZE)
logging.info("  Num steps = %d", num_train_steps)

# ######### Inputs #########
train_input = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=True,
            drop_remainder=True)

optimizer, lr_scheduler = transformers.optimization_tf.create_optimizer(
    init_lr = 5e-5,
    num_train_steps = num_train_steps,
    num_warmup_steps = num_warmup_steps,
)


config = BertConfig.from_json_file(config_path)

if USE_CRF:
    model = TFMyBertModel('my_bert', pretrain_model=PRETRAINED_PATH)
else:
    model = TFMyBertModel_SM('my_bert', pretrain_model=PRETRAINED_PATH)

model.compile(run_eagerly=True,optimizer=optimizer)
model.load_weights(MIDDLE_OUTPUT +'/ckpt')


pre_list = []
for batch_data in train_input:
    pre_list.append(model.predict_on_batch(batch_data))

with open("./output_ralation.json", 'w') as wf:
    result = pre_list
    for jj in result:
        for ii in range(len(jj['output'])):
            jsObj = json.dumps(jj['output'][ii],ensure_ascii=False)
            wf.writelines(jsObj)
            wf.write('\n')

with open(MIDDLE_OUTPUT + '/label2id.pkl', 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with open("./output_sequence.txt", 'w') as wf:
    count_all = -1
    for jj in result:
        for ii in range(len(jj['ner'])):
            count_all +=1
            NER = jj['ner'][ii]
            p = []
            for kk in range(len(NER)):
                p.append(NER[kk])

            t = train_examples[count_all].label
            text = train_examples[count_all].text
            for l in zip(text,p,t):
                new_label = id2label[l[1]]
                wf.write(str(l[0]) + "<s>" + new_label)
                wf.write('\n')
            wf.write('\n')
