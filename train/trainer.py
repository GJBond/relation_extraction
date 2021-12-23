import transformers
from transformers import BertTokenizer, TFBertModel, BertConfig
import tensorflow as tf
import os
import logging

import sys
sys.path.append('/home/gaojun/git_new_model/odie/script')

from train.data_module import file_based_input_fn_builder,filed_based_convert_examples_to_features,NerProcessor,TriProcessor
from train.config import *
from train.model import *
from train.model_softmax import *


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


num_train_steps = 30
num_warmup_steps = None
#num_train_steps = int(
#    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
print("train_examples = " + str(len(train_examples)))
print("num_train_steps = " + str(num_train_steps))

num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

config_path = os.path.join(PRETRAINED_PATH, 'config.json')

vocab_path = os.path.join(PRETRAINED_PATH)

tokenizer = BertTokenizer.from_pretrained(vocab_path)

train_file = os.path.join(TFINPUT_PATH, "train.tf_record")

######## TFRecord Writer #########
filed_based_convert_examples_to_features(
   train_examples,tri_examples, label_list,  MAX_SEQ_LENGTH, tokenizer, train_file)
logging.info("***** Running training *****")
logging.info("  Num examples = %d", len(train_examples))
logging.info("  Batch size = %d", TRAIN_BATCH_SIZE)
logging.info("  Num steps = %d", num_train_steps)

######### Inputs #########
train_input = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=True)

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

config = BertConfig.from_json_file(config_path)

if USE_CRF:
    model = TFMyBertModel('my_bert', pretrain_model=PRETRAINED_PATH)
    model.compile(optimizer=optimizer,optimizerner=optimizerner)

else:
    model = TFMyBertModel_SM('my_bert', pretrain_model=PRETRAINED_PATH)
    model.compile(optimizer=optimizer)


# ######### Callback#########
ckpath = os.path.join(OUTPUT_PATH, 'CK')
outweightpath = os.path.join(OUTPUT_PATH,'ckpt')

callbacks =  tf.keras.callbacks.ModelCheckpoint(ckpath, save_weights_only=True,verbose=1)
    # tf.keras.callbacks.TensorBoard(logdir),

model.fit(x=train_input,epochs=NUM_TRAIN_EPOCHS,steps_per_epoch=(num_train_steps/NUM_TRAIN_EPOCHS),callbacks=callbacks)

model.save_weights(outweightpath)
