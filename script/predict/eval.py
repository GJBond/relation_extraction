import os
import json
import numpy as np
import argparse

from six import string_types
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-l", "--label", help="label file", dest="argA", type=str, default='/home/gaojun/REEXTRACT/PREPROCESS/data/dev_re_skim.json')
parser.add_argument("-p", "--predict", help="predict file", dest="argB", type=str, default='./PRE.json')
args = parser.parse_args()


label_file = args.argA
predict_file = args.argB

def read_js(input_file):
    with open(input_file,"r",encoding='utf-8') as f0:
        triargu = []
        for line in f0.readlines():
            data = json.loads(line)
            triargu.append(data)
    return triargu

labeled_tri = read_js(label_file)
predict_tri = read_js(predict_file)

total_pre = 0
total_lab = 0
total_score = 0

ii = 0
for r_lable,r_predict in zip(labeled_tri,predict_tri):
    ii +=1
    total_pre += len(r_predict)
    rlist = []
    if r_lable == ['NULL']:
        total_lab = total_lab
    else:
        total_lab += len(r_lable)
        for ritem in r_lable:
            rlist.append([ritem['pre'][0]+1,ritem['pre'][1]+1,ritem['obj'][0]+1,ritem['obj'][1]+1])
    
    for item in r_predict:
        plist = []
        plist = [item['pre'][0],item['pre'][1],item['obj'][0],item['obj'][1]]
        if plist in rlist:
            total_score+=1
        # if plist not in rlist:
        #     print(ii)


precise = total_score/total_pre
recall = total_score/total_lab
    
f1 =2/(1/precise+1/recall)

print("准确率:%f,召回率:%f,f1:%f,count:%d" % (precise,recall,f1,total_lab))

# print(labeled_tri)
# print(predict_tri)



