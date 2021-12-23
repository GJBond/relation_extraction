import json
import random
import os
import re
import hashlib
import zipfile

rows = []

# test_ratio = 0.0
# dev_ratio = 0.10
random.seed(1)

type_list=['','品牌','人名','歌曲名','综艺节目','书名','电影名','电视剧名','游戏名','产品','组织名','功效','成分/原料/材质','气味/味道','品类']

max_len = 250


def md5(str):
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()
# for f in os.listdir('/home/hins/音乐/new_ke'):
#
#     if not f.endswith('.zip'):
#         continue
#     zip_file = zipfile.ZipFile('/home/hins/音乐/new_ke/'+f)
#     zip_list = zip_file.namelist()  # 得到压缩包里所有文件
#
#     for f in zip_list:
#         zip_file.extract(f, '/home/hins/音乐/new_ke/all')  # 循环解压文件到指定目录
#
#     zip_file.close()  # 关闭文件，必须有，释放内存

path='./开放知识提取-已修改20210824/'

file_list = os.listdir(path)

type_set = set()

for filedir in file_list:
    for filedir2 in os.listdir(path + filedir):

        if filedir2.endswith('.json'):
            with open(path+filedir+'/'+filedir2) as f:
                content = f.read()
                rows.append(content)

        else:

            for filedir3 in os.listdir(path+filedir+'/'+filedir2):
                if filedir3.endswith('.json'):
                    with open(path+filedir+'/'+filedir2+'/'+filedir3) as f:
                        content = f.read()
                        rows.append(content)

                else:
                    for filedir4 in os.listdir(path+filedir+'/'+filedir2+'/'+filedir3):
                        if filedir4.endswith('.json'):
                            with open(path+filedir+'/'+filedir2+'/'+filedir3+'/'+filedir4) as f:
                                content=f.read()
                                rows.append(content)

train_text_list = []
test_text_list = []
dev_text_list = []
train_label_list = []
test_label_list = []
train_re = []
dev_re = []
test_re = []
dev_label_list = []

train_relaiton_label_list = []
test_relation_label_list = []
dev_relation_label_list = []

train_subject_label_list = []
test_subject_label_list = []
dev_subject_label_list = []

already_exist_list = []

keyword_set=["人名:何宇轩",'产品:iPhone10','品牌:老强手机',"综艺节目:EXM93",'电视剧名:创新雷达']


for row in rows:
    j = json.loads(row)

    text_list = []
    label_list = []
    re_list = []
    relation_list = []
    subject_list=[]


    keyword = j['content'].split('|*|')[1]

    if not keyword.startswith('活动'):
        keyword_set.append(keyword)

    text = j['content'].split('|*|')[0]
    index = -1

    text_tail_list = ['[SEP]'] + list(keyword)
    label_tail_list = ['O'] + ['O'] * len(keyword)
    relation_tail_list = [0] + [0] * len(keyword)

    keyword = keyword.split(':')[1]

    text_split=re.split("[…|\t|\n|\r|。|；|！|？|;|!|\\?]",text)
    text_split=[t+text[min(len(text)-1,len(t)+sum([len(j) for j in text_split[0:i]])+i)] for i,t in enumerate(text_split)]
    # print(text_split)

    sentence_index=-1
    sentence_start_len=-len(text_split[0])
    sentence_end_len=0

    # if int(j['path'].split('-')[-5][-2:],16)/256<test_ratio:
    #     random_flag='test'
    if md5(j['path'].split('\\')[-3])[-1] in ['1','4','7']:
        random_flag='dev'
    else:
        random_flag='train'

    for _ in range(text.count(keyword)):
        index = text.index(keyword, index + 1)
        # print(index)

        repeat_flag=True
        while index>sentence_end_len and sentence_index<len(text_split):
            sentence_start_len=sentence_end_len
            # print(sentence_start_len)
            sentence_index+=1
            sentence_end_len+=len(text_split[sentence_index])
            repeat_flag=False

        if repeat_flag:
            continue

        tmp_text=text_split[sentence_index]

        if(len(tmp_text)+len(label_tail_list))>max_len:
            tmp_text=tmp_text[:max_len-len(label_tail_list)]
        else:
            additional_sentence_index=sentence_index+1
            while(additional_sentence_index<len(text_split) and len(tmp_text)+len(text_split[additional_sentence_index])+len(label_tail_list)<max_len):
                tmp_text+=text_split[additional_sentence_index]
                additional_sentence_index+=1

            if additional_sentence_index<len(text_split):
                tmp_text+=text_split[additional_sentence_index][:max_len-len(tmp_text)-len(label_tail_list)]

        # print(tmp_text)
        # sentence_start_len = max(index - int(max_len / 3), 0)
        # sentence_end_len = sentence_start_len + max_len -len(label_tail_list)
        # tmp_text = text[sentence_start_len:sentence_end_len]

        label = ['O'] * len(tmp_text)
        relation_label = [0] * len(tmp_text)
        subject_label=[0]*len(tmp_text)
        re_output = []

        if 'annotation' in j['outputs']:

            nodes = j['outputs']['annotation']['T']
        else:
            nodes = []

        # print(j['outputs']['annotation']['R'])
        subjects = [i for i in nodes if i and i['name'] == '指定实体']
        subjects_id_list=[i['id'] for i in subjects]
        

        nodes_with_zhidingshiti=[i for i in nodes if i]

        nodes = [i for i in nodes if i and i['name'] != '指定实体' and i['name']!= '标签' and i['name']!= '别称']

        nodes=[i for i in nodes if i ]
        # if len(nodes)==0:
        #     continue

        for i, node in enumerate(nodes):
            if node['end'] - node['start'] != len(node['value']):
                node['end'] = node['start'] + len(node['value'])

        nodes = sorted(nodes, key=lambda a: a['start'] - a['end'])
        if 'annotation' not in j['outputs']:
            print('error')
            continue
        relations = j['outputs']['annotation']['R']

        relations = [i for i in relations if i]
        


        candi = {}
        candidate = []
        for node in nodes:
            if node == '':
                continue
            if not node:
                continue

            node['start_tmp'] = node['start'] - sentence_start_len
            # print(node['start_tmp'])
            node['end_tmp'] = node['end'] - sentence_start_len
            if node['start_tmp'] < 0 or node['end_tmp'] >= len(tmp_text):
                continue
            subjects_candidate=[r for r in relations if r['from']==node['id'] and r['to'] in subjects_id_list]
            # print(subjects_candidate[0]['to'])

            # 是否因为断句把部分标注内容去掉
            is_cut=False
            if subjects_candidate:
                if len(subjects_candidate)>1:
                    print('subject error')



                subjects_candidate=subjects_candidate[0]
                subject_index_max=max([i['end'] for i in nodes if i['id']==node['id']]+[i['end'] for i in subjects if i['id']==subjects_candidate['to']])-sentence_start_len
                
                subject_index_min=min([i['start'] for i in nodes if i['id']==node['id']]+[i['start'] for i in subjects if i['id']==subjects_candidate['to']])-sentence_start_len
                # print(subject_index_max)

                if subject_index_min>=0 and subject_index_max<len(subject_label):
                    for su in range(subject_index_min,subject_index_max):
                        subject_label[su]=1
                else:
                    is_cut=True

            if not is_cut:
                candidate.append([r for r in relations if r['from']==node['id'] and r['to'] not in subjects_id_list])

                if node['id'] not in candi:

                    candi.update({node['id']:(node['start_tmp'],node['end_tmp'])})


                if label[node['start_tmp']] == 'O' and label[node['end_tmp'] - 1] == 'O':
                    for i in range(node['start_tmp'] + 1, node['end_tmp'] - 1):
                        label[i] = 'I-' + node['name']
                    label[node['end_tmp'] - 1] = 'E-' + node['name']
                    label[node['start_tmp']] = 'B-' + node['name']

                elif label[node['start_tmp']] != 'O' and label[node['end_tmp'] - 1] != 'O':
                    for i in range(node['start_tmp'] + 1, node['end_tmp'] - 1):
                        label[i] = 'I-' + node['name']
                    if label[node['end_tmp'] - 1] == 'O':
                        label[node['end_tmp'] - 1] = 'E-' + node['name']
                    elif label[node['end_tmp'] - 1].startswith('E'):
                        label[node['end_tmp'] - 1] = 'E-' + node['name']
                    elif label[node['end_tmp'] - 1].startswith('I'):
                        if node['name']=='标签':
                            label[node['end_tmp'] - 1] = 'e-' + node['name']
                    elif label[node['end_tmp'] - 1].startswith('e'):
                        if node['name'] == '标签':
                            label[node['end_tmp'] - 1] = 'e-' + node['name']
                    else:
                        print('error')

                    if label[node['start_tmp']] == 'O':
                        label[node['start_tmp']] = 'B-' + node['name']
                    elif label[node['start_tmp']].startswith('I'):
                        if node['name'] == '标签':
                            label[node['start_tmp']] = 'b-' + node['name']
                    elif label[node['start_tmp']].startswith('B'):
                        pass
                    elif label[node['start_tmp']].startswith('b'):
                        pass
                    else:
                        print('error')
                else:
                    print('error')

        if candidate:
            # print("Check ",candidate, candi)
            for kan in candidate:
                for kanl in kan:
                    # print("kan",kanl)
                    if (kanl['to'] in candi) and (kanl['from'] in candi):
                        re_output.append({"sub":keyword,"pre":candi[kanl['to']],"obj":candi[kanl['from']]})
                # print(re_output)
        else:
            re_output.append("NULL")


        for r in relations:
            try:

                alias_nodes = [i for i in nodes_with_zhidingshiti if i['id'] == r['from'] or i['id'] == r['to']]
                if ('别称' in alias_nodes[0]['name'] or '别称' in alias_nodes[1]['name']):
                    # print(alias_nodes)

                    if 'start_tmp' not in alias_nodes[0]:
                        alias_nodes[0]['start_tmp']=alias_nodes[0]['start']-sentence_start_len
                        alias_nodes[0]['end_tmp']=alias_nodes[0]['end']-sentence_start_len

                    if 'start_tmp' not in alias_nodes[1]:
                        alias_nodes[1]['start_tmp']=alias_nodes[1]['start']-sentence_start_len
                        alias_nodes[1]['end_tmp']=alias_nodes[1]['end']-sentence_start_len

                    alias_nodes_end_index=max(alias_nodes[0]['start_tmp'],alias_nodes[1]['start_tmp'])
                    alias_nodes_start_index=min(alias_nodes[0]['end_tmp'],alias_nodes[1]['end_tmp'])

                    # if alias_nodes_start_index>=0 and alias_nodes_end_index<len(tmp_text):
                    #     print(tmp_text[alias_nodes_start_index:alias_nodes_end_index])


                start_index = min([i['start_tmp'] for i in nodes if i['id'] == r['from'] or i['id'] == r['to']])
                end_index = max([i['end_tmp'] for i in nodes if i['id'] == r['from'] or i['id'] == r['to']])
            except Exception as e:
                # print(e)
                continue


            if start_index >= 0 and end_index < len(tmp_text):
                for i in range(start_index, end_index):
                    relation_label[i] = 1

            elif end_index <= 0:
                pass

            elif start_index >= len(tmp_text):
                pass

            else:
                print('relation error')



            type_set.add(node['type'])

        # if len(label) < max_len:
        try:
            type_index=type_list.index("".join(text_tail_list[1:text_tail_list.index(':')]))
        except:
            continue
        # print()
        text_list.append( ['[unused'+str(type_index)+']'] +list(tmp_text) + text_tail_list)
        # print(text_list + label_tail_list)
        label_list.append(['O']+label + label_tail_list)
        # print(label_list)
        relation_list.append([0]+relation_label + relation_tail_list)
        subject_list.append([0]+subject_label+relation_tail_list)
        re_list.append(re_output)
        # print(re_list)

    # print(re_output)
    if random_flag =='test':
        test_label_list.extend(label_list)
        test_text_list.extend(text_list)
        test_relation_label_list.extend(relation_list)
        test_subject_label_list.extend(subject_list)
    elif random_flag =='dev':
        dev_text_list.extend(text_list)
        dev_label_list.extend(label_list)
        dev_relation_label_list.extend(relation_list)
        dev_subject_label_list.extend(subject_list)
        dev_re.extend(re_list)
    else:
        train_text_list.extend(text_list)
        train_label_list.extend(label_list)
        train_relaiton_label_list.extend(relation_list)
        train_subject_label_list.extend(subject_list)
        train_re.extend(re_list)
# print(type_set)

print(len(re_list))
print(len(train_label_list))
print(len(dev_label_list))
print(len(test_label_list))

# print(train_re)

with open('data/train2.txt', 'w') as f, open ('data/train_re2.json','w', encoding="utf-8") as f2:
    z=list(zip(train_text_list, train_label_list, train_relaiton_label_list,train_subject_label_list,train_re))
    random.shuffle(z)
    for i in z:
        text, label, relation,subject,rner = i
        # print(rner)
        # strt = ",".join(rner)
        jsObj = json.dumps(rner,ensure_ascii=False)

        f2.writelines(jsObj)
        f2.write('\n')
        for t, l, r,s in zip(text, label, relation,subject):
            # print(t, l)
            f.write(t + ' ' + l )
            f.write('\n')
        # print()
        f.write('\n')

with open('data/dev2.txt', 'w') as f, open ('data/dev_re2.json','w', encoding="utf-8") as f2:
    for i in zip(dev_text_list, dev_label_list, dev_relation_label_list,dev_subject_label_list,dev_re):
        text, label, relation,subject,rner = i
        jsObj = json.dumps(rner,ensure_ascii=False)
        f2.writelines(jsObj)
        f2.write('\n')
        for t, l, r,s in zip(text, label, relation,subject):
            # print(t, l)
            f.write(t + ' ' + l )
            f.write('\n')
        # print()
        f.write('\n')

with open('data/test.txt', 'w') as f:
    for i in zip(test_text_list, test_label_list, test_relation_label_list,test_subject_label_list):
        text, label, relation,subject = i
        for t, l, r,s in zip(text, label, relation,subject):
            # print(t, l)
            f.write(t + ' ' + l )
            f.write('\n')
        # print()
        f.write('\n')
