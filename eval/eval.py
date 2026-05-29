import json
from evaluate_bleu import evaluate_bleu
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu

import pdb,os
def read_json(json_path):
    json_dict = json.load(open(json_path, encoding='utf-8'))
    return json_dict

def write_json(write_dict, json_save_path,has_indent=False):
    if has_indent:
        json.dump(write_dict, open(json_save_path,'w',encoding='utf-8'),ensure_ascii=False)
    else:
        json.dump(write_dict, open(json_save_path,'w',encoding='utf-8'),indent=3,ensure_ascii=False)
    
    return None

def read_txt(input_path,split_str='\t'):
    list_ = []
    with open(input_path,encoding='utf-8') as var:
        for line in var:
            list_.append(list(line.strip('\n').split(split_str)))
    return list_


def compute_sent_bleu(data_ref, data_sys):
    data_ref_new = [ref.split() for ref in data_ref[0]]
    data_sys_new = data_sys[0].split()
    bleu = sentence_bleu(data_ref_new,data_sys_new)
    return bleu

def compute_corpus_bleu(data_ref, data_sys):
    
    # list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    # hypotheses = [hyp1, hyp2]
    
    data_ref_new = []
    for ref in data_ref:
        new_ref = [r.split() for r in ref]
        data_ref_new.append(new_ref)
        
    data_sys_new = [syss.split() for syss in data_sys]
    # pdb.set_trace()
    bleu = corpus_bleu(data_ref_new,data_sys_new,weights=[0,0,0,1])
    return bleu

def transfer(input_list):
    input_dict = {}
    for data in input_list:
        input_dict[data['input_string']] = data
    return input_dict

def read_txt(input_path,split_str='\t'):
    out_list = []
    with open(input_path,encoding='utf-8') as var:
        for line in var:
            # out_list.append(eval(line.strip('\n')))
            out_list.append(json.loads(line.strip('\n')))
    return out_list

if __name__ == '__main__':
    input_path = '1.jsonl'

    keys = ['Bleu_1','Bleu_2','Bleu_3','Bleu_4','METEOR','ROUGE_L']
    predictions = []
    data_ref = []
    data_list = read_txt(input_path)
    for data in data_list:
        predictions.append(data['Recovered_prompt'])
        data_ref.append([data['instruction']])

    scores = evaluate_bleu(data_ref=data_ref, data_sys=predictions)
    print(scores)


    