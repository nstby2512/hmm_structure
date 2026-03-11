import pickle
import json
import re

from pprint import pprint

def transform_both_to_conll(input_path, output_path, rule_path):
    # 匹配 ID 的正则表达式，例如 100/100-0002
    id_pattern = re.compile(r'^[^/]+/(?P<id_part>\d+-\d+)')
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout, \
         open(rule_path, 'rb',) as fr:
        
        lines = fin.readlines()
        derivation = pickle.load(fr)
        abnormal = {}

        #匹配特殊情况('s|)
        contraction_re = re.compile(r"(\w+)('(s|re|ve|ll|d|m|n't))", re.I)
        for i in range(len(lines)):
            line = lines[i].strip()
            
            # 1. 寻找 ID 行
            match = id_pattern.match(line)
            if match:
                short_id = match.group('id_part')
                rule = derivation[short_id]
                # 2. 获取下一行作为句子
                if i + 1 < len(lines):
                    sentence = lines[i+1].strip()
                    
                    # 写入 ID 备注（可选，CoNLL 常用 # id: xxx）
                    # fout.write(f"# id: {short_id}\n")
                    # fout.write(f"# text: {sentence}\n")
                    
                    # 3. 将句子分词并转为 CoNLL 格式(regex处理特殊情况)
                    deal_with_punc = re.sub(r"([,])", r" \1 ", sentence)
                    standardized_sentence = contraction_re.sub(r"\1 \2", deal_with_punc)
                    tokens = standardized_sentence.split()
                    
                    if len(tokens) * 2 - 1 != len(rule):
                        abnormal[short_id] = [f'Token数: {len(tokens)}', f'Rule数: {len(rule)}']
                        # fout.write("!Wrong Mapping!\n")
                        # fout.write("\n")
                        continue

                    for index, token in enumerate(tokens, start=0):
                        # 格式：ID  TOKEN(FORM) 
                        if index == 0:
                            terminal_id, terminal_rule = rule[index][0], rule[index][1:]
                            combination_id, combination_rule = 'None', 'None'
                            mixrule = terminal_rule
                        else:
                            terminal_id, terminal_rule = rule[index*2 - 1][0], rule[index*2 - 1][1:]
                            combination_id, combination_rule = rule[index*2][0], rule[index*2][1:-2]
                            mixrule = terminal_rule + combination_rule
                        fout.write(f"{index + 1}\t{token}\t{terminal_id}\t{terminal_rule}\t{combination_id}\t{combination_rule}\t{mixrule}\n")
                    
                    # 句子之间加空行
                    fout.write("\n")
        print(f"Wrong Mapping Rule:{len(abnormal)}")#7244条异常---2578条---1806条
        


def transform_sentence_to_conll(input_path, output_path):
    # 匹配 ID 的正则表达式，例如 100/100-0002
    id_pattern = re.compile(r'^[^/]+/(?P<id_part>\d+-\d+)')
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        lines = fin.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            
            # 1. 寻找 ID 行
            match = id_pattern.match(line)
            if match:
                short_id = match.group('id_part')
                
                # 2. 获取下一行作为句子
                if i + 1 < len(lines):
                    sentence = lines[i+1].strip()
                    
                    # 写入 ID 备注（可选，CoNLL 常用 # id: xxx）
                    fout.write(f"# id: 100-{short_id}\n")
                    fout.write(f"# text: {sentence}\n")
                    
                    # 3. 将句子分词并转为 CoNLL 格式
                    # 注意：这里简单用 split()，复杂情况建议用 nltk 或 spacy 分词
                    tokens = sentence.split()
                    for index, token in enumerate(tokens, start=1):
                        # 格式：ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
                        # 目前 POS 等位置先用 _ 占位
                        fout.write(f"{index}\t{token}\t_\t_\t_\t_\t_\t_\t_\t_\n")
                    
                    # 句子之间加空行
                    fout.write("\n")

with open('data/childes_by_stage/mature/mature.derivations.p', 'rb') as f:
    data = pickle.load(f)
if __name__ == "__main__":
    # der = data['66-0661']
    # pprint(der)
    transform_both_to_conll('data/childes_by_stage/mature/mature.graphs.txt', 'cds_mature_re2.conll', 'data/childes_by_stage/mature/mature.derivations.p') 
    #transform_sentence_to_conll('data/childes_by_stage/mature/mature.graphs.txt', 'cds_mature_sentence.conll', )
