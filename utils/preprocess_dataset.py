import logging

from datasets import Dataset

# test
from pprint import pprint

logger = logging.getLogger()


def load_ptb_dataset(filename="./ptb-train.conllu", line_num=None):
    with open(filename, "r") as f:
        lines = f.readlines()

    sentences = []
    tokens = {"idx": [], "form": [], "upos": [], "xpos": [], "head": [], "deprel": []}
    upos_set = set([])
    xpos_set = set([])
    for i, line in enumerate(lines):
        if line.strip() == "":
            if len(tokens) == 0:
                continue

            sentences.append(tokens.copy())
            upos_set.update(tokens["upos"])
            xpos_set.update(tokens["xpos"])
            tokens = {
                "idx": [],
                "form": [],
                "upos": [],
                "xpos": [],
                "head": [],
                "deprel": [],
            }
            continue

        if line_num is not None and i > line_num:
            break

        fields = line.strip().split()
        idx, form, lemma, upos, xpos, feats, head, deprel, deps, misc = fields
        tokens["idx"].append(idx)
        tokens["form"].append(form)
        tokens["upos"].append(upos)
        tokens["xpos"].append(xpos)
        tokens["head"].append(head)
        tokens["deprel"].append(deprel)

    logger.info(f"| Number of sentences: {len(sentences)}")

    return sentences, list(upos_set), list(xpos_set)



def load_shrg_dataset(filename="./cds_mature_few.conll", line_num=None):
    with open(filename, "r") as f:
        lines = f.readlines()

    #如何把两个rule排列组合呢？
    sentences = []
    tokens = {"idx": [], "form": [], "mix_rule":[]}
    rule_set = set([])
    for i, line in enumerate(lines):
        if line.strip() == "":
            if len(tokens) == 0:
                continue

            sentences.append(tokens.copy())
            rule_set.update(tokens["mix_rule"])
            tokens = {
                "idx": [],
                "form": [],
                "mix_rule":[],
            }
            continue

        if line_num is not None and i > line_num:
            break

        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            print(f"Error at line: {fields}")
        idx, form, mix_rule = fields
        tokens["idx"].append(idx)
        tokens["form"].append(form)
        tokens["mix_rule"].append(mix_rule)

    logger.info(f"| Number of sentences: {len(sentences)}")

    return sentences, list(rule_set)


def wrap_dataset(sentences):
    dataset = Dataset.from_list(sentences)
    return dataset


def create_tag_mapping(tags):
    return {tag: i for i, tag in enumerate(sorted(tags))}


def create_obs_mapping(sentences):
    obs = {}
    cnt = 0
    for sentence in sentences:
        tokens = sentence["form"]
        for token in tokens:
            if token not in obs:
                obs[token] = cnt
                cnt += 1
    return obs

if __name__ == "__main__":
    sentences, u, x = load_ptb_dataset()
    s = wrap_dataset(sentences)
    s = create_obs_mapping(sentences)
    u, x = create_tag_mapping(u), create_tag_mapping(x)
    print(s)
