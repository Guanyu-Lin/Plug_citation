import json
import os
from torch.utils.data import Dataset
import numpy as np
# from utils import make_demo, make_question, make_ctxs
# from utils import *

from tqdm import tqdm

def get_shorter_text(item, docs, ndoc, key):
    doc_list = []
    for item_id, item in enumerate(docs):
        if key not in item:
            if len(doc_list) == 0:
                # If there aren't any document, at least provide one (using full text)
                item[key] = item['text']
                doc_list.append(item)
            logger.warn(f"No {key} found in document. It could be this data do not contain {key} or previous documents are not relevant. This is document {item_id}. This question will only have {len(doc_list)} documents.")
            break
        if "irrelevant" in item[key] or "Irrelevant" in item[key]:
            continue
        doc_list.append(item)
        if len(doc_list) >= ndoc:
            break
    return doc_list
def make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=None):
    # For doc prompt:
    # - {ID}: doc id (starting from 1)
    # - {T}: title
    # - {P}: text
    # use_shorter: None, "summary", or "extraction"

    text = doc['text']
    if use_shorter is not None:
        text = doc[use_shorter]
    return doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id+1))

def make_demo(item, prompt, ndoc=None, doc_prompt=None, instruction=None, use_shorter=None, test=False):
    # For demo prompt
    # - {INST}: the instruction
    # - {D}: the documents
    # - {Q}: the question
    # - {A}: the answers
    # ndoc: number of documents to put in context
    # use_shorter: None, "summary", or "extraction"

    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "") # if there is no doc we also delete the empty line
        else:
            doc_list = get_shorter_text(item, item["docs"], ndoc, use_shorter) if use_shorter is not None else item["docs"][:ndoc]
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{D}", text)

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip() # remove any space or \n

    return prompt
def make_ctxs(item, prompt, ndoc=None, doc_prompt=None, use_shorter=None):

    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "") # if there is no doc we also delete the empty line
        else:
            doc_list = get_shorter_text(item, item["docs"], ndoc, use_shorter) if use_shorter is not None else item["docs"][:ndoc]
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{D}", text)

    return prompt

def make_question(item, prompt, instruction=None, test=False):

    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip() # remove any space or \n

    return prompt


class ASQADataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.data = []
        # self.ctxnum = 3
        # fin = open(config.get("data", "%s_data_path" % mode), "r")
        self.shot = 2
        self.ndoc = 5
        self.no_doc_in_demo = False
        self.fewer_doc_in_demo = False
        self.ndoc_in_demo = None
        self.use_shorter = None
        self.retrieve_in_all_docs = False
        # if mode == "valid":
        prompt_data_question_ctx = json.load(open(config.get("data", "prompt_question_data_path")))
        # prompt_data_question_ctx = json.load(open(config.get("data", "prompt_question_ctxs_data_path")))
        prompt_data_ctx = json.load(open(config.get("data", "prompt_ctxs_data_path")))

        eval_data = json.load(open(config.get("data", "%s_data_path" % mode)))
        incomplete_doc_list = 0
        # Generate the demonstration part
        head_prompt = ""
        train_ids = np.random.choice(len(prompt_data_question_ctx["demos"]), self.shot, replace=False)
        for train_id in train_ids:
            train_item = prompt_data_question_ctx["demos"][train_id]
            ndoc = self.ndoc
            if self.no_doc_in_demo:
                ndoc = 0
            elif self.fewer_doc_in_demo:
                assert self.ndoc_in_demo is not None
                ndoc = self.ndoc_in_demo
            head_prompt += make_demo(
                train_item, prompt=prompt_data_question_ctx["demo_prompt"], ndoc=ndoc, doc_prompt=prompt_data_ctx["doc_prompt"], 
                instruction=prompt_data_question_ctx["instruction"], use_shorter=self.use_shorter 
            )
            head_prompt += prompt_data_question_ctx["demo_sep"]

        for idx, eval_item in enumerate(tqdm(eval_data)):
           # question = head_prompt + make_demo(
            #    eval_item, prompt=prompt_data_question_ctx["demo_prompt"], ndoc=self.ndoc, doc_prompt=prompt_data_question_ctx["doc_prompt"],
            #    instruction=prompt_data_question_ctx["instruction"], use_shorter=self.use_shorter, 
            #    test=True
            #)
            question = head_prompt + make_question(
                eval_item, prompt=prompt_data_question_ctx["demo_prompt"],
                instruction=prompt_data_question_ctx["instruction"],
                test=True
            )
            ctxs = make_ctxs(
                eval_item, prompt=prompt_data_ctx["demo_prompt"], ndoc=self.ndoc, doc_prompt=prompt_data_ctx["doc_prompt"], use_shorter=self.use_shorter
            )
            answer = "\n" + "\n".join(eval_item["answer"]) if isinstance(eval_item["answer"], list) else eval_item["answer"]
            self.data.append({
                "context": ctxs,
                "question": question,
                "pure_question": eval_item["question"],
                "qa_pairs": eval_item["qa_pairs"],
                "answers": answer
            })

            doc_list = get_shorter_text(eval_item, eval_item["docs"], self.ndoc, self.use_shorter) if self.use_shorter is not None else eval_item["docs"][:self.ndoc]
            if not self.retrieve_in_all_docs:
                # If --retrieve_in_all_docs, we keep the original docs and do not trim them by ndoc
                # Otherwise, take the new docs (truncated by ndoc and filtered if using summary/extraction)
                eval_data[idx]['docs'] = doc_list
            if len(doc_list) < self.ndoc:
                incomplete_doc_list += 1
        print("Done.")
        if incomplete_doc_list > 0:
            print(f"There are {incomplete_doc_list} questions that have incomplete document list (may due to a lot of them are filtered out by summary/extraction).")


    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


import random
class FewOpenQADataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        
        data = []
        self.ctxnum = 3
        fin = open(config.get("data", "%s_data_path" % mode), "r")
        for line in fin:
            line = json.loads(line)
            question = line["input"]
            ctxs = line["output"][0]["provenance"][:self.ctxnum]
            answer = [l["answer"] for l in line["output"][1:]]
            data.append({
                "context": ctxs,
                "question": question,
                "answers": answer
            })

        self.few_num = config.getint("fewshot", "few_num")
        self.seed = config.getint("fewshot", "dataset_seed")
        if mode == "train":
            random.seed(self.seed)
            self.data = random.sample(data, self.few_num)
        else:
            self.data = data

    def __getitem__(self, idx):
        return self.data[idx % len(self.data)]

    def __len__(self):
        return max(200, len(self.data))
