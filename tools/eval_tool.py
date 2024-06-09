import logging
import os
from threading import local
from typing import List
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from timeit import default_timer as timer
from bmtrain import print_rank
import bmtrain as bmt
from tools import reduce
from kara_storage.pytorch.base import KaraPytorchDatasetBase
logger = logging.getLogger(__name__)
import json
import copy
from utils import normalize_answer, get_max_memory, remove_citations
from eval import *

def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr

def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []

        for qa_pair in item['qa_pairs']:
            # import pdb
            # pdb.set_trace()
            loc_acc.append(exact_presence(qa_pair['short_answers'], item["output"]))
        acc.append(np.mean(loc_acc))
        hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), 100 * np.mean(hit)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config, lr="", otherinfo=""):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 10:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 18:
        s += " "
    s = s + str(step) + " "
    while len(s) < 30:
        s += " "
    s += str(time)
    while len(s) < 50:
        s += " "
    s += str(loss)
    while len(s) < 58:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    s += "\t%s" % lr
    s += "\t%s" % otherinfo
    if not (end is None):
        print_rank(s, end=end)
    else:
        print_rank(s)


def valid(model, dataset, epoch, config, gpu_list, output_function, mode="valid"):
    model.eval()
    local_rank = bmt.rank() #config.getint('distributed', 'local_rank')
    pre_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""

    output_time = config.getint("output", "output_time")
    step = -1

    if hasattr(dataset, "dataset") and isinstance(dataset.dataset, KaraPytorchDatasetBase): 
        dataset.dataset.set_epoch(0)

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])
        if step == total_len - 1:
            results = model(data, config, gpu_list, pre_result, mode=mode, save_score="valid_epoch%d"%(epoch+1))
        else:
            results = model(data, config, gpu_list, pre_result, mode=mode, save_score=None)
        # import pdb
        # pdb.set_trace()
        loss, pre_result = results["loss"], results["pre_result"]
        # print(loss)
        # print(loss, acc_result)
        total_loss += bmt.sum_loss(loss).item()
        cnt += 1
        # if step % output_time == 0:
        #     delta_t = timer() - start_time

            # output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
            #     gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
            #              "%.3lf" % (total_loss / (step + 1)), output_info, None, config)
    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError
  
    if not os.path.exists("result"):
        os.makedirs("result")
    json.dump(pre_result, open("result/" + "plug_question_ctx_projection_30th_train" + ".json", "w"), indent=4)
    data = pre_result

    for i in range(len(data)):
        data[i]['output'] = data[i]['output'].strip().split("\n")[0]
        data[i]['output'] = data[i]['output'].replace("<|im_end|>", "")
    normalized_data = copy.deepcopy(data)
    for i in range(len(normalized_data)):
        normalized_data[i]['output'] = remove_citations(normalized_data[i]['output'])

    result = {}
    result['length'] = compute_len(normalized_data)
    result['str_em'], result['str_hit'] = compute_str_em(normalized_data)

    model.train()
    return result['str_em'], result['str_hit'], result['length']
