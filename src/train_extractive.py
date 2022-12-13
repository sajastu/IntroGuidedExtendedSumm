#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
from multiprocessing.pool import Pool

import numpy as np
import os
import pickle
import random
import signal
import time

import torch
from tqdm import tqdm

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import ExtSummarizer
from models.trainer_ext import build_trainer, _mult_top_sents
from others.logging import logger, init_logger
from prepro.data_builder import LongformerData

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


def train_multi_ext(args, is_joint):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue, is_joint,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue, is_joint):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_single_ext(args, device_id, is_joint)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def validate_ext(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test_ext(args, device_id, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test_ext(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = ExtSummarizer(args, device, checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.validate(valid_iter, step)
    return stats.total_loss()

def test_ext_2nd_phase(args):
    _picking_sentences_for_2nd_phase(args)

def _picking_sentences_for_2nd_phase(args):
    preds_sent_numbers = {}
    saved_dict = {}
    paper_sent_scores = []
    logger.info("Picking top sentences for second phase...")
    bert = LongformerData(args)

    saved_dict_ = pickle.load(open(args.saved_list_name, 'rb'))

    for p_idx, (p_id, (p_id, sent_scores, paper_src, paper_tgt, sent_sects_true,
                       sent_sects_whole_true, sent_sections_txt_whole, sent_labels_true,
                       sent_sect_wise_rg, sent_numbers)) in enumerate(saved_dict_.items()):
        paper_sent_true_labels = np.array(sent_labels_true)
        sent_scores = np.array(sent_scores)
        p_src = np.array(paper_src)
        # import pdb;
        # pdb.set_trace()
        p_sent_numbers = np.array(sent_numbers)


        p_sent_sent_sects_true = np.array(sent_sects_true)

        saved_dict[p_id] = (sent_scores, p_id, p_src, p_sent_numbers, p_sent_sent_sects_true)

        keep_ids = [idx for idx, s in enumerate(p_src) if
                    len(s.replace('.', '').replace(',', '').replace('(', '').replace(')', '').
                        replace('-', '').replace(':', '').replace(';', '').replace('*', '').split()) > 5 and
                    len(s.replace('.', '').replace(',', '').replace('(', '').replace(')', '').
                        replace('-', '').replace(':', '').replace(';', '').replace('*', '').split()) < 120
                    ]

        keep_ids = sorted(keep_ids)

        # top_sent_indexes = top_sent_indexes[top_sent_indexes]
        p_src = p_src[keep_ids]
        p_sent_numbers = p_sent_numbers[keep_ids]
        # p_sent_tokens_count = p_sent_tokens_count[keep_ids]
        sent_scores = sent_scores[keep_ids]
        p_sent_sent_sects_true = p_sent_sent_sects_true[keep_ids]
        paper_sent_true_labels = paper_sent_true_labels[keep_ids]
        # sent_true_labels = sent_true_labels[keep_ids]

        # sent_scores = np.asarray([s - 1.00 for s in sent_scores])

        paper_sent_scores.append((sent_scores, p_id, p_src, p_sent_numbers, p_sent_sent_sects_true,
                                  paper_sent_true_labels, bert, "normal"))

    # pickle.dump(paper_sent_scores, open(self.args.saved_list_name.replace('.p', '-sent-scores.p'), "wb"))
    # paper_sent_scores = pickle.load(open(self.args.saved_list_name.replace('.p', '-sent-scores.p'),'rb'))

    overall_recall1, preds_sent_numbers = extract_top_sents(args, paper_sent_scores)

    logger.info("Recall-top section stat: %4.4f" % (overall_recall1))

    # new_sents = []
    # for p in paper_sent_scores:
    #     new_sents.append(p[:-1] + ("normal",))
    # overall_recall2, preds_sent_numbers = self.extract_top_sents(new_sents)
    # logger.info("Recall-top normal: %4.4f" % (overall_recall2))
    #
    # new_sents = []
    # for p in paper_sent_scores:
    #     new_sents.append(p[:-1] + ("section-equal",))
    # overall_recall3, preds_sent_numbers = self.extract_top_sents(new_sents)
    # logger.info("Recall-top section equal: %4.4f" % (overall_recall3))
    #

    # overall_recall2, overall_recall3 = 0, 0
    # stats.set_overall_recall(overall_recall1, overall_recall2, overall_recall3)


    # if write_scores_to_pickle:
    pickle.dump(preds_sent_numbers,
                open(args.saved_list_name.replace('.p', '-top-sents.p'), "wb"))

def extract_top_sents(self, paper_sent_scores):
    preds_sent_numbers = {}
    recalls = []

    # for idx, p in tqdm(enumerate(paper_sent_scores), total=len(paper_sent_scores)):
    #     # if idx>2485:
    #     a, b, c = _mult_top_sents(p, idx)
    #     preds_sent_numbers[a] = b
    #     recalls.append(c)

    pool = Pool(24)
    for d in tqdm(pool.imap_unordered(_mult_top_sents, paper_sent_scores), total=len(paper_sent_scores)):
        preds_sent_numbers[d[0]] = d[1]
        recalls.append(d[2])
    pool.close()
    pool.join()

    recalls = np.array(recalls)
    overall_recall = np.mean(recalls)

    return overall_recall, preds_sent_numbers


def test_ext(args, device_id, pt, step, intro_cls=False, intro_sents_cls=False, intro_top_cls=False, pick_top=False):
    pick_top = args.pick_top

    if pick_top:
        test_ext_2nd_phase(args)

    else:

        device = "cpu" if args.visible_gpus == '-1' else "cuda"
        if (pt != ''):
            test_from = pt
        else:
            test_from = args.test_from
        logger.info('Loading checkpoint from %s' % test_from)
        checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
        print(args)

        def test_iter_fct():
            return data_loader.Dataloader(args, load_dataset(args, args.exp_set, shuffle=False), args.test_batch_size, device,
                                          shuffle=False, is_test=True)

        model = ExtSummarizer(args, device, checkpoint, intro_cls=intro_cls)
        model.eval()

        # test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
        #                                    args.test_batch_size, device,
        #                                    shuffle=False, is_test=True)
        trainer = build_trainer(args, device_id, model, None)
        # trainer.test(test_iter, step)
        # trainer.test(test_iter_fct, step)
        # trainer.validate_rouge_mmr(test_iter_fct, step)
        # trainer.validate_rouge(test_iter_fct, step)
        trainer.validate_rouge_baseline(test_iter_fct, step, write_scores_to_pickle=True)
        # trainer.validate_cls(test_iter_fct, step)

def train_ext(args, device_id, intro_cls=False, intro_sents_cls=False, intro_top_cls=False):
    if (args.world_size > 1):
        train_multi_ext(args, intro_cls)
    else:
        train_single_ext(args, device_id, intro_cls)


def train_single_ext(args, device_id, intro_cls=False, intro_sents_cls=False, intro_top_cls=False):
    init_logger(args.log_file)


    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=False), args.batch_size, device,
                                      shuffle=True, is_test=False)
    def val_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'val', shuffle=False), args.test_batch_size, device,
                                      shuffle=False, is_test=True)
    def test_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False), args.test_batch_size, device,
                                      shuffle=False, is_test=True)

    model = ExtSummarizer(args, device, checkpoint, intro_cls, intro_sents_cls, intro_top_cls)
    optim = model_builder.build_optim(args, model, checkpoint)

    logger.info(model)

    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(train_iter_fct, args.train_steps, valid_iter_fct=val_iter_fct)

    test_ext(args, device_id, args.model_path + '/model_step_' + str(trainer.best_val_step) + '.pt', trainer.best_val_step, is_joint)