import collections
import glob
import json
import os
import pickle
# from multiprocessing.pool import Pool
import traceback

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pool
from tqdm import tqdm

import distributed
from models.reporter_ext import ReportMgr, Statistics
from others.logging import logger
from prepro.data_builder import LongformerData
# from utils.rouge_pap import get_rouge_pap
from utils.rouge_score import evaluate_rouge
from utils.spreadsheet import update_rouge_score_drive, update_recall_drive, update_step


def _multi_rg(params):
    return evaluate_rouge([params[0]], [params[1]])


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path + '/stats/'

    # if not os.path.exists(tensorboard_log_dir) :
    #     os.makedirs(tensorboard_log_dir)
    # else:
    #     os.remove(tensorboard_log_dir)
    #     os.makedirs(tensorboard_log_dir)

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.
    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, args, model, optim,
                 grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 report_manager=None):
        # Basic attributes.
        self.args = args
        self.alpha = args.alpha_mtl
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.intro_cls = getattr(self.model, 'intro_cls')
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.valid_trajectories = []
        self.valid_rgls = []
        self.overall_recalls = []
        self.best_val_step = 0
        # self.loss = torch.nn.BCELoss(reduction='none')
        self.rg_predictor = False

        self.rmse_loss = torch.nn.MSELoss(reduction='none')
        # self.loss_sect = torch.nn.CrossEntropyLoss(reduction='none')

        self.min_val_loss = 100000
        self.min_rl = -100000
        self.overall_recall = -100000
        self.softmax = nn.Softmax(dim=1)
        self.softmax_acc_pred = nn.Softmax(dim=2)
        self.softmax_sent = nn.Softmax(dim=1)
        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

        self.bert = LongformerData(args)

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`
        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):
        Return:
            None
        """
        logger.info('Start training...')

        # step =  self.optim._step + 1
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        sent_num_normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        valid_global_stats = Statistics(stat_file_dir=self.args.model_path)
        valid_global_stats.write_stat_header(self.intro_cls)
        # report_stats = Statistics(print_traj=self.is_joint)
        report_stats = Statistics(print_traj=self.intro_cls)
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        # if step % 120 == 0:
                        #     for name, param in self.model.named_parameters():
                        #         if name == 'sentence_encoder.ext_transformer_layer.transformer_inter.0.self_attn.linear_keys.weight':
                        #             print(param)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,report_stats
                            # self.model.uncertainty_loss._sigmas_sq[0].item() if self.intro_cls else 0,
                            # self.model.uncertainty_loss._sigmas_sq[1].item() if self.intro_cls else 0,
                            )

                        self._report_step(self.optim.learning_rate, step,
                                          # self.model.uncertainty_loss._sigmas_sq[0] if self.intro_cls else 0,
                                          # self.model.uncertainty_loss._sigmas_sq[1] if self.intro_cls else 0,
                                          train_stats=report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0

                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            val_stat, best_model_save, best_recall_model_save = self.validate_rouge_baseline(
                                valid_iter_fct, step,
                                valid_gl_stats=valid_global_stats)
                            self._save(step, best=best_model_save, recall_model=best_recall_model_save,
                                       valstat=val_stat)

                        # if step == 100 or step % self.args.val_interval == 0:  # Validation
                        if step != 10000 and step % self.args.val_interval == 0:  # Validation
                        # if step % self.args.val_interval == 0:  # Validation
                            logger.info('----------------------------------------')
                            logger.info('Start evaluating on evaluation set... ')
                            self.args.pick_top = True
                            self.args.finetune_bert = False

                            val_stat, best_model_save, best_recall_model_save = self.validate_rouge_baseline(
                                valid_iter_fct, step,
                                valid_gl_stats=valid_global_stats)


                            if best_model_save:
                                self._save(step, best=True, valstat=val_stat)
                                logger.info(f'Best model saved sucessfully at step %d' % step)
                                self.best_val_step = step

                            update_step(self.args.bert_data_path, self.args.gd_cell_step, "{} / {}".format(self.best_val_step, step))

                            # if best_recall_model_save:
                            #     self._save(step, best=True, valstat=val_stat, recall_model=True)
                            #     logger.info(f'Best model saved sucessfully at step %d' % step)
                            #     self.best_val_step = step

                            self.save_validation_results(step, val_stat)


                            logger.info('----------------------------------------')
                            self.model.train()
                            self.args.finetune_bert = True

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def plot_confusion(self, y_pred, y_true):
        cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=['Objective', 'Background', 'Method', 'Results', 'Other'],
                              title='Confusion matrix, without normalization')

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
        # pickle.dump(preds_sent_numbers, open(self.args.saved_list_name.replace('save_lists', 'second_phase'), "wb"))




    def validate_rouge_baseline(self, valid_iter_fct, step=0, valid_gl_stats=None, write_scores_to_pickle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """

        preds = {}
        preds_with_idx = {}
        golds = {}
        can_path = '%s_step%d.source' % (self.args.result_path, step)
        gold_path = '%s_step%d.target' % (self.args.result_path, step)
        json_pred = '%s_step%d.json' % (self.args.result_path, step)

        if step == self.best_val_step:
            can_path = '%s_step%d.source' % (self.args.result_path_test, step)
            gold_path = '%s_step%d.target' % (self.args.result_path_test, step)

        save_pred = open(can_path, 'w')
        save_gold = open(gold_path, 'w')
        save_json = open(json_pred, 'w')

        sent_scores_whole = {}
        sent_sects_whole_pred = {}
        sent_sects_whole_true = {}
        sent_labels_true = {}
        sent_tokens_count_whole = {}
        sent_numbers_whole = {}
        paper_srcs = {}
        paper_tgts = {}
        sent_sect_wise_rg_whole = {}
        sent_sections_txt_whole = {}
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()
        best_model_saved = False
        best_recall_model_saved = False

        valid_iter = valid_iter_fct()

        if not self.args.pick_top:
            with torch.no_grad():
                for batch in tqdm(valid_iter):
                    src = batch.src
                    src_intro = batch.src_intro
                    labels = batch.src_sent_rg
                    sent_bin_labels = batch.sent_labels
                    intro_sent_labels = batch.intro_sent_labels

                    segs = batch.segs

                    clss = batch.clss
                    intro_clss = batch.intro_clss

                    # section_rg = batch.section_rg
                    mask = batch.mask_src
                    mask_cls = batch.mask_cls
                    mask_intro_cls = batch.mask_intro_cls

                    mask_src_intro = batch.mask_src_intro
                    p_id = batch.paper_id
                    segment_src = batch.src_str
                    paper_tgt = batch.tgt_str
                    sent_sect_wise_rg = batch.sent_sect_wise_rg
                    sent_sections_txt = batch.sent_sections_txt
                    sent_numbers = batch.sent_numbers
                    sent_tokens_count = batch.sent_token_count

                    sent_sect_labels = batch.sent_sect_labels
                    if self.intro_cls:
                        sent_scores, mask_cls, loss, intro_loss, loss_src = self.model(src, src_intro, segs, clss, intro_clss,
                                                                 mask, mask_src_intro, mask_cls, mask_intro_cls,
                                                                 sent_bin_labels, intro_sent_labels,
                                                                 sent_sect_labels, p_id)

                        # acc, _ = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask,
                        #                             task='sent_sect')

                        batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                                 loss_sect=float(loss_src.cpu().data.numpy().sum()),
                                                 loss_sent=float(intro_loss.cpu().data.numpy().sum()),
                                                 n_docs=len(labels),
                                                 # n_acc=batch.batch_size,
                                                 RMSE=self._get_mertrics(sent_scores, labels, mask=mask_cls, task='sent'),
                                                 # accuracy=acc
                                                 )


                    stats.update(batch_stats)

                    # sent_scores = sent_scores + mask_cls.float()
                    sent_scores = sent_scores.cpu().data.numpy()

                    for idx, p_id in enumerate(p_id):
                        p_id = p_id.split('___')[0]

                        if p_id not in sent_scores_whole.keys():
                            masked_scores = sent_scores[idx] * mask_cls[idx].cpu().data.numpy()
                            masked_scores = masked_scores[np.nonzero(masked_scores)]

                            masked_sent_labels_true = (sent_bin_labels[idx] + 1) * mask_cls[idx].long()

                            masked_sent_labels_true = masked_sent_labels_true[np.nonzero(masked_sent_labels_true)].flatten()
                            masked_sent_labels_true = (masked_sent_labels_true - 1)

                            sent_scores_whole[p_id] = masked_scores
                            sent_labels_true[p_id] = masked_sent_labels_true.cpu()

                            masked_sents_sections_true = (sent_sect_labels[idx] + 1) * mask_cls[idx].long()

                            masked_sents_sections_true = masked_sents_sections_true[
                                np.nonzero(masked_sents_sections_true)].flatten()
                            masked_sents_sections_true = (masked_sents_sections_true - 1)
                            sent_sects_whole_true[p_id] = masked_sents_sections_true.cpu()


                            paper_srcs[p_id] = segment_src[idx]
                            if sent_numbers[0] is not None:
                                sent_numbers_whole[p_id] = sent_numbers[idx]
                            paper_tgts[p_id] = paper_tgt[idx]
                            sent_sect_wise_rg_whole[p_id] = sent_sect_wise_rg[idx]
                            sent_sections_txt_whole[p_id] = sent_sections_txt[idx]


                        else:
                            masked_scores = sent_scores[idx] * mask_cls[idx].cpu().data.numpy()
                            masked_scores = masked_scores[np.nonzero(masked_scores)]

                            masked_sent_labels_true = (sent_bin_labels[idx] + 1) * mask_cls[idx].long()
                            masked_sent_labels_true = masked_sent_labels_true[np.nonzero(masked_sent_labels_true)].flatten()
                            masked_sent_labels_true = (masked_sent_labels_true - 1)

                            sent_scores_whole[p_id] = np.concatenate((sent_scores_whole[p_id], masked_scores), 0)
                            sent_labels_true[p_id] = np.concatenate((sent_labels_true[p_id], masked_sent_labels_true.cpu()),
                                                                    0)

                            masked_sents_sections_true = (sent_sect_labels[idx] + 1) * mask_cls[idx].long()
                            masked_sents_sections_true = masked_sents_sections_true[
                                np.nonzero(masked_sents_sections_true)].flatten()
                            masked_sents_sections_true = (masked_sents_sections_true - 1)
                            sent_sects_whole_true[p_id] = np.concatenate(
                                (sent_sects_whole_true[p_id], masked_sents_sections_true.cpu()), 0)

                            paper_srcs[p_id] = np.concatenate((paper_srcs[p_id], segment_src[idx]), 0)
                            if sent_numbers[0] is not None:
                                sent_numbers_whole[p_id] = np.concatenate((sent_numbers_whole[p_id], sent_numbers[idx]), 0)
                                # sent_tokens_count_whole[p_id] = np.concatenate(
                                #     (sent_tokens_count_whole[p_id], sent_tokens_count[idx]), 0)

                            sent_sect_wise_rg_whole[p_id] = np.concatenate(
                                (sent_sect_wise_rg_whole[p_id], sent_sect_wise_rg[idx]), 0)
                            sent_sections_txt_whole[p_id] = np.concatenate(
                                (sent_sections_txt_whole[p_id], sent_sections_txt[idx]), 0)


            saved_dict = {}
            # source_sent_encodings = {}
            # sent_sect_wise_rg_whole = {}
            for p_id, sent_scores in tqdm(sent_scores_whole.items(), total=len(sent_scores_whole)):
                # source_sent_encodings[p_id] = [0]
                saved_dict[p_id] = (
                    p_id, sent_scores_whole[p_id], paper_srcs[p_id], paper_tgts[p_id], sent_sects_whole_true[p_id],
                    sent_sects_whole_true[p_id], sent_sections_txt_whole[p_id],
                    sent_labels_true[p_id], sent_sect_wise_rg_whole[p_id], sent_numbers_whole[p_id])
            print('Now saving picke file...')
            pickle.dump(saved_dict, open(self.args.saved_list_name, "wb"))

            # else  :
            PRED_LEN = self.args.val_pred_len
            # LENGTH_LIMIT= 100
            acum_f_sent_labels = 0
            acum_p_sent_labels = 0
            acum_r_sent_labels = 0
            acc_total = 0
            f_scores = {}

            for p_idx, (p_id, sent_scores) in enumerate(sent_scores_whole.items()):
                # sent_true_labels = pickle.load(open("sent_labels_files/pubmedL/val.labels.p", "rb"))
                # section_textual = np.array(section_textual)
                paper_sent_true_labels = np.array(sent_labels_true[p_id])

                sent_scores = np.array(sent_scores)
                p_src = np.array(paper_srcs[p_id])

                # selected_ids_unsorted = np.argsort(-sent_scores, 0)
                keep_ids = [idx for idx, s in enumerate(p_src) if
                            len(s.replace('.', '').replace(',', '').replace('(', '').replace(')', '').
                                replace('-', '').replace(':', '').replace(';', '').replace('*', '').split()) > 5 and
                            len(s.replace('.', '').replace(',', '').replace('(', '').replace(')', '').
                                replace('-', '').replace(':', '').replace(';', '').replace('*', '').split()) < 100
                            ]

                keep_ids = sorted(keep_ids)

                # top_sent_indexes = top_sent_indexes[top_sent_indexes]
                p_src = p_src[keep_ids]
                sent_scores = sent_scores[keep_ids]
                paper_sent_true_labels = paper_sent_true_labels[keep_ids]

                sent_scores = np.asarray([s - 1.00 for s in sent_scores])

                selected_ids_unsorted = np.argsort(-sent_scores, 0)

                _pred = []
                for j in selected_ids_unsorted:
                    if (j >= len(p_src)):
                        continue
                    candidate = p_src[j].strip()
                    if True:
                        # if (not _block_tri(candidate, _pred)):
                        _pred.append((candidate, j))

                    if (len(_pred) == PRED_LEN):
                        break
                _pred = sorted(_pred, key=lambda x: x[1])
                _pred_final_str = '<q>'.join([x[0] for x in _pred])

                preds[p_id] = _pred_final_str
                golds[p_id] = paper_tgts[p_id]
                preds_with_idx[p_id] = _pred
                if p_idx > 10:
                    f, p, r = _get_precision_(paper_sent_true_labels, [p[1] for p in _pred])
                    f_scores[p_id] = f

                else:
                    f, p, r = _get_precision_(paper_sent_true_labels, [p[1] for p in _pred], print_few=True, p_id=p_id)
                    f_scores[p_id] = f

                acum_f_sent_labels += f
                acum_p_sent_labels += p
                acum_r_sent_labels += r

            # for id, pred in preds.items():
            #     save_pred.write(pred.strip().replace('<q>', ' ') + '\n')
            #     save_gold.write(golds[id].replace('<q>', ' ').strip() + '\n')

            # print(f'Gold: {gold_path}')
            # print(f'Prediction: {can_path}')

            r1, r2, rl = self._report_rouge(preds.values(), golds.values())

            with open(json_pred, mode='w') as jF:
                for id, pred in preds.items():
                    dic = {'paper_id': id, 'pred': pred.strip().replace('<q>', ' '),
                           'gold': golds[id].strip().replace('<q>', ' '), 'f1': f_scores[id]}

                    json.dump(dic, jF)
                    jF.write('\n')

            stats.set_rl(r1, r2, rl)
            logger.info("F-score: %4.4f, Prec: %4.4f, Recall: %4.4f" % (
                acum_f_sent_labels / len(sent_scores_whole), acum_p_sent_labels / len(sent_scores_whole),
                acum_r_sent_labels / len(sent_scores_whole)))

            stats.set_ir_metrics(acum_f_sent_labels / len(sent_scores_whole),
                                 acum_p_sent_labels / len(sent_scores_whole),
                                 acum_r_sent_labels / len(sent_scores_whole))
            self.valid_rgls.append((r2 + rl) / 2)
            self._report_step(0, step,
                              # self.model.uncertainty_loss._sigmas_sq[0] if self.intro_cls else 0,
                              # self.model.uncertainty_loss._sigmas_sq[1] if self.intro_cls else 0,
                              valid_stats=stats)

            if len(self.valid_rgls) > 0:
                if self.min_rl < self.valid_rgls[-1]:
                    self.min_rl = self.valid_rgls[-1]
                    best_model_saved = True


        else:
            preds_sent_numbers = {}

            saved_dict = {}
            paper_sent_scores = []
            logger.info("Picking top sentences for second phase...")

            saved_dict_ = pickle.load(open(self.args.saved_list_name,'rb'))

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
                                          paper_sent_true_labels, self.bert, "normal"))


            # pickle.dump(paper_sent_scores, open(self.args.saved_list_name.replace('.p', '-sent-scores.p'), "wb"))
            # paper_sent_scores = pickle.load(open(self.args.saved_list_name.replace('.p', '-sent-scores.p'),'rb'))

            overall_recall1, preds_sent_numbers = self.extract_top_sents(paper_sent_scores)

            logger.info("Recall-top section stat: %4.4f" % (overall_recall1))

            self.overall_recalls.append(overall_recall1)
            if len(self.overall_recalls) > 0:
                if self.overall_recall < self.overall_recalls[-1]:
                    self.overall_recall = self.overall_recalls[-1]
                    best_recall_model_saved = True

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

            overall_recall2, overall_recall3 = 0, 0
            stats.set_overall_recall(overall_recall1, overall_recall2, overall_recall3)

            # if write_scores_to_pickle:
            pickle.dump(preds_sent_numbers,
                        open(self.args.saved_list_name.replace('.p', '-top-sents.p'), "wb"))

            ###############################################################################

            # else:
            #     if write_scores_to_pickle:

        return stats, best_model_saved, best_recall_model_saved

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):



        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            src_intro = batch.src_intro

            sent_rg_scores = batch.src_sent_rg

            sent_sect_labels = batch.sent_sect_labels

            sent_bin_labels = batch.sent_labels
            intro_sent_bin_labels = batch.intro_sent_labels

            sent_rg_true_scores = batch.src_sent_rg
            segs = batch.segs

            clss = batch.clss
            intro_clss = batch.intro_clss

            mask_src = batch.mask_src
            mask_src_intro = batch.mask_src_intro

            mask_cls = batch.mask_cls
            mask_intro_cls = batch.mask_intro_cls

            src_str = batch.src_str
            paper_ids = batch.paper_id
            if self.intro_cls:
                sent_scores, mask_cls, loss, loss_src, loss_intro = self.model(src, src_intro, segs, clss, intro_clss,
                                                         mask_src, mask_src_intro,
                                                         mask_cls, mask_intro_cls,
                                                         sent_bin_labels, intro_sent_bin_labels,
                                                         sent_sect_labels, paper_ids)
                batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                         n_acc=batch.batch_size,
                                         loss_sect=float(loss_src.cpu().data.numpy().sum()),
                                         loss_sent=float(loss_intro.cpu().data.numpy().sum()),
                                         n_docs = batch.batch_size,
                                         RMSE=self._get_mertrics(sent_scores, sent_rg_scores, mask=mask_cls, task='sent'),
                                         )


            loss.backward()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)


            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                # self.optim.step(report_stats=report_stats)

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step(report_stats)

    def _save(self, step, best=False, valstat=None, recall_model=False):

        BEST_MODEL_NAME = 'BEST_model_s%d_%4.4f_%4.4f_%4.4f.pt' % (step, valstat.r1, valstat.r2, valstat.rl)

        if recall_model:
            BEST_MODEL_NAME = 'Recall_BEST_model_s%d_rec01(%4.4f)_rec(%4.4f).pt' % (
                step, valstat.top_sents_recall_01_oracle, valstat.top_sents_recall_oracle)

        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        if best:
            checkpoint_path_best = os.path.join(self.args.model_path, BEST_MODEL_NAME)

        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)

        if not recall_model:
            logger.info("Saving checkpoint %s" % checkpoint_path)
        else:
            logger.info("Saving checkpoint recall %s" % checkpoint_path)

        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)

        if best:
            if not recall_model:
                update_rouge_score_drive(self.args.bert_data_path, self.args.gd_cells_rg,
                                         [valstat.r1, valstat.r2, valstat.rl])
                best_path = glob.glob(self.args.model_path + '/BEST_model*.pt')
            else:
                # update_recall_drive(self.args.bert_data_path, self.args.gd_cell_recall,
                #                     ("rec01: %4.4f / rec: %4.4f") % (
                #                         valstat.top_sents_recall_01_oracle, valstat.top_sents_recall_oracle))
                best_path = glob.glob(self.args.model_path + '/Recall_BEST_model*.pt')

            if len(best_path) > 0:
                for best in best_path:
                    os.remove(best)
                torch.save(checkpoint, checkpoint_path_best)
            else:
                torch.save(checkpoint, checkpoint_path_best)

        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases
        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)
        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats, alpha_1=0, alpha2=0
                               ):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, alpha_1, alpha2, report_stats, is_joint=self.intro_cls,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, alpha1=0, alpha2=0, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, alpha1, alpha2, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

    def _report_rouge(self, predictions, references):

        a_lst = []
        predictions = list(predictions)
        references = list(references)
        for i, p in enumerate(predictions):
            a_lst.append((p, references[i]))

        pool = Pool(24)
        rouge_scores = {"r1": [], "r2": [], "rl": []}
        for d in tqdm(pool.imap(_multi_rg, a_lst), total=len(a_lst)):
            if d is not None:
                rouge_scores["r1"].append(d[0])
                rouge_scores["r2"].append(d[1])
                rouge_scores["rl"].append(d[2])
        pool.close()
        pool.join()

        r1 = np.mean(rouge_scores["r1"])
        r2 = np.mean(rouge_scores["r2"])
        rl = np.mean(rouge_scores["rl"])

        if len(self.args.log_folds) > 0:
            with open(self.args.log_folds, mode='a') as f:
                f.write("{:.4f}\t{:.4f}\t{:.4f}".format(r1 / 100, r2 / 100, rl / 100))
                f.write('\n')
        logger.info("Metric\tScore\t95% CI")
        logger.info("ROUGE-1\t{:.2f}\t({:.2f},{:.2f})".format(r1 * 100, 0, 0))
        logger.info("ROUGE-2\t{:.2f}\t({:.2f},{:.2f})".format(r2 * 100, 0, 0))
        logger.info("ROUGE-L\t{:.2f}\t({:.2f},{:.2f})".format(rl * 100, 0, 0))

        logger.info("Data path: %s" % self.args.bert_data_path)
        logger.info("Model path: %s" % self.args.model_path)

        return r1, r2, rl

    def _get_mertrics(self, sent_scores, labels, mask=None, task='sent_sect'):

        labels = labels.to('cuda')
        sent_scores = sent_scores.to('cuda')
        mask = mask.to('cuda')

        if task == 'sent_sect':

            sent_scores = self.softmax_acc_pred(sent_scores)
            pred = torch.max(sent_scores, 2)[1]
            acc = (((pred == labels) * mask.cuda()).sum(dim=1)).to(dtype=torch.float) / \
                  mask.sum(dim=1).to(dtype=torch.float)

            return acc.sum().item(), pred

        else:
            mseLoss = self.rmse_loss(sent_scores.float(), labels.float())
            try:
                mseLoss = (mseLoss.float() * mask.float()).sum(dim=1)
            except:
                import pdb;pdb.set_trace()
            # sent_scores = self.softmax_sent(sent_scores)
            # import pdb;pdb.set_trace()
            # pred = torch.max(sent_scores, 2)[1]
            # acc = (((pred == labels) * mask.cuda()).sum(dim=1)).to(dtype=torch.float) / mask.sum(dim=1).to(dtype=torch.float)
            return mseLoss.sum().item()

    def _get_preds(self, sent_scores, labels, mask=None, task='sent_sect'):

        sent_scores = sent_scores.to('cuda')
        sent_scores = self.softmax_acc_pred(sent_scores)
        pred = torch.max(sent_scores, 2)[1]
        return pred

    def save_validation_results(self, step, val_stat):
        def is_non_zero_file(fpath):
            return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

        def check_path_existence(dir):
            if os.path.exists(dir):
                return
            else:
                os.makedirs(dir)

        check_path_existence(os.path.join(self.args.model_path, "val_results"))
        if not is_non_zero_file(os.path.join(self.args.model_path, "val_results", "val.json")) or step < 100:
            # if not is_non_zero_file(os.path.join(self.args.model_path, "val_results", "val.json")):
            results = collections.defaultdict(dict)

            for metric, score in zip(["RG-1", "RG-2", "RG-L"], [val_stat.r1, val_stat.r2, val_stat.rl]):
                results[str(step)][metric] = score

            # results[str(step)]["Recall-top-section-normal"] = val_stat.top_sents_recall_normal
            # results[str(step)]["Recall-top-section-eq"] = val_stat.top_sents_recall_sect_eq
            # results[str(step)]["Recall-top-section-stat"] = val_stat.top_sents_recall_sect_stat
            results[str(step)]["F1"] = val_stat.f
            with open(os.path.join(self.args.model_path, "val_results", "val.json"), mode='w') as F:
                json.dump(results, F, indent=4)
        else:
            results = json.load(open(os.path.join(self.args.model_path, "val_results", "val.json")))
            results_all = collections.defaultdict(dict)
            for key, val in results.items():
                for k, v in val.items():
                    results_all[key][k] = v

            # results_all[str(step)] = step
            # results_all[str(step)]["F1"] = val_stat.f
            for metric, score in zip(["RG-1", "RG-2", "RG-L"], [val_stat.r1, val_stat.r2, val_stat.rl]):
                results_all[str(step)][metric] = score

            # results_all[str(step)]["Recall-top-section-normal"] = val_stat.top_sents_recall_normal
            # results_all[str(step)]["Recall-top-section-eq"] = val_stat.top_sents_recall_sect_eq
            # results_all[str(step)]["Recall-top-section-stat"] = val_stat.top_sents_recall_sect_stat
            results_all[str(step)]["F1"] = val_stat.f
            with open(os.path.join(self.args.model_path, "val_results", "val.json"), mode='w') as F:
                json.dump(results_all, F, indent=4)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion.pdf')


def _get_ir_eval_metrics(preds_with_idx, sent_labels_true, n=10):
    avg_scores = {'f': [], 'p': [], 'r': []}
    for p_id, pred_with_idx in preds_with_idx.items():
        retrieved_idx = [pred[1] for pred in pred_with_idx]
        retrieved_true_labels = [sent_labels_true[p_id][idx] for idx in retrieved_idx]
        avg_scores['p'].append(retrieved_true_labels.count(1) / n)
    return np.mean(avg_scores['p'])


def _get_precision_(sent_true_labels, summary_idx, print_few=False, p_id=''):
    oracle_cout = sum(sent_true_labels)
    if oracle_cout == 0:
        return 0, 0, 0

    # oracle_cout = oracle_cout if oracle_cout > 0 else 1
    pos = 0
    neg = 0
    for idx in summary_idx:
        if sent_true_labels[idx] == 0:
            neg += 1
        else:
            pos += 1

    if print_few:
        logger.info("paper_id: {} ==> positive/negative cases: {}/{}".format(p_id, pos, neg))

    if pos == 0:
        return 0, 0, 0
    prec = pos / len(summary_idx)

    # recall --how many relevants are retrieved?
    recall = pos / int(oracle_cout)

    try:
        F = (2 * prec * recall) / (prec + recall)
        return F, prec, recall

    except Exception:
        traceback.print_exc()
        os._exit(2)


def _get_accuracy_sections(sent_sects_true, sent_sects_pred, summary_idx, print_few=False, p_id=''):
    acc = 0

    for idx in summary_idx:
        if sent_sects_true[idx] == sent_sects_pred[idx]:
            acc += 1

    if print_few:
        logger.info("paper_id: {} ==> acc: {}".format(p_id, acc / len(summary_idx)))

    return acc / len(summary_idx)


def _mult_top_sents(params, idx=None):
    sent_scores, p_id, p_src, p_sent_numbers, p_sent_sent_sects_true, p_sent_true_labels, bert, type = params
    indices = [i for i in range(len(p_sent_true_labels))]
    # print("idx {}; paper {}".format(idx, p_id))

    zip_sents_score = zip(indices, sent_scores, p_sent_sent_sects_true)

    sent_scores_section_sorted = sorted(zip_sents_score, key=lambda element: (element[2], -element[1]))
    paper_sampling_sent_numbers = []
    paper_sampling_sent_indeces = []

    if type=="section-equal":
        sent_scores_0 = [s for s in sent_scores_section_sorted if s[-1] == 0]
        sent_scores_1 = [s for s in sent_scores_section_sorted if s[-1] == 1]
        sent_scores_2 = [s for s in sent_scores_section_sorted if s[-1] == 2]
        sent_scores_3 = [s for s in sent_scores_section_sorted if s[-1] == 3]
        sections = []
        sent_poninters_in_sects = {}
        sent_scores_dict = {}
        checked_full = {}

        if len(sent_scores_0) > 0:
            sent_scores_dict[0] = sent_scores_0
            sections.append(0)
            sent_poninters_in_sects[0] = 0
            checked_full[0] = False

        if len(sent_scores_1) > 0:
            sent_scores_dict[1] = sent_scores_1
            sections.append(1)
            sent_poninters_in_sects[1] = 0
            checked_full[1] = False

        if len(sent_scores_2) > 0:
            sent_scores_dict[2] = sent_scores_2
            sections.append(2)
            sent_poninters_in_sects[2] = 0
            checked_full[2] = False

        if len(sent_scores_3) > 0:
            sent_scores_dict[3] = sent_scores_3
            sections.append(3)
            sent_poninters_in_sects[3] = 0
            checked_full[3] = False

        section_pointer = 0
        _pred = []
        paper_sampling_sent_numbers = []
        paper_sent_sects = []


        while bert.cal_token_len_prep(_pred) <= 2500:
            try:
                pred_item = sent_scores_dict[sections[section_pointer]][sent_poninters_in_sects[sections[section_pointer]]]
            except:
                checked_full[sections[section_pointer]] = True
                full_flag = True
                for k, v in checked_full.items():
                    full_flag = v and full_flag

                if full_flag:
                    break
                else:
                    # sections.remove(section_pointer)
                    if sections[section_pointer] == sections[-1]:
                        section_pointer = 0
                    else:
                        # print('here {}'.format(sections[section_pointer]))
                        section_pointer += 1
                continue
            # import pdb;
            # pdb.set_trace()

            sent_poninters_in_sects[sections[section_pointer]] += 1
            _pred.append((p_src[pred_item[0]], pred_item[0]))
            paper_sampling_sent_numbers += [p_sent_numbers[pred_item[0]]]
            paper_sent_sects += [p_sent_sent_sects_true[pred_item[0]]]
            paper_sampling_sent_indeces += [pred_item[0]]

            if section_pointer == len(sections) - 1:
                section_pointer = 0
            else:
                section_pointer += 1

            # if bert.cal_token_len_prep(_pred) > 2500:
            #     break
        if bert.cal_token_len_prep(_pred) > 2500:
            _pred = _pred[:-1]
        # print("processed {}".format(p_id))
        # print("----")
        _pred = sorted(_pred, key=lambda x: x[1])

    elif type=="normal":
        section_based_train_stat=False
        zip_sents_score = zip(indices, sent_scores)
        sent_scores = sorted(zip_sents_score, key=lambda element: element[1], reverse=True)
        # import pdb;pdb.set_trace()
        sentence_pointer = 0
        _pred = []

        # oracle_real_sent_numbers = p_sent_numbers[oracle_indeces]

        while bert.cal_token_len_prep(_pred) <= 3000:
            try:
                pred_item = sent_scores[sentence_pointer]
            except:
                break
            sentence_pointer += 1
            _pred.append((p_src[pred_item[0]], pred_item[0]))
            paper_sampling_sent_numbers += [p_sent_numbers[pred_item[0]]]
            paper_sampling_sent_indeces += [pred_item[0]]

        _pred = _pred[:-1]
        # print("processed {}".format(p_id))
        # print("----")
        _pred = sorted(_pred, key=lambda x: x[1])

    elif type=="section-stat":
        # train_stat = {"0": 0.2930, "1": 0.2027, "2": 0.4666, "3": 0.0378} #arXiv-L
        # train_stat = {"0": 0.2098, "1": 0.2897, "2": 0.415, "3": 0.0854} #LongSumm
        # train_stat = {"0": 0.3044, "1": 0.1672, "2": 0.4872, "3": 0.0412} #PubmedL
        train_stat = {"0": 0.2096, "1": 0.2900, "2": 0.4154, "3": 0.0850} #LongSumm-introGuided
        train_stat_norm_first_pass = {}

        for k, t in train_stat.items():
            train_stat_norm_first_pass[k] = int(t / min(train_stat.values()))

        sent_scores_0 = [s for s in sent_scores_section_sorted if s[-1] == 0]
        sent_scores_1 = [s for s in sent_scores_section_sorted if s[-1] == 1]
        sent_scores_2 = [s for s in sent_scores_section_sorted if s[-1] == 2]
        sent_scores_3 = [s for s in sent_scores_section_sorted if s[-1] == 3]
        sections = []
        sent_poninters_in_sects = {}
        sent_scores_dict = {}
        checked_full = {}

        if len(sent_scores_0) > 0:
            sent_scores_dict[0] = sent_scores_0
            sections.append(0)
            sent_poninters_in_sects[0] = 0
            checked_full[0] = False

        if len(sent_scores_1) > 0:
            sent_scores_dict[1] = sent_scores_1
            sections.append(1)
            sent_poninters_in_sects[1] = 0
            checked_full[1] = False

        if len(sent_scores_2) > 0:
            sent_scores_dict[2] = sent_scores_2
            sections.append(2)
            sent_poninters_in_sects[2] = 0
            checked_full[2] = False

        if len(sent_scores_3) > 0:
            sent_scores_dict[3] = sent_scores_3
            sections.append(3)
            sent_poninters_in_sects[3] = 0
            checked_full[3] = False

        section_pointer = 0
        _pred = []
        paper_sampling_sent_numbers = []
        paper_sent_sects = []
        iteration=1
        while bert.cal_token_len_prep(_pred) <= 2500:
            try:
                pred_items = sent_scores_dict[sections[section_pointer]][sent_poninters_in_sects[sections[section_pointer]]:iteration*train_stat_norm_first_pass[str(section_pointer)]]
            except:
                # print('jer')
                continue
                # if sent_poninters_in_sects[section_pointer] >= len(sent_scores_dict[sections[section_pointer]]):
                #     checked_full[sections[section_pointer]] = True
                #     # sections.remove(section_pointer)
                #     if section_pointer == len(sections) - 1:
                #         section_pointer = 0
                #     else:
                #         # print('here {}'.format(sections[section_pointer]))
                #         section_pointer += 1
                #     full_flag = True
                #     for k, v in checked_full.items():
                #         full_flag = v and full_flag
                #
                #     if full_flag:
                #         break
                # else:
                #     continue
            if len(pred_items) != 0:
                sent_poninters_in_sects[sections[section_pointer]] += len(pred_items)
                for pred_item in pred_items:
                    _pred.append((p_src[pred_item[0]], pred_item[0]))
                    paper_sampling_sent_numbers += [p_sent_numbers[pred_item[0]]]
                    paper_sent_sects += [p_sent_sent_sects_true[pred_item[0]]]
                    paper_sampling_sent_indeces += [pred_item[0]]

                if section_pointer == len(sections) - 1:
                    section_pointer = 0
                    iteration+=1
                else:
                    section_pointer += 1

            else:
                checked_full[sections[section_pointer]] = True
                # sections.remove(section_pointer)
                if section_pointer == len(sections) - 1:
                    section_pointer = 0
                else:
                    # print('here {}'.format(sections[section_pointer]))
                    section_pointer += 1
                full_flag = True
                for k, v in checked_full.items():
                    full_flag = v and full_flag

                if full_flag:
                    break


        _pred = _pred[:-1]
        _pred = sorted(_pred, key=lambda x: x[1])

    # calculate recall for the top retrieved docs...


    tp = len([1 for x in paper_sampling_sent_indeces if p_sent_true_labels[x] == 1])
    if sum(p_sent_true_labels) != 0:
        recall = (tp) / sum(p_sent_true_labels)
    else:
        recall=1

    return p_id, sorted(paper_sampling_sent_numbers), recall

