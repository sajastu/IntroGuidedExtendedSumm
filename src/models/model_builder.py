import copy

import torch
import torch.nn as nn
# from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
# from torch.autograd import Variable
from transformers import BertModel, BertConfig, LongformerModel, LongformerConfig
# from models.pointer_generator.PG_transformer import PointerGeneratorTransformer
from models.decoder import TransformerDecoder
from models.encoder import ExtTransformerEncoder
from models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    return optim


def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=1)
    generator = nn.Sequential(nn.Linear(dec_hidden_size, vocab_size), gen_func)
    generator.to(device)
    return generator


class Bert(nn.Module):
    def __init__(self, large, model_name, temp_dir, finetune=False):
        super(Bert, self).__init__()

        if model_name == 'bert':
            if (large):
                self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
            else:
                self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
                # config = BertConfig.from_pretrained('allenai/scibert_scivocab_uncased')
                # config.gradient_checkpointing = True
                # self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=temp_dir, config=config)
                # self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=temp_dir)

        elif model_name == 'scibert':
            self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=temp_dir)

        elif model_name == 'longformer':
            if large:
                self.model = LongformerModel.from_pretrained('allenai/longformer-large-4096', cache_dir=temp_dir)
            else:
                # configuration = LongformerConfig()
                # model = LongformerModel(configuration)
                # configuration = model.config
                # import pdb;pdb.set_trace()
                self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096', cache_dir=temp_dir)
                self.model.config.gradient_checkpointing = True
                self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096', cache_dir=temp_dir, config=self.model.config)



        self.model_name = model_name
        self.finetune = finetune

    def forward(self, x, segs, mask_src, mask_cls, clss):
        if (self.finetune):

            if self.model_name == 'bert' or self.model_name == 'scibert':
                top_vec, _ = self.model(x, attention_mask=mask_src, token_type_ids=segs)

            elif self.model_name == 'longformer':
                # import pdb;
                # pdb.set_trace()
                global_mask = torch.zeros(mask_src.shape, dtype=torch.long, device='cuda').unsqueeze(0)
                global_mask[:, :, clss.long()] = 1
                global_mask = global_mask.squeeze(0)
                top_vec = self.model(x, attention_mask=mask_src.long(), global_attention_mask=global_mask)['last_hidden_state']

        else:
            self.eval()
            with torch.no_grad():
                if self.model_name == 'bert' or self.model_name == 'scibert':
                    top_vec, _ = self.model(x, attention_mask=mask_src, token_type_ids=segs)

                elif self.model_name == 'longformer':
                    global_mask = torch.zeros(mask_src.shape, dtype=torch.long, device='cuda').unsqueeze(0)
                    global_mask[:, :, clss.long()] = 1
                    global_mask = global_mask.squeeze(0)
                    top_vec = self.model(x, attention_mask=mask_src.long(), global_attention_mask=global_mask)[
                        'last_hidden_state']

        return top_vec


class BertVanilla(nn.Module):
    def __init__(self, large, model_name, temp_dir, finetune=False):
        super(BertVanilla, self).__init__()

        if model_name == 'bert':
            if (large):
                self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
            else:
                self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        elif model_name == 'scibert':
            self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=temp_dir)

        elif model_name == 'longformer':
            if large:
                self.model = LongformerModel.from_pretrained('allenai/longformer-large-4096', cache_dir=temp_dir)
            else:
                self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096', cache_dir=temp_dir)
                self.model.config.gradient_checkpointing = True
                self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096', cache_dir=temp_dir,
                                                             config=self.model.config)

                # self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096', cache_dir=temp_dir)

        self.model_name = model_name
        self.finetune = finetune

    def forward(self, x, segs, mask_src, mask_cls, clss):
        if (self.finetune):
            if self.model_name == 'bert' or self.model_name == 'scibert':
                cls_vec, _ = self.model(x, attention_mask=mask_src, token_type_ids=segs)

            elif self.model_name == 'longformer':

                global_mask = torch.zeros(mask_src.shape, dtype=torch.long, device='cuda').unsqueeze(0)
                global_mask[:, :, [0]] = 1 # <s> or [cls] token
                global_mask = global_mask.squeeze(0)
                top_vec = self.model(x, attention_mask=mask_src.long(), global_attention_mask=global_mask)['last_hidden_state']

        else:
            self.eval()
            with torch.no_grad():
                if self.model_name == 'bert' or self.model_name == 'scibert':
                    top_vec, _ = self.model(x, attention_mask=mask_src, token_type_ids=segs)

                elif self.model_name == 'longformer':
                    global_mask = torch.zeros(mask_src.shape, dtype=torch.long, device='cuda').unsqueeze(0)
                    global_mask[:, :, [0]] = 1  # <s> or [cls] token
                    global_mask = global_mask.squeeze(0)
                    top_vec = self.model(x, attention_mask=mask_src.long(), global_attention_mask=global_mask)[
                        'last_hidden_state']

        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint, intro_cls=True, intro_sents_cls=False, intro_top_cls=False, num_intro_sample=8):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.num_intro_sample = num_intro_sample
        self.intro_cls = intro_cls
        self.sentence_encoder = SentenceEncoder(args, device, checkpoint)
        self.sentence_encoder_intro = SentenceEncoder(args, device, checkpoint)
        self.sentence_predictor = SentenceExtLayer()
        self.intro_sentence_predictor = SentenceExtLayer()
        self.loss_sentence_picker = torch.nn.BCELoss(reduction='none')
        self.loss_intro_sentence_picker = torch.nn.BCELoss(reduction='none')
        self.intro_combiner = IntroSentenceCombiner(num_intro_sample)

        if intro_cls:
            self.intro_cls_encoder = BertVanilla(args.large, args.model_name, args.temp_dir, args.finetune_bert)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)

        else:
            for p in self.sentence_predictor.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        self.to(device)

    def _get_topN_sents_ids(self, p_id, intro_sents_score, mask_intro_cls, n):
        """
        :param intro_sents_score: tensor [B, CLS_LEN]
        :param mask_intro_cls: tensor [B, CLS_LEN]
        :param n: int N

        :return: top_out tensor [B, N]
        """
        try:
            vals, indices = intro_sents_score.topk(n, dim=1)
        except:
            import pdb;pdb.set_trace()

        top_out = torch.zeros((mask_intro_cls.size(0), mask_intro_cls.size(1))).cuda()
        top_out.scatter_(1, indices, 1.)
        return top_out

    def _get_top_representations(self, top_sent_mask, intro_source_sents_repr):
        """
        :param top_sent_mask: tensor [B (int), CLS_LEN (int)]:
        :param intro_source_sents_repr: tensor [B (int), CLS_LEN (int)]:
        :return: zero_out_repr: tensor [B (int), N (top_sents), EMB_DIM (int)]
        """

        zero_out_repr = top_sent_mask.unsqueeze(2).repeat(1, 1, intro_source_sents_repr.size(2)).cuda() * intro_source_sents_repr
        zero_out_repr = zero_out_repr[top_sent_mask>0].view(intro_source_sents_repr.size(0), int(top_sent_mask.sum(dim=1)[0].item()), -1)
        return zero_out_repr


    def forward(self, src, src_intro, segs, clss, intro_clss, mask_src, mask_src_intro, mask_cls, mask_intro_cls, sent_bin_labels, intro_sent_bin_labels, sent_sect_labels, p_id, is_inference=False,
                return_encodings=False):

        source_sents_repr = self.sentence_encoder(src, segs, clss, mask_src, mask_cls)
        intro_source_sents_repr = self.sentence_encoder_intro(src_intro, segs, intro_clss, mask_src_intro, mask_intro_cls)

        # intro_repr = self.intro_cls_encoder(x=src_intro, segs=None, mask_src=mask_src_intro, mask_cls=mask_intro_cls, clss=intro_clss)

        # intro_aware_repr = torch.cat((source_sents_repr, intro_repr.repeat(1, source_sents_repr.size(1), 1)), 2)
        intro_sents_score = self.intro_sentence_predictor(intro_source_sents_repr, mask_intro_cls)
        intro_repr_topN = self._get_top_representations(self._get_topN_sents_ids(p_id, intro_sents_score, mask_intro_cls, self.num_intro_sample), intro_source_sents_repr)
        try:
            src_intro_guided = self.intro_combiner(intro_repr_topN, source_sents_repr)
        except:
            import pdb;pdb.set_trace()
        src_sents_score = self.sentence_predictor(src_intro_guided, mask_cls)
        if self.intro_cls:
            try:
                loss = self.loss_sentence_picker(src_sents_score, sent_bin_labels.float())
            except:
                import pdb;pdb.set_trace()
            intro_loss = self.loss_intro_sentence_picker(intro_sents_score, intro_sent_bin_labels.float())
            loss = ((loss * mask_cls.float()).sum() / mask_cls.sum(dim=1)).sum()
            loss_src = loss
            intro_loss = ((intro_loss * mask_intro_cls.float()).sum() / mask_intro_cls.sum(dim=1)).sum()
            if not is_inference:
                loss = loss / loss.numel()
                loss_src = loss
                intro_loss = intro_loss / intro_loss.numel()

            loss = (0.5*loss) + (0.5*intro_loss)

            if return_encodings:
                return src_sents_score, mask_cls, loss, None, None, source_sents_repr

            return src_sents_score, mask_cls, loss, intro_loss, loss_src


class SentenceEncoder(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(SentenceEncoder, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.model_name, args.temp_dir, args.finetune_bert)
        self.ext_transformer_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size,
                                                           args.ext_heads,
                                                           args.ext_dropout, args.ext_layers)

        if args.max_pos > 512 and args.model_name == 'bert':
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            import pdb;
            pdb.set_trace()

            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if args.max_pos > 4096 and args.model_name == 'longformer':
            my_pos_embeddings = nn.Embedding(args.max_pos + 2, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:4097] = self.bert.model.embeddings.position_embeddings.weight.data[:-1]
            my_pos_embeddings.weight.data[4097:] = self.bert.model.embeddings.position_embeddings.weight.data[
                                                   1:args.max_pos + 2 - 4096]
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        self.sigmoid = nn.Sigmoid()

    def forward(self, src, segs, clss, mask_src, mask_cls):
        # self.eval()

        top_vec = self.bert(src, segs, mask_src, mask_cls, clss)
        # top_vec = checkpoint.checkpoint(
        #     self.custom_sent_decider(self.bert),
        #     src, segs, mask_src
        # )
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        encoded_sent = self.ext_transformer_layer(sents_vec, mask_cls)
        # import pdb;pdb.set_trace()
        return encoded_sent


class IntroSentenceCombiner(nn.Module):
    def __init__(self, n_input, n_output=1):
        super(IntroSentenceCombiner, self).__init__()
        self.flatten = nn.Flatten()
        self.combiner1 = nn.Linear(n_input*768, n_output*768)
        self.combiner2 = nn.Linear(2*768, 768)

    def forward(self, x, y):
        out = self.combiner1(self.flatten(x)).unsqueeze(1)
        intro_aware_repr = torch.cat((y, out.repeat(1, y.size(1), 1)), 2)
        intro_aware_repr = self.combiner2(intro_aware_repr)
        # intro_aware_repr = self.combiner3(intro_aware_repr)

        # intro_aware_repr_gate = self.filtering_gate(intro_aware_repr)
        # intro_aware_repr = y * intro_aware_repr_gate

        return intro_aware_repr

class SentenceExtLayer(nn.Module):
    def __init__(self,):
        super(SentenceExtLayer, self).__init__()
        # self.combiner = nn.Linear(1536, 768)
        self.wo = nn.Linear(768, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        # sent_scores = self.seq_model(x)

        # sent_scores = self.sigmoid(self.wo(self.combiner(x)))
        sent_scores = self.sigmoid(self.wo(x))

        # modules = [module for k, module in self.seq_model._modules.items()]
        # input_var = torch.autograd.Variable(x, requires_grad=True)
        # sent_scores = checkpoint_sequential(modules, 2, input_var)
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores


class SectionExtLayer(nn.Module):
    def __init__(self):
        super(SectionExtLayer, self).__init__()
        self.wo_2 = nn.Linear(768, 5, bias=True)
        #
        # self.dropout = nn.Dropout(0.5)
        # self.leakyReLu = nn.LeakyReLU()

    def forward(self, x, mask):
        # sent_sect_scores = self.dropout(self.wo_2(x))
        sent_sect_scores = self.wo_2(x)
        sent_sect_scores = sent_sect_scores.squeeze(-1) * mask.unsqueeze(2).expand_as(sent_sect_scores).float()

        return sent_sect_scores


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None