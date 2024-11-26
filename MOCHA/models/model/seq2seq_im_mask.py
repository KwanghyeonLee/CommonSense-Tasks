import os
import cv2
import torch
import numpy as np
import nn.vnn as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask

from PIL import Image

import constants
classes = [0] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']

from nn.resnet import Resnet


class Module(Base):

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        # encoder and self-attention
        self.enc_goal = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att_goal = vnn.SelfAttn(args.dhid*2)

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        # frame mask decoder
        if not args.panoramic:
            decoder = vnn.ConvFrameMaskDecoderProgressMonitor if self.subgoal_monitoring else vnn.ConvFrameMaskDecoder
        else:
            decoder = vnn.ConvFrameMaskDecoderProgressMonitorPanoramicConcat if args.panoramic_concat else vnn.ConvFrameMaskDecoderProgressMonitorPanoramicHier
            
        self.dec = decoder(vocab, args, self.emb_action_low, args.dframe, 2*args.dhid,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing,
                           panoramic=args.panoramic,
                           orientation=args.orientation)

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # bce reconstruction loss
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.ce_loss = torch.nn.CrossEntropyLoss()

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv_panoramic.pt'

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()

        self.panoramic = args.panoramic
        self.orientation = args.orientation
        self.man_action = self.vocab['action_low'].word2index('Manipulate', train=False)
        
    def featurize(self, batch, action_high_order, objects2find, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        for ex, swapColor in batch:
            ###########
            # auxillary
            ###########

            # action_high_order = np.array([ah['action'] for ah in ex['num']['action_high']])
            feat['action_high_order'].append(action_high_order)
            if not self.test_mode:
                low_to_high_idx = ex['num']['low_to_high_idx']
                action_high = action_high_order[low_to_high_idx]
                feat['action_high'].append(action_high)
            
                val_action_high = (action_high == self.vocab['action_high'].word2index('GotoLocation', train=False)).astype(np.int64)
            
                v=0
                while v < (len(val_action_high)-1):
                    if (val_action_high[v] - val_action_high[v+1]) == 1:
                        val_action_high[v+1] = 1
                        v+=1
                    v+=1
                val_action_high[-1]=1
            
            #########
            # inputs
            #########

            # serialize segments
            self.serialize_lang_action(ex, action_high_order)

            # goal and instr language
            lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']

            # zero inputs if specified
            lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
            lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr

            # append goal + instr
            feat['lang_goal'].append(lang_goal)
            feat['lang_instr'].append(lang_instr)

            #########
            # outputs
            #########

            if not self.test_mode:
                alow = []
                alow_manip = []
                obj_high_indices = []
               

                for ia, a in enumerate(ex['num']['action_low']):
                    if val_action_high[ia]==1 and a['action'] in self.vocab['action_low'].word2index(['<<pad>>', '<<seg>>', '<<stop>>', 
                                                                                                     'LookDown_15', 'LookUp_15',
                                                                                                     'RotateLeft_90', 'RotateRight_90', 
                                                                                                     'MoveAhead_25'], train=False):
                        alow.append(a['action'])
                    elif val_action_high[ia]==1:
                        alow.append(self.vocab['action_low'].word2index('Manipulate', train=False))
                    
                    if not (a['action'] in self.vocab['action_low'].word2index(['<<pad>>', '<<seg>>', '<<stop>>', 
                                                                                 'LookDown_15', 'LookUp_15',
                                                                                 'RotateLeft_90', 'RotateRight_90', 
                                                                                 'MoveAhead_25'], train=False)):
                        alow_manip.append(a['action'])
                        obj_high_indices.append(low_to_high_idx[ia])

                    


                feat['action_low'].append(alow)
                feat['action_low_manip'].append(alow_manip)
                feat['obj_high_indices'].append(obj_high_indices)

                if self.args.subgoal_aux_loss_wt > 0:
                    feat['subgoals_completed'].append(np.array(ex['num']['low_to_high_idx'])[val_action_high.nonzero()[0].astype(int)] / self.max_subgoals)

                # progress monitor supervision
                if self.args.pm_aux_loss_wt > 0:
                    num_actions = len(alow)
                    subgoal_progress = [(i+1)/float(num_actions) for i in range(num_actions)]
                    feat['subgoal_progress'].append(subgoal_progress)



            new_obj_list = [self.vocab['objnav'].word2index(of.lower(), train=False) for of in objects2find]

            feat['objnav'].append(new_obj_list)
                # feat['action_low_mask_label'].append(indices)
                


            # low-level valid interact

            # load Resnet features from disk
            if load_frames and not self.test_mode:
                root = self.get_task_root(ex)
                if not swapColor:
                    im = torch.load(os.path.join(root, self.feat_pt))
                elif swapColor in [1, 2]:
                    im = torch.load(os.path.join(root, 'feat_conv_colorSwap{}_panoramic.pt'.format(swapColor)))
                elif swapColor in [3, 4, 5, 6]:
                    im = torch.load(os.path.join(root, 'feat_conv_onlyAutoAug{}_panoramic.pt'.format(swapColor - 2)))


                feat['frames'].append(im[2][val_action_high.nonzero()[0]])
                feat['frames_left'].append(im[0][val_action_high.nonzero()[0]])
                feat['frames_up'].append(im[1][val_action_high.nonzero()[0]])
                feat['frames_down'].append(im[3][val_action_high.nonzero()[0]])
                feat['frames_right'].append(im[4][val_action_high.nonzero()[0]])

        assert(np.all(np.array(list(map(len, feat['lang_instr']))) == np.array(list(map(len, feat['objnav']))))) 
        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal'}:
                # language embedding and padding
                seqs = [torch.tensor(vv, device=device) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                embed_seq = self.emb_word(pad_seq)
                packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = packed_input
            elif k in {'lang_instr'}:
                # language embedding and padding
                num_instr = np.array(list(map(len, v)))
                seqs = [torch.tensor(vvv, device=device) for vv in v for vvv in vv]


                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                
                embed_seq = self.emb_word(pad_seq)
                fin_seq = []
                
                in_idx = 0
                for l in num_instr:
                    fin_seq.append(embed_seq[in_idx:in_idx+l])
                    in_idx += l
                feat[k] = {'seq': fin_seq} 
            
            elif k in {'action_low_mask'}:
                # mask padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                feat[k] = seqs
            elif k in {'action_low_mask_label'}:
                # label
                seqs = torch.tensor([vvv for vv in v for vvv in vv], device=device, dtype=torch.long)
                feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                # auxillary padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
            elif k in {'action_high', 'action_high_order'}:
                seqs = [torch.tensor(vv, device=device, dtype= torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.vocab['action_high'].word2index('<<pad>>'))
                feat[k] = pad_seq
            elif k in {'objnav'}:
                seqs = [torch.tensor(vv, device=device, dtype= torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.vocab['objnav'].word2index('<<pad>>'))
                embed_seq = self.emb_objnav(pad_seq)
                feat[k] = embed_seq
            else:
                # default: tensorize and pad sequence
                seqs = [torch.tensor(vv, device=device, dtype=torch.float if ('frames' in k or 'orientation' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq

        return feat


    def serialize_lang_action(self, feat, action_high_order):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        assert(len(action_high_order) == len(feat['num']['lang_instr']))
        # print(action_high_order)
        action_high_order = (action_high_order == self.vocab['action_high'].word2index('GotoLocation', train=False)).nonzero()[0]
        # print('#', action_high_order)
        li = []
        for ai in range(len(action_high_order)-1):
            li.append([word for desc in feat['num']['lang_instr'][action_high_order[ai]:action_high_order[ai+1]] for word in desc])
        
        # print(len(feat['num']['lang_instr']), action_high_order)
        li.append([word for desc in feat['num']['lang_instr'][action_high_order[-1]:] for word in desc])

        feat['num']['lang_instr'] = li


        if not self.test_mode:
            feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]


    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask


    def forward(self, feat, max_decode=300):

        frames = self.vis_dropout(feat['frames'])
        if self.panoramic:
            frames_left = self.vis_dropout(feat['frames_left'])
            frames_up = self.vis_dropout(feat['frames_up'])
            frames_down = self.vis_dropout(feat['frames_down'])
            frames_right = self.vis_dropout(feat['frames_right'])
            res = self.dec(feat['objnav'], feat['lang_instr']['seq'], frames, frames_left, frames_up, frames_down, frames_right, max_decode=max_decode, gold=feat['action_low'])#, state_0_instr=state_0_instr)
        else:
            res = self.dec(enc_lang_goal, enc_lang_instr, frames, max_decode=max_decode, gold=feat['action_low'], state_0_goal=state_0_goal, state_0_instr=state_0_instr)
        feat.update(res)
        return feat


    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        emb_lang = feat['lang_goal']
        
        self.lang_dropout(emb_lang.data)
        
        enc_lang, _ = self.enc_goal(emb_lang)
        enc_lang, _ = pad_packed_sequence(enc_lang, batch_first=True)
        
        self.lang_dropout(enc_lang)
        
        cont_lang = self.enc_att_goal(enc_lang)

        return cont_lang, enc_lang

    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t_goal': None,
            'state_t_instr': None,
            'e_t': None,
            'cont_lang_goal': None,
            'enc_lang_goal': None,
            'cont_lang_instr': None,
            'enc_lang_instr': None,
            'lang_index':0,
            'enc_obj': None,
        }


    def step(self, feat, lang_index=0, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features (instr)
        if self.r_state['cont_lang_instr'] is None and self.r_state['enc_lang_instr'] is None:
            self.r_state['cont_lang_instr'], self.r_state['enc_lang_instr'] = self.dec.encode_lang_instr(feat['lang_instr']['seq'][0][lang_index].unsqueeze(0))
            self.r_state['enc_obj'] = self.dec.object_enc(feat['objnav'][0][lang_index].unsqueeze(0))

        # print(len(feat['lang_instr']['seq']), feat['lang_instr']['seq'][0].shape)

        # initialize embedding and hidden states (instr)
        if self.r_state['e_t'] is None and self.r_state['state_t_instr'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang_instr'].size(0), 1)
            self.r_state['state_t_instr'] = self.r_state['cont_lang_instr'], torch.zeros_like(self.r_state['cont_lang_instr'])

        if lang_index != self.r_state['lang_index']:
            self.r_state['cont_lang_instr'], self.r_state['enc_lang_instr'] = self.dec.encode_lang_instr(feat['lang_instr']['seq'][0][lang_index].unsqueeze(0))
            self.r_state['enc_obj'] = self.dec.object_enc(feat['objnav'][0][lang_index].unsqueeze(0))
            self.r_state['lang_index'] = lang_index



        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        # decode and save embedding and hidden states
        if self.panoramic:

            out_action_low, state_t_instr, attn_score_t_instr, _, _ = self.dec.step(enc_obj=self.r_state['enc_obj'], 
                                                                                enc_instr=self.r_state['enc_lang_instr'], 
                                                                                frame=feat['frames'][:, 0],
                                                                                frame_left=feat['frames_left'][:, 0],
                                                                                frame_up=feat['frames_up'][:, 0],
                                                                                frame_down=feat['frames_down'][:, 0],
                                                                                frame_right=feat['frames_right'][:, 0],
                                                                                e_t=e_t, 
                                                                                state_tm1_instr=self.r_state['state_t_instr'])

        # save states
        self.r_state['state_t_instr'] = state_t_instr
        self.r_state['e_t'] = self.dec.emb(out_action_low.max(1)[1])

        # output formatting
        feat['out_action_low'] = out_action_low.unsqueeze(0)

        return feat


    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        for (ex, _), alow, alow_mask in zip(batch, feat['out_action_low'].max(2)[1].tolist(), feat['out_action_low_mask']):
            
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]

            if clean_special_tokens:
                # remove <<stop>> tokens
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]

            # index to API actions
            words = self.vocab['action_low'].index2word(alow)


            pred[self.get_task_and_ann_id(ex)] = {
                'action_low': ' '.join(words),
            }

        return pred


    def embed_action(self, action):
        '''
        embed low-level action
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        action_num = torch.tensor(self.vocab['action_low'].word2index(action), device=device)
        action_emb = self.dec.emb(action_num).unsqueeze(0)
        return action_emb


    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # GT and predictions
        p_alow = out['out_action_low'].view(-1, len(self.vocab['action_low']))
        l_alow = feat['action_low'].view(-1)
        p_alow_mask = out['out_action_low_mask']
        valid = feat['action_low_valid_interact']

        # action loss
        pad_valid = (l_alow != self.pad)
        alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none')
        alow_loss *= pad_valid.float()
        alow_loss = alow_loss.mean()
        losses['action_low'] = alow_loss * self.args.action_loss_wt

        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            p_subgoal = feat['out_subgoal'].squeeze(2)
            l_subgoal = feat['subgoals_completed']
            sg_loss = self.mse_loss(p_subgoal, l_subgoal)
            sg_loss = sg_loss.view(-1) * pad_valid.float()
            subgoal_loss = sg_loss.mean()
            losses['subgoal_aux'] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.pm_aux_loss_wt > 0:
            p_progress = feat['out_progress'].squeeze(2)
            l_progress = feat['subgoal_progress']
            pg_loss = self.mse_loss(p_progress, l_progress)
            pg_loss = pg_loss.view(-1) * pad_valid.float()
            progress_loss = pg_loss.mean()
            losses['progress_aux'] = self.args.pm_aux_loss_wt * progress_loss

        return losses


    def weighted_mask_loss(self, pred_masks, gt_masks):
        '''
        mask loss that accounts for weight-imbalance between 0 and 1 pixels
        '''
        bce = self.bce_with_logits(pred_masks, gt_masks)
        flipped_mask = self.flip_tensor(gt_masks)
        inside = (bce * gt_masks).sum() / (gt_masks).sum()
        outside = (bce * flipped_mask).sum() / (flipped_mask).sum()
        return inside + outside


    def flip_tensor(self, tensor, on_zero=1, on_non_zero=0):
        '''
        flip 0 and 1 values in tensor
        '''
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res


    def compute_metric(self, preds, data):
        '''
        compute f1 and extract match scores for output
        '''
        m = collections.defaultdict(list)
        for (task, _) in data:
            # try:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['low_actions']])
            m['action_low_f1'].append(compute_f1(label.lower(), preds[i]['action_low'].lower()))
            m['action_low_em'].append(compute_exact(label.lower(), preds[i]['action_low'].lower()))
            # except:
            #     print("KeyError in valid")
            #     pass
        return {k: sum(v)/len(v) for k, v in m.items()}
