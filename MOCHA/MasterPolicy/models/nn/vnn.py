import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class SelfAttn(nn.Module):
    '''
    self-attention with learnable parameters
    '''

    def __init__(self, dhid):
        super().__init__()
        self.scorer = nn.Linear(dhid, 1)

    def forward(self, inp):
        scores = F.softmax(self.scorer(inp), dim=1)
        cont = scores.transpose(1, 2).bmm(inp).squeeze(1)
        return cont


class DotAttn(nn.Module):
    '''
    dot-attention (or soft-attention)
    '''

    def forward(self, inp, h):
        score = self.softmax(inp, h)
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        raw_score = inp.bmm(h.unsqueeze(2))
        score = F.softmax(raw_score, dim=1)
        return score


class ResnetVisualEncoder(nn.Module):
    '''
    visual encoder
    '''

    def __init__(self, dframe):
        super(ResnetVisualEncoder, self).__init__()
        self.dframe = dframe
        self.flattened_size = 64*7*7

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        return x


class MaskDecoder(nn.Module):
    '''
    mask decoder
    '''

    def __init__(self, dhid, pframe=300, hshape=(64,7,7)):
        super(MaskDecoder, self).__init__()
        self.dhid = dhid
        self.hshape = hshape
        self.pframe = pframe

        self.d1 = nn.Linear(self.dhid, hshape[0]*hshape[1]*hshape[2])
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(16)
        self.dconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dconv1 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.d1(x))
        x = x.view(-1, *self.hshape)

        x = self.upsample(x)
        x = self.dconv3(x)
        x = F.relu(self.bn2(x))

        x = self.upsample(x)
        x = self.dconv2(x)
        x = F.relu(self.bn1(x))

        x = self.dconv1(x)
        x = F.interpolate(x, size=(self.pframe, self.pframe), mode='bilinear')

        return x


class ScaledDotAttn(nn.Module):
    def __init__(self, dim_key_in=1024, dim_key_out=128, dim_query_in=1024 ,dim_query_out=128):
        super().__init__()
        self.fc_key = nn.Linear(dim_key_in, dim_key_out)
        self.fc_query = nn.Linear(dim_query_in, dim_query_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, value, h): # key: lang_feat_t_instr, query: h_tm1_instr
        key = F.relu(self.fc_key(value))
        query = F.relu(self.fc_query(h)).unsqueeze(-1)

        scale_1 = np.sqrt(key.shape[-1])
        scaled_dot_product = torch.bmm(key, query) / scale_1
        softmax = self.softmax(scaled_dot_product)
        element_wise_product = value*softmax
        weighted_lang_t_instr = torch.sum(element_wise_product, dim=1)

        return weighted_lang_t_instr, softmax.squeeze(-1)


class DynamicConvLayer(nn.Module):
    def __init__(self, dhid=512, dframe=512):
        super().__init__()
        self.head1 = nn.Linear(dhid, dframe)
        self.head2 = nn.Linear(dhid, dframe)
        self.head3 = nn.Linear(dhid, dframe)
        self.filter_activation = nn.Tanh()

    def forward(self, frame, weighted_lang_t_instr):
        """ dynamic convolutional filters """
        df1 = self.head1(weighted_lang_t_instr)
        df2 = self.head2(weighted_lang_t_instr)
        df3 = self.head3(weighted_lang_t_instr)
        dynamic_filters = torch.stack([df1, df2, df3]).transpose(0, 1)
        dynamic_filters = self.filter_activation(dynamic_filters)
        dynamic_filters = F.normalize(dynamic_filters, p=2, dim=-1)

        """ attention map """
        frame = frame.view(frame.size(0), frame.size(1), -1)
        scale_2 = np.sqrt(frame.shape[1]) #torch.sqrt(torch.tensor(frame.shape[1], dtype=torch.double))
        attention_map = torch.bmm(frame.transpose(1,2), dynamic_filters.transpose(-1, -2)) / scale_2
        attention_map = attention_map.reshape(attention_map.size(0), -1)

        return attention_map
######################################################################################################################

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=512, type="bn_relu_drop"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn" or self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.bn(x)
        if self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.relu(x)
        if self.type == "bn_relu_drop":
            x = self.dropout(x)
        return x

class ConvFrameMaskDecoderProgressMonitorPanoramicConcat(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, vocab, args, emb, dframe, dhid, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False, panoramic=False, orientation=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.panoramic = panoramic
        self.orientation = orientation
        self.vocab = vocab
        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        # self.cell_goal = nn.LSTMCell(dhid+dframe*5+demb, dhid)
        self.cell_instr = nn.LSTMCell(dhid+dframe*5+demb+args.dhid, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe*5+demb+args.dhid, demb)

        self.teacher_forcing = teacher_forcing
        # self.h_tm1_fc_goal = nn.Linear(dhid, dhid)
        self.h_tm1_fc_instr = nn.Linear(dhid, dhid)

        self.subgoal = nn.Linear(dhid+dhid+dframe*5+demb+args.dhid, 1)
        self.progress = nn.Linear(dhid+dhid+dframe*5+demb+args.dhid, 1)

        nn.init.uniform_(self.go, -0.1, 0.1)

        self.scale_dot_attn = ScaledDotAttn(dhid, 128, dhid, 128)
        self.dynamic_conv = DynamicConvLayer(dhid, 512 + 4 if orientation else 512)

        self.enc_instr = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att_instr = SelfAttn(args.dhid*2)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)

        self.object_enc = feat_bootleneck(args.demb, args.dhid)

    def step(self, enc_obj, enc_instr, frame, frame_left, frame_up, frame_down, frame_right, e_t, state_tm1_instr):
        # Panoramic views
        vis_feat_t_left = frame_left
        vis_feat_t_up = frame_up
        vis_feat_t_front = frame
        vis_feat_t_down = frame_down
        vis_feat_t_right = frame_right

        # previous decoder hidden state (instr decoder)
        h_tm1_instr = state_tm1_instr[0]

        # encode vision and lang feat (instr decoder)
        lang_feat_t_instr = enc_instr # language is encoded once at the start

        # scaled dot product attention
        weighted_lang_t_instr, lang_attn_t_instr = self.scale_dot_attn(lang_feat_t_instr, h_tm1_instr)


        vis_feat_t_instr_left = self.dynamic_conv(vis_feat_t_left, weighted_lang_t_instr)
        vis_feat_t_instr_up = self.dynamic_conv(vis_feat_t_up, weighted_lang_t_instr)
        vis_feat_t_instr_front = self.dynamic_conv(vis_feat_t_front, weighted_lang_t_instr)
        vis_feat_t_instr_down = self.dynamic_conv(vis_feat_t_down, weighted_lang_t_instr)
        vis_feat_t_instr_right = self.dynamic_conv(vis_feat_t_right, weighted_lang_t_instr)
        vis_feat_t_instr = torch.cat([
            vis_feat_t_instr_left, vis_feat_t_instr_up, vis_feat_t_instr_front, vis_feat_t_instr_down, vis_feat_t_instr_right
        ], dim=1)

        inp_t_instr = torch.cat([vis_feat_t_instr, weighted_lang_t_instr, e_t, enc_obj], dim=1)
        inp_t_instr = self.input_dropout(inp_t_instr)

        state_t_instr = self.cell_instr(inp_t_instr, state_tm1_instr)
        state_t_instr = [self.hstate_dropout(x) for x in state_t_instr]
        h_t_instr, _ = state_t_instr[0], state_t_instr[1]

        # decode action (instr decoder)
        cont_t_instr = torch.cat([h_t_instr, inp_t_instr], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t_instr))
        action_t = action_emb_t.mm(self.emb.weight.t())

        # predict subgoals completed and task progress
        subgoal_t = F.sigmoid(self.subgoal(cont_t_instr))
        progress_t = F.sigmoid(self.progress(cont_t_instr))

        return action_t, state_t_instr, lang_attn_t_instr, subgoal_t, progress_t

    def forward(self, emb_objnav, emb_instr, frames, frames_left, frames_up, frames_down, frames_right, gold=None, max_decode=150): #, state_0_instr=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        
        batch = len(emb_instr)
        e_t = self.go.repeat(batch, 1)

        actions = []
        masks = []
        attn_scores_goal = []
        attn_scores_instr = []
        subgoals = []
        progresses = []

        lang_index = np.zeros((batch, ), dtype=int)
        curr_emb = torch.stack([lan[lang_index[i]] for i, lan in enumerate(emb_instr)], dim=0)
        cont_instr, enc_instr = self.encode_lang_instr(curr_emb)
        state_t_instr = cont_instr, torch.zeros_like(cont_instr)

        stop_action_indices = (gold==self.vocab['action_low'].word2index('Manipulate', train=False)).cpu().numpy().astype(int)
        sai_sum = stop_action_indices.sum(1)

        curr_obj = torch.stack([lan[lang_index[i]] for i, lan in enumerate(emb_objnav)], dim=0)
        enc_obj = self.object_enc(curr_obj)


        with torch.autograd.set_detect_anomaly(True):
            for t in range(max_t):
                lang_index += stop_action_indices[:,t]

                action_t, state_t_instr, attn_score_t_instr, subgoal_t, progress_t = self.step(enc_obj, enc_instr, frames[:, t], frames_left[:, t], frames_up[:, t], frames_down[:, t], frames_right[:, t], e_t, state_t_instr)
                actions.append(action_t)
                attn_scores_instr.append(attn_score_t_instr)
                subgoals.append(subgoal_t)
                progresses.append(progress_t)

                # find next emb
                if self.teacher_forcing and self.training:
                    w_t = gold[:, t]
                else:
                    w_t = action_t.max(1)[1]
                e_t = self.emb(w_t)

                if 1 in stop_action_indices[:,t]:
                    ids2change = (stop_action_indices[:,t] == 1).nonzero()[0]
                    
                    
                    
                    if np.all(lang_index[ids2change]>=sai_sum[ids2change]):
                        pass
                    else:
                        excd_ids = np.where(lang_index>=sai_sum)[0]
                        if len(excd_ids) > 0:
                            lang_index[excd_ids] -= 1

                        curr_emb = torch.stack([lan[lang_index[i]] for i, lan in enumerate(emb_instr)], dim=0)
                        cont_instr, enc_instr = self.encode_lang_instr(curr_emb)

                        curr_obj = torch.stack([lan[lang_index[i]] for i, lan in enumerate(emb_objnav)], dim=0)
                        enc_obj = self.object_enc(curr_obj)

            results = {
                'out_action_low': torch.stack(actions, dim=1),
                'out_attn_scores_instr': torch.stack(attn_scores_instr, dim=1),
                'out_subgoal': torch.stack(subgoals, dim=1),
                'out_progress': torch.stack(progresses, dim=1),
                'state_t_instr': state_t_instr,
            }
            return results

    def encode_lang_instr(self, lang_goal_instr):
        '''
        encode goal+instr language
        '''
        emb_lang_goal_instr = lang_goal_instr
        self.lang_dropout(emb_lang_goal_instr)
        enc_lang_goal_instr, _ = self.enc_instr(emb_lang_goal_instr)
        self.lang_dropout(enc_lang_goal_instr)
        cont_lang_goal_instr = self.enc_att_instr(enc_lang_goal_instr)

        return cont_lang_goal_instr, enc_lang_goal_instr
