import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import json
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from eval_task import EvalTask
from env.thor_env import ThorEnv
import torch.multiprocessing as mp

import torch
import constants
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn

classes = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']


import math
import random

def loop_detection(vis_feats, actions, window_size=10):

    # not enough vis feats for loop detection
    if len(vis_feats) < window_size*2:
        return False, None

    nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
    random.shuffle(nav_actions)

    start_idx = len(vis_feats) - 1

    for end_idx in range(start_idx - window_size, window_size - 1, -1):
        if (vis_feats[start_idx] == vis_feats[end_idx]).all():
            if all((vis_feats[start_idx-i] == vis_feats[end_idx-i]).all() for i in range(window_size)):
                return True, nav_actions[1] if actions[end_idx] == nav_actions[0] else nav_actions[0]

    return False, None

def get_panoramic_views(env):
    horizon = np.round(env.last_event.metadata['agent']['cameraHorizon'])
    rotation = env.last_event.metadata['agent']['rotation']
    position = env.last_event.metadata['agent']['position']

    # Left
    env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotateOnTeleport": True,
        "rotation": (rotation['y'] + 270.0) % 360,
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_left = Image.fromarray(np.uint8(env.last_event.frame))

    # Right
    env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotateOnTeleport": True,
        "rotation": (rotation['y'] + 90.0) % 360,
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_right = Image.fromarray(np.uint8(env.last_event.frame))

    # Up
    env.step({
        "action": "TeleportFull",
        "horizon": np.round(horizon - constants.AGENT_HORIZON_ADJ),
        "rotateOnTeleport": True,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_up = Image.fromarray(np.uint8(env.last_event.frame))

    # Down
    env.step({
        "action": "TeleportFull",
        "horizon": np.round(horizon + constants.AGENT_HORIZON_ADJ),
        "rotateOnTeleport": True,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_down = Image.fromarray(np.uint8(env.last_event.frame))

    # Back to original
    env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotateOnTeleport": True,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })

    return curr_image_left, curr_image_right, curr_image_up, curr_image_down


def get_panoramic_actions(env):
    action_pairs = [
        ['RotateLeft_90', 'RotateRight_90'],
        ['RotateRight_90', 'RotateLeft_90'],
        ['LookUp_15', 'LookDown_15'],
        ['LookDown_15', 'LookUp_15'],
    ]
    imgs = []
    actions = []

    curr_image = Image.fromarray(np.uint8(env.last_event.frame))

    for a1, a2 in action_pairs:
        t_success, _, _, err, api_action = env.va_interact(a1, interact_mask=None, smooth_nav=False)
        actions.append(a1)
        imgs.append(Image.fromarray(np.uint8(env.last_event.frame)))
        #if len(err) == 0:
        if curr_image != imgs[-1]:
            t_success, _, _, err, api_action = env.va_interact(a2, interact_mask=None, smooth_nav=False)
            actions.append(a2)
        else:
            #print(err)
            printing_log('Error while {}'.format(a1))
    return actions, imgs



def printing_log(*kargs):
    print(*kargs)

    new_args = list(kargs)
    with open("leaderboard_3_woGoBack_earlyManReturn_loop_break_scr0.3_logs_sep2_11_30am.txt", 'a') as f:
        for ar in new_args:
            f.write(f'{ar}\n')



class Leaderboard(EvalTask):
    '''
    dump action-sequences for leaderboard eval
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, splits, seen_actseqs, unseen_actseqs):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                traj = model['nav'].load_task_json(task)
                r_idx = task['repeat_idx']
                printing_log("Evaluating: %s" % (traj['root']))
                printing_log("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, r_idx, resnet, traj, args, lock, splits, seen_actseqs, unseen_actseqs)
            except Exception as e:
                import traceback
                traceback.print_exc()
                printing_log("Error: " + repr(e))

        # stop THOR
        env.stop()


    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, splits, seen_actseqs, unseen_actseqs):
        
        import copy
        # reset model
        for mk in model.keys():
            model[mk].reset()

        # setup scene
        reward_type = 'dense'
        nav_traj_data = copy.deepcopy(traj_data)
        cls.setup_scene(env, traj_data, r_idx, args)

        feat_subgoal = model['subgoal'].featurize([(copy.deepcopy(traj_data), False)], load_mask=True)
        out_subgoal = model['subgoal'].forward(feat_subgoal)
        # out_subgoal = out_obj['out_sub']
        subgoal_mask = torch.ones(len(model['subgoal'].vocab['action_high']), dtype=torch.float).cuda()
        subgoal_mask[model['subgoal'].vocab['action_high'].word2index(['<<pad>>', '<<seg>>', '<<stop>>', 'diningtable', 'knife', 'lettuce', 'fridge', 
                                                           'countertop', 'candle', 'cabinet', 'toilet', 'egg', 'microwave',  
                                                           'sinkbasin', 'spraybottle', 'stoveburner', 'kettle', 'coffeetable', 
                                                           'keychain', 'sofa', 'tomato', 'garbagecan', 'sidetable', 'alarmclock', 'desk', 'box', 
                                                           'spatula', 'spoon', 'drawer', 'dishsponge', 'butterknife', 'cup', 'floorlamp', 
                                                           'bathtubbasin', 'cart', 'pot', 'mug', 'shelf', 'toiletpaper', 'potato', 
                                                           'creditcard', 'armchair', 'remotecontrol', 'fork', 'pan', 'apple', 'ottoman', 
                                                           'toiletpaperhanger', 'coffeemachine', 'cellphone', 'safe', 'pen', 'dresser', 'pencil', 
                                                           'soapbar', 'basketball', 'desklamp', 'tissuebox', 'wateringcan', 'ladle', 'plate', 'statue', 
                                                           'bread', 'watch', 'peppershaker', 'cd', 'bed', 'pillow', 'cloth', 'vase', 'book', 'bowl', 
                                                           'soapbottle', 'handtowelholder', 'handtowel', 'winebottle', 'newspaper', 'tennisracket', 
                                                           'saltshaker', 'laptop', 'glassbottle', 'plunger', 'baseballbat', ''])] = -1
        # preds = F.softmax(out)
        pred_subgoal = torch.argmax(subgoal_mask*out_subgoal['out_sub'], dim=1)
        subgoals_to_complete = [model['nav'].vocab['action_high'].index2word(list(pred_subgoal.cpu().numpy()))]
        # subgoals_to_complete_orig = [model['subgoal'].vocab['action_high'].index2word(list(pred_subgoal.cpu().numpy()))]
        printing_log(subgoals_to_complete)
        # return
        # print(pred_subgoal, len(pred_subgoal))
        if (len(pred_subgoal) % 2) :
            for iii in range(len(pred_subgoal)-1):
                if not (iii % 2):
                    pred_subgoal[iii] = pred_subgoal[0] 


        if pred_subgoal[-2].item() == model['nav'].vocab['action_high'].word2index('GotoLocation', train=False):
            pred_subgoal[-2] = model['nav'].vocab['action_high'].word2index('PutObject', train=False)

            printing_log("changes", [model['nav'].vocab['action_high'].index2word(list(pred_subgoal.cpu().numpy()))])
        # exit()

        feat_obj = model['object'].featurize([(copy.deepcopy(traj_data), False)], pred_subgoal.cpu().numpy(), load_mask=True)
        out_obj = model['object'].forward(feat_obj)
        out_obj = out_obj['out_obj']     # classes.index need conversion to value corresponding to nn.embedding 
        pred_obj = torch.argmax(out_obj, dim=1)
        objects2find = [classes[o.item()] for o in pred_obj]


        # extract language features
        feat = model['nav'].featurize([(nav_traj_data, False)], pred_subgoal.cpu().numpy(), objects2find,  load_mask=False)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        maskrcnn = maskrcnn_resnet50_fpn(num_classes=119)
        maskrcnn.eval()
        maskrcnn.load_state_dict(torch.load('weight_maskrcnn.pt'))
        maskrcnn = maskrcnn.cuda()

        prev_vis_feat = None
        prev_action = None
        nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
        man_actions = ['PickupObject', 'SliceObject', 'OpenObject', 'PutObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff', '<<stop>>', '<<pad>>', '<<seg>>']
        manipulate_action = ['Manipulate']

        prev_class = 0
        prev_center = torch.zeros(2)

        done, success = False, False
        actions = list()
        fails = 0
        t = 0
        lang_index=0
        max_lang_index = len(feat['lang_instr']['seq'][0])
        man_t = 0
        action_list = []

        subgoal_running = 0
        sub_conversion_dict = {'PickupObject':'pickup', 'PutObject':'put', 'CleanObject':'clean', 'HeatObject':'heat', 'CoolObject':'cool', 'ToggleObject':'toggle', 'SliceObject':'slice'}
        vis_feats = []
        pred_actions = []
        loop_count = 0
        while not done:
            # break if max_steps reached
            if t >= args.max_steps:
                printing_log("max steps exceeded")
                break

            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            vis_feat = resnet.featurize([curr_image], batch=1).unsqueeze(0)
            feat['frames'] = vis_feat
            vis_feats.append(vis_feat)

            if model['nav'].panoramic:
                #curr_image_left, curr_image_right, curr_image_up, curr_image_down = get_panoramic_views(env)
                panoramic_actions, imgs = get_panoramic_actions(env)
                curr_image_left, curr_image_right, curr_image_up, curr_image_down = imgs
                feat['frames_left'] = resnet.featurize([curr_image_left], batch=1).unsqueeze(0)
                feat['frames_right'] = resnet.featurize([curr_image_right], batch=1).unsqueeze(0)
                feat['frames_up'] = resnet.featurize([curr_image_up], batch=1).unsqueeze(0)
                feat['frames_down'] = resnet.featurize([curr_image_down], batch=1).unsqueeze(0)
                #t += 8
                #panoramic_actions = get_panoramic_actions(env)
                for pa in panoramic_actions:
                    actions.append({"action": pa[:-3], "forceAction": True})
                t += len(panoramic_actions)
                if t >= args.max_steps:
                    break

            m_out = model['nav'].step(feat, lang_index)

            m_pred = feat['out_action_low'].max(2)[1].tolist()
            
            dist_action = m_out['out_action_low'][0][0].detach().cpu()
            dist_action = F.softmax(dist_action, dim=-1)
            action_mask = torch.ones(len(model['nav'].vocab['action_low']), dtype=torch.float)

            action_mask[model['nav'].vocab['action_low'].word2index(man_actions)] = -1
            action_mask[model['nav'].vocab['action_low'].word2index(cls.STOP_TOKEN)] = -1
            if t<(man_t+3): 
                action_mask[model['nav'].vocab['action_low'].word2index(cls.MANIPULATE_TOKEN)] = -1
            
            action = model['nav'].vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

            if len(action_list) > 19:
                action_list.pop(0)
            
                if (sum(np.array(action_list) == 'LookUp_15') + sum(np.array(action_list)=='LookDown_15') + sum(np.array(action_list)=='Manipulate')) == len(action_list):
                    action_mask[model['nav'].vocab['action_low'].word2index(cls.MANIPULATE_TOKEN)] = -1
                    action_mask[model['nav'].vocab['action_low'].word2index(['LookDown_15', 'LookUp_15'])] = -1    
                    action = model['nav'].vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

                if (sum(np.array(action_list) == 'RotateRight_90') + sum(np.array(action_list)=='RotateLeft_90')) == 20:
                    action_mask[model['nav'].vocab['action_low'].word2index(['RotateLeft_90', 'RotateRight_90'])] = -1    
                    action = model['nav'].vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

                if (sum(np.array(action_list) == 'Manipulate')) > 3:
                    action_mask[model['nav'].vocab['action_low'].word2index(['Manipulate'])] = -1    
                    action = model['nav'].vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))
            
            action_list.append(action)
            pred_actions.append(action)
            # print(action)

            if action == cls.MANIPULATE_TOKEN:
                subgoal_running+=1
                action_high = model['nav'].vocab['action_high'].index2word((feat['action_high_order'][0][subgoal_running].item()))

                man_success, t, fails, actions = cls.doManipulation(actions, copy.deepcopy(traj_data), pred_subgoal.cpu().numpy(), resnet, feat, model[sub_conversion_dict[action_high]], model['nav'], subgoal_running, maskrcnn, curr_image, lang_index, env, args, t, fails)
                if fails >= args.max_fails:
                    printing_log("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break


                if man_success:
                    subgoal_running+=1
                    new_action_high = model['nav'].vocab['action_high'].index2word((feat['action_high_order'][0][subgoal_running]))

                    if new_action_high != 'GotoLocation' and new_action_high != 'NoOp':
                        curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                        vis_feat = resnet.featurize([curr_image], batch=1).unsqueeze(0)
                        feat['frames'] = vis_feat
                        man_success, t, fails, actions = cls.doManipulation(actions, copy.deepcopy(traj_data), pred_subgoal.cpu().numpy(), resnet, feat, model[sub_conversion_dict[new_action_high]], model['nav'], subgoal_running, maskrcnn, curr_image, lang_index, env, args, t, fails)                        

                        if fails >= args.max_fails:
                            printing_log("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                            break
                        subgoal_running+=1
                        new_action_high3 = model['nav'].vocab['action_high'].index2word((feat['action_high_order'][0][subgoal_running]))                            

                        if new_action_high3 != 'GotoLocation' and new_action_high3 != 'NoOp':
                            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                            vis_feat = resnet.featurize([curr_image], batch=1).unsqueeze(0)
                            feat['frames'] = vis_feat
                            man_success, t, fails, actions = cls.doManipulation(actions, copy.deepcopy(traj_data), pred_subgoal.cpu().numpy(), resnet, feat, model[sub_conversion_dict[new_action_high3]], model['nav'], subgoal_running, maskrcnn, curr_image, lang_index, env, args, t, fails)                        

                            if fails >= args.max_fails:
                                printing_log("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                                break
                            subgoal_running+=1
                            lang_index+=1
                        else:
                            lang_index+=1
                        
                    else:
                        lang_index+=1
                else:
                    subgoal_running-=1


                if lang_index == max_lang_index:
                   
                    break
                
                prev_action = cls.MANIPULATE_TOKEN
                t+=1
                man_t = t           
                continue


            isLoop, rand_action = loop_detection(vis_feats, pred_actions, 10)
            if isLoop:
                action = rand_action
                loop_count += 1
                printing_log("loop_count", loop_count)

            if prev_vis_feat != None:
                od_score = ((prev_vis_feat - vis_feat)**2).sum().sqrt()
                epsilon = 1
                if od_score < epsilon:
                    dist_action = m_out['out_action_low'][0][0].detach().cpu()
                    dist_action = F.softmax(dist_action, dim=-1)
                    action_mask = torch.ones(len(model['nav'].vocab['action_low']), dtype=torch.float)
                    action_mask[model['nav'].vocab['action_low'].word2index(prev_action)] = -1
                    action_mask[model['nav'].vocab['action_low'].word2index(man_actions)] = -1
                    action_mask[model['nav'].vocab['action_low'].word2index(cls.STOP_TOKEN)] = -1
                    action_mask[model['nav'].vocab['action_low'].word2index(cls.MANIPULATE_TOKEN)] = -1
                    action = model['nav'].vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

                
            mask = None

            t_success, _, _, err, api_action = env.va_interact(action, interact_mask=None, smooth_nav=False)

            if not t_success:
                fails += 1
                if fails >= args.max_fails:
                    printing_log("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # save action
            if api_action is not None:
                actions.append(api_action)

            # next time-step
            t += 1

            prev_vis_feat = vis_feat
            prev_action = action

        # actseq
        seen_ids = [t['task'] for t in splits['tests_seen']]
        actseq = {traj_data['task_id']: actions}

        # log action sequences
        lock.acquire()

        if traj_data['task_id'] in seen_ids:
            seen_actseqs.append(actseq)
        else:
            unseen_actseqs.append(actseq)

        lock.release()


    @classmethod
    def doManipulation(cls, actions, traj_data, action_high_order, resnet, feat_nav, model, model_nav, eval_idx, maskrcnn, curr_image, lang_index, env, args, t, fails):

        model.reset()

        # setup scene
        reward_type = 'dense'
       
        obj_class = feat_nav['objnav'][0][lang_index].unsqueeze(0).mm(model_nav.emb_objnav.weight.t()).max(1)[1].tolist()
        obj_name = model_nav.vocab['objnav'].index2word(obj_class)

        prev_vis_feat = None
        m_prev_action = None
        nav_actions1 = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15', '<<seg>>', '<<pad>>']
        
        prev_class = 0
        prev_center = torch.zeros(2)

        # extract language features
        feat1 = model.featurize([(traj_data, False)], action_high_order, load_mask=False)

        # previous action for teacher-forcing during expert execution (None is used for initialization)
        prev_action = None

        done, subgoal_success = False, False
        # fails = 0
        # t = 0
        reward = 0

        

        with torch.no_grad():
            out = maskrcnn([to_tensor(curr_image).cuda()])[0]
            for k in out:
                out[k] = out[k].detach().cpu()

        objects_present = [classes[o].lower() for o in out['labels']]
        
        if obj_name[0] in objects_present:
            posi = objects_present.index(obj_name[0])
            scr = out['scores'][posi]
            man_action_success = []
            if scr > 0.3:
                prev_class = 0
                while not done:

                    # extract visual feats
                    curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                    feat1['frames'] = resnet.featurize([curr_image], batch=1).unsqueeze(0)


                    # forward model
                    m_out = model.step(feat1, eval_idx)
                    m_pred = model.extract_preds(m_out, [(traj_data, False)], feat1, clean_special_tokens=False)
                    m_pred = list(m_pred.values())[0]

                    dist_action = m_out['out_action_low'][0][0].detach().cpu()
                    dist_action = F.softmax(dist_action, dim=-1)
                    action_mask = torch.ones(len(model.vocab['action_low']), dtype=torch.float)

                    action_mask[model.vocab['action_low'].word2index(nav_actions1)] = -1
                    action = model.vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))
                    
                    # print(action)
                    if action == cls.STOP_TOKEN:
                        return True, t, fails, actions

                    # mask generation
                    mask = None
                    if model.has_interaction(action):
                        class_dist = m_pred['action_low_mask'][0]
                        pred_class = np.argmax(class_dist)

                        with torch.no_grad():
                            out = maskrcnn([to_tensor(curr_image).cuda()])[0]
                            for k in out:
                                out[k] = out[k].detach().cpu()

                        if sum(out['labels'] == pred_class) == 0:
                            mask = np.zeros((300,300))
                        else:
                            masks = out['masks'][out['labels'] == np.argmax(class_dist)].detach().cpu()
                            scores = out['scores'][out['labels'] == np.argmax(class_dist)].detach().cpu()
                        
                            if prev_class != pred_class:
                                scores, indices = scores.sort(descending=True)
                                masks = masks[indices]
                                prev_class = pred_class
                                prev_center = masks[0].squeeze(dim=0).nonzero().double().mean(dim=0)
                            else:
                                cur_centers = torch.stack([m.nonzero().double().mean(dim=0) for m in masks.squeeze(dim=1)])
                                distances = ((cur_centers - prev_center)**2).sum(dim=1)
                                distances, indices = distances.sort()
                                masks = masks[indices]
                                prev_center = cur_centers[0]

                            mask = np.squeeze(masks[0].numpy(), axis=0)

                    # update prev action
                    prev_action = str(action)

                           
                    # use predicted action and mask (if provided) to interact with the env
                    t_success, _, _, err, api_action = env.va_interact(action, interact_mask=mask)

                    if api_action is not None:
                        actions.append(api_action)

                    curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                    vis_feat = resnet.featurize([curr_image], batch=1).unsqueeze(0)
                    od_score = ((feat1['frames'] - vis_feat)**2).sum().sqrt()
                    # print(od_score)
                    epsilon = 1
                    if od_score < epsilon:
                        return False, t, fails, actions

                    if not t_success:
                        fails += 1
                    man_action_success.append(t_success)

                    # increment time index
                    t += 1

                    m_prev_action = action

        t+=1
        return False, t, fails, actions



    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        printing_log("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        seen_files, unseen_files = self.splits['tests_seen'], self.splits['tests_unseen']

        # add seen trajectories to queue
        for traj in seen_files:
            task_queue.put(traj)

        # add unseen trajectories to queue
        for traj in unseen_files:
            task_queue.put(traj)

        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        # start threads
        threads = []
        lock = self.manager.Lock()
        for k in self.model.keys():
            self.model[k].test_mode = True

        for n in range(self.args.num_threads):
            thread = mp.Process(target=self.run, args=(self.model, self.resnet, task_queue, self.args, lock,
                                                       self.splits, self.seen_actseqs, self.unseen_actseqs))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        # save
        self.save_results()

    def create_stats(self):
        '''
        storage for seen and unseen actseqs
        '''
        self.seen_actseqs, self.unseen_actseqs = self.manager.list(), self.manager.list()

    def save_results(self):
        '''
        save actseqs as JSONs
        '''
        results = {'tests_seen': list(self.seen_actseqs),
                   'tests_unseen': list(self.unseen_actseqs)}

        save_path = os.path.dirname(self.args.nav_model_path)
        save_path = os.path.join(save_path, args.dout + '_tests_actseqs_dump_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)


if __name__ == '__main__':
    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--splits', type=str, default="data/splits/oct21.json")
    parser.add_argument('--data', type=str, default="data/json_feat_2.1.0")
    parser.add_argument('--dout', type=str, default='10')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=True)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--eval_split', type=str, choices=['train', 'valid_seen', 'valid_unseen'])
    parser.add_argument('--nav_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--pickup_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--put_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--cool_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--heat_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--toggle_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--slice_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--clean_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--object_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--subgoal_model_path', type=str, default="exp/pretrained/pretrained.pth")

    parser.add_argument('--nav_model', type=str, default='models.model.seq2seq_im_mask')
    parser.add_argument('--sub_model', type=str, default='models.model.seq2seq_im_mask_sub')
    parser.add_argument('--object_model', type=str, default='models.model.seq2seq_im_mask_obj')
    parser.add_argument('--subgoal_pred_model', type=str, default='models.model.seq2seq_im_mask_subgoal_pred')
    parser.add_argument('--logs', type=str, required=True)

    # parse arguments
    args = parser.parse_args()

    # fixed settings (DO NOT CHANGE)
    args.max_steps = 1000
    args.max_fails = 10

    # leaderboard dump
    eval = Leaderboard(args, manager)

    # start threads
    eval.spawn_threads()
