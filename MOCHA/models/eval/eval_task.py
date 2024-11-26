import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
from eval import Eval
from env.thor_env import ThorEnv
import shutil
import torch
import constants
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import random

classes = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']

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

import math
def get_orientation(d):
    if d == 'left':
        h, v = -math.pi/2, 0.0
    elif d == 'up':
        h, v = 0.0, -math.pi/12
    elif d == 'down':
        h, v = 0.0, math.pi/12
    elif d == 'right':
        h, v = math.pi/2, 0.0
    else:
        h, v = 0.0, 0.0

    orientation = torch.cat([
        torch.cos(torch.ones(1)*(h)),
        torch.sin(torch.ones(1)*(h)),
        torch.cos(torch.ones(1)*(v)),
        torch.sin(torch.ones(1)*(v)),
    ]).unsqueeze(-1).unsqueeze(-1).repeat(1,7,7).unsqueeze(0).unsqueeze(0)

    return orientation

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

    # Left
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

# nameflag = False
def printing_log(*args):
    print(*args)

    new_args = list(args)
    # if nameflag == False:
    filename = 'new_logs/loop_break_0.3_thresh_val_unseen_latest_logs.txt'
        # flag = True

    with open(filename, 'a') as f:
        for ar in new_args:
            f.write(f'{ar}\n')


class EvalTask(Eval):
    '''
    evaluate overall task performance
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()
        count = 0
        while True:
            

            if task_queue.qsize() == 0:
                break

            task = task_queue.get()


            if os.path.exists(os.path.join('logs/success', task['task'], str(task['repeat_idx']))) \
                or os.path.exists(os.path.join('logs/failure', task['task'], str(task['repeat_idx']))):
                print("skipping:", os.path.join('logs/failure', task['task'], str(task['repeat_idx'])))
                continue


            try:
                traj = model['nav'].load_task_json(task)
                r_idx = task['repeat_idx']
                printing_log("Evaluating: %s" % (traj['root']))
                printing_log("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, r_idx, resnet, traj, args, lock, successes, failures, results, task)
            except Exception as e:
                import traceback
                traceback.print_exc()
                printing_log("Error: " + repr(e))
        # exit()
        # stop THOR
        env.stop()



    @classmethod
    def doManipulation(cls, total_actions, total_frames, action_high_name, traj_data, action_high_order, resnet, feat_nav, model, model_nav, eval_idx, maskrcnn, curr_image, lang_index, env, args, t, fails):

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
        if args.debug:
            printing_log(obj_name[0],  "in objects_present", obj_name[0] in objects_present)

        if obj_name[0] in objects_present:
            posi = objects_present.index(obj_name[0])
            scr = out['scores'][posi]
            man_action_success = []
            # err_list = []
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


                    if args.debug:
                        printing_log(action_high_name, action)
                    if action == cls.STOP_TOKEN:

                        if args.debug:
                            printing_log("\tpredicted STOP")
                        return True, t, fails, total_actions, total_frames

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

                    # debug
                    if args.debug:
                        printing_log("Pred: ", action_high_name, action, classes[pred_class])

                    # update prev action
                    prev_action = str(action)


                           
                    # if action not in cls.TERMINAL_TOKENS:
                    # use predicted action and mask (if provided) to interact with the env
                    total_actions.append(action + '_' + classes[pred_class])
                    curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                    curr_mask = (np.uint8(255*(mask>0.5)))
                    curr_mask = Image.fromarray(np.stack([curr_mask, curr_mask, curr_mask], 2))
                    total_frames.append(Image.blend(curr_image, curr_mask, 0.25))
                    
                    t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)

                    curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                    total_frames.append(curr_image)
                    total_actions.append(action + '_' + classes[pred_class])

                    vis_feat = resnet.featurize([curr_image], batch=1).unsqueeze(0)
                    od_score = ((feat1['frames'] - vis_feat)**2).sum().sqrt()

                    epsilon = 1
                    if od_score < epsilon:
                        return False, t, fails, total_actions, total_frames

                    if not t_success:
                        fails += 1

                    # next time-step
                    t_reward, t_done = env.get_transition_reward()
                    reward += t_reward
                    man_action_success.append(t_success)

                    # increment time index
                    t += 1

                    # prev_image = curr_image
                    m_prev_action = action

        t+=1
        return False, t, fails, total_actions, total_frames





    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures, results, task):


        import copy

        # reset model
        for mk in model.keys():
            model[mk].reset()

        # setup scene
        reward_type = 'dense'
        nav_traj_data = copy.deepcopy(traj_data)
        cls.setup_scene(env, (nav_traj_data), r_idx, args, reward_type=reward_type)

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

        pred_subgoal = torch.argmax(subgoal_mask*out_subgoal['out_sub'], dim=1)
        subgoals_to_complete = [model['nav'].vocab['action_high'].index2word(list(pred_subgoal.cpu().numpy()))]

        with open('subgoal_predictions.csv', 'a') as f:
            f.write(str(subgoals_to_complete)+'\n')
        printing_log(subgoals_to_complete)
        
        # return
        printing_log(pred_subgoal, len(pred_subgoal))
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
        feat = model['nav'].featurize([(nav_traj_data, False)], pred_subgoal.cpu().numpy(), objects2find, load_mask=True)

        # goal instr
        goal_instr = nav_traj_data['turk_annotations']['anns'][r_idx]['task_desc']

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
        fails = 0
        t = 0
        reward = 0
        lang_index=0
        max_lang_index = len(feat['lang_instr']['seq'][0])
        
        goal_satisfied=False

        st_oh = 0
        man_t = 0
        action_list = []
        vis_feats = []
        pred_actions = []
        loop_count = 0

        subgoal_running = 0
        sub_conversion_dict = {'PickupObject':'pickup', 'PutObject':'put', 'CleanObject':'clean', 'HeatObject':'heat', 'CoolObject':'cool', 'ToggleObject':'toggle', 'SliceObject':'slice'}

        total_frames = []
        total_actions = []
        total_objnav = []
        

        
        # total_actions.append('Start')
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
                panoramic_actions, imgs = get_panoramic_actions(env)
                curr_image_left, curr_image_right, curr_image_up, curr_image_down = imgs
                feat['frames_left'] = resnet.featurize([curr_image_left], batch=1).unsqueeze(0)
                feat['frames_right'] = resnet.featurize([curr_image_right], batch=1).unsqueeze(0)
                feat['frames_up'] = resnet.featurize([curr_image_up], batch=1).unsqueeze(0)
                feat['frames_down'] = resnet.featurize([curr_image_down], batch=1).unsqueeze(0)
          
            # forward model
            # if not double_Manipulation:

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

                if (sum(np.array(action_list) == 'RotateRight_90') + sum(np.array(action_list)=='RotateLeft_90')) == len(action_list):
                    action_mask[model['nav'].vocab['action_low'].word2index(['RotateLeft_90', 'RotateRight_90'])] = -1    
                    action = model['nav'].vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

                if (sum(np.array(action_list) == 'Manipulate')) > 3:
                    action_mask[model['nav'].vocab['action_low'].word2index(['Manipulate'])] = -1    
                    action = model['nav'].vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

            pred_actions.append(action)

            action_list.append(action)
            if args.debug:
                print(action)
            # num_manipulations
            if action == cls.MANIPULATE_TOKEN:
                subgoal_running+=1
                action_high = model['nav'].vocab['action_high'].index2word((feat['action_high_order'][0][subgoal_running].item()))

                man_success, t, fails, total_actions, total_frames = cls.doManipulation(total_actions, total_frames, action_high, copy.deepcopy(traj_data), pred_subgoal.cpu().numpy(), resnet, feat, model[sub_conversion_dict[action_high]], model['nav'], subgoal_running, maskrcnn, curr_image, lang_index, env, args, t, fails)
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
                        man_success, t, fails, total_actions, total_frames = cls.doManipulation(total_actions, total_frames, new_action_high, copy.deepcopy(traj_data), pred_subgoal.cpu().numpy(), resnet, feat, model[sub_conversion_dict[new_action_high]], model['nav'], subgoal_running, maskrcnn, curr_image, lang_index, env, args, t, fails)                        

                        if fails >= args.max_fails:
                            printing_log("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                            break

                        # if man_success:
                        subgoal_running+=1
                        new_action_high3 = model['nav'].vocab['action_high'].index2word((feat['action_high_order'][0][subgoal_running]))                            

                        if new_action_high3 != 'GotoLocation' and new_action_high3 != 'NoOp':
                            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                            vis_feat = resnet.featurize([curr_image], batch=1).unsqueeze(0)
                            feat['frames'] = vis_feat
                            man_success, t, fails, total_actions, total_frames = cls.doManipulation(total_actions, total_frames, new_action_high3, copy.deepcopy(traj_data), pred_subgoal.cpu().numpy(), resnet, feat, model[sub_conversion_dict[new_action_high3]], model['nav'], subgoal_running, maskrcnn, curr_image, lang_index, env, args, t, fails)                        

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


            # mask prediction
            mask = None

            # use predicted action and mask (if available) to interact with the env
            total_actions.append(action)
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            total_frames.append(curr_image)
            if not t_success:
                fails += 1

                if fails >= args.max_fails:
                    printing_log("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

            prev_vis_feat = vis_feat
            prev_action = action

        
        print(len(total_frames), len(total_actions))
        save_dir = 'logs/'
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            printing_log("Goal Reached")
            success = True
            save_dir += 'success/'
        else:
            save_dir += 'failure/'
        os.makedirs(os.path.join(save_dir, task['task']), exist_ok=True)
        os.makedirs(os.path.join(save_dir, task['task'], str(task['repeat_idx']) ))

        for f, a, k in zip(total_frames, total_actions, range(len(total_frames))):
            f.save(os.path.join(save_dir, task['task'], str(task['repeat_idx']), str(k) + '_' + a + '.png'))
            print(f'saving: ' + os.path.join(save_dir, task['task'], str(task['repeat_idx']), str(k) + '_' + a + '.png'))
        
        json_path = os.path.join(model['nav'].args.data, task['task'], '%s' % model['nav'].args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        shutil.copyfile(json_path, os.path.join(save_dir, task['task'], str(task['repeat_idx']), 'ann_%d.json' % task['repeat_idx']))

        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / (float(t) + 1e-4))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / (float(t) + 1e-4))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        lock.acquire()
        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward)}
        if success:
            successes.append(log_entry)
        else:
            failures.append(log_entry)

        # overall results
        results['all'] = cls.get_metrics(successes, failures)

        printing_log("-------------")
        printing_log("SR: %d/%d = %.5f" % (results['all']['success']['num_successes'],
                                    results['all']['success']['num_evals'],
                                    results['all']['success']['success_rate']))
        printing_log("PLW SR: %.5f" % (results['all']['path_length_weighted_success_rate']))
        printing_log("GC: %d/%d = %.5f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                    results['all']['goal_condition_success']['total_goal_conditions'],
                                    results['all']['goal_condition_success']['goal_condition_success_rate']))
        printing_log("PLW GC: %.5f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
        printing_log("-------------")

        # task type specific results
        task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                      'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                      'pick_and_place_with_movable_recep']
        for task_type in task_types:
            task_successes = [s for s in (list(successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                results[task_type] = cls.get_metrics(task_successes, task_failures)
            else:
                results[task_type] = {}

        lock.release()

    @classmethod
    def get_metrics(cls, successes, failures):
        '''
        compute overall succcess and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions)
        plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                  total_path_len_weight)
        plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                  total_path_len_weight)

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                        'total_goal_conditions': total_goal_conditions,
                                        'goal_condition_success_rate': pc}
        res['path_length_weighted_success_rate'] = plw_sr
        res['path_length_weighted_goal_condition_success_rate'] = plw_pc

        return res

    def create_stats(self):
            '''
            storage for success, failure, and results info
            '''
            self.successes, self.failures = self.manager.list(), self.manager.list()
            self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.nav_model_path)
        save_path = os.path.join(save_path, (os.path.basename(self.args.nav_model_path)).split('.')[0] + '_man_' + (os.path.basename(self.args.pickup_model_path)).split('.')[0] + '_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')        
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

