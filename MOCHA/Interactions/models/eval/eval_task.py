import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
from eval import Eval
from env.thor_env import ThorEnv

import torch
import constants
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import random

classes = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']


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
            print('Error while {}'.format(a1))
    return actions, imgs


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

            # if count < 23:
            #     traj = model.load_task_json(task)
            #     r_idx = task['repeat_idx']
            #     print("Skipping: %s" % (traj['root']))
            #     count+=1
            #     continue

            try:
                traj = model.load_task_json(task)
                r_idx = task['repeat_idx']
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, r_idx, resnet, traj, args, lock, successes, failures, results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()


    @classmethod
    def doManipulation(cls, feat, model, maskrcnn, curr_image, alow_m, alow_mask, lang_index, env, args, t, fails):


        # if action == cls.MANIPULATE_TOKEN:
        obj_class = feat['objnav'][0][lang_index].unsqueeze(0).mm(model.emb_objnav.weight.t()).max(1)[1].tolist()
        obj_name = model.vocab['objnav'].index2word(obj_class)

        with torch.no_grad():
            out = maskrcnn([to_tensor(curr_image).cuda()])[0]
            for k in out:
                out[k] = out[k].detach().cpu()

        objects_present = [classes[o].lower() for o in out['labels']]
        
        if obj_name[0] in objects_present:
            posi = objects_present.index(obj_name[0])
            scr = out['scores'][posi]
            # print("posi, scr", posi, scr)
            man_action_success = []
            err_list = []
            if scr > 0.3:
                # man_action_success = []
                
                # man_prev_vis_feat = None 
                prev_class = 0                     
                for man_action, pred_class in zip(alow_m[lang_index], alow_mask[lang_index]):
                    print("man, msk", man_action, classes[pred_class])

                    man_curr_image = Image.fromarray(np.uint8(env.last_event.frame))

                    with torch.no_grad():
                        out = maskrcnn([to_tensor(man_curr_image).cuda()])[0]
                        for k in out:
                            out[k] = out[k].detach().cpu()
                            
                    if sum(out['labels'] == pred_class) == 0:
                        mask = np.zeros((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
                    else:
                        masks = out['masks'][out['labels'] == pred_class].detach().cpu()
                        scores = out['scores'][out['labels'] == pred_class].detach().cpu()

                    # masks = out['masks'][out['labels'] == pred_class].detach().cpu()
                    # scores = out['scores'][out['labels'] == pred_class].detach().cpu()

                        # Instance selection based on the minimum distance between the prev. and cur. instance of a same class.
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
                    t_success, _, _, err, _ = env.va_interact(man_action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                    # inventory_objects = env.last_event.metadata['inventoryObjects'][0]['objectType']
                    # if man_action=='PickupObject' and classes[] not in inventory_objects
                    err_list.append(err)
                    man_action_success.append(t_success)
                    # print("t_success, err", t_success, err)
                    t+=1

                    # if np.all(np.array(err_list[-5:] == err_list[-1])):
                    #     print("same error repeating")
                    #     break

                    if not t_success:
                        fails += 1
                        # print(fails)
                        # if fails >= args.max_fails:
                        #     print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                        #     break

                    
            if np.all(np.array(man_action_success)) and man_action_success!=[]:
                return True, t, fails
            else:
                return False, t, fails

        return False, t, fails
                    # lang_index+=1
                    # print("\nlang_index", lang_index)                        




    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures, results):
        # reset model
        model.reset()

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # extract language features
        feat = model.featurize([(traj_data, False)], load_mask=True)

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
        fails = 0
        t = 0
        reward = 0
        lang_index=0
        max_lang_index = len(feat['lang_instr']['seq'][0])
        # print(max_lang_index)
        # exit()
        goal_satisfied=False


        # print(feat['action_low_manip'][0].tolist(), model.vocab['action_low'].index2word(feat['action_low_manip'][0].tolist()))
        # print(feat['action_low_mask_label'].tolist(), [classes[o].lower() for o in feat['action_low_mask_label'].tolist()])
        # print(feat['obj_high_indices'][0].tolist())
        # st_oh = feat['obj_high_indices'][0].tolist()[0]
        st_oh = 0
        alow_m = []
        alow_mask = []

        for ohi, oh in enumerate(feat['obj_high_indices'][0].tolist()):
            # print(st_oh, oh)
            if oh != feat['obj_high_indices'][0].tolist()[st_oh]:
                alow_m.append(model.vocab['action_low'].index2word(feat['action_low_manip'][0].tolist())[st_oh:ohi])
                alow_mask.append(feat['action_low_mask_label'].tolist()[st_oh:ohi])
                st_oh = ohi
        alow_m.append(model.vocab['action_low'].index2word(feat['action_low_manip'][0].tolist())[st_oh:])
        alow_mask.append(feat['action_low_mask_label'].tolist()[st_oh:])
        
        look_count=0
        rotate_count=0
        move_count=0
        # man_flag = 0
        man_t = 0
        action_list = []
        while not done:
            # break if max_steps reached
            if t >= args.max_steps:
                print("max steps exceeded")
                break

            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            vis_feat = resnet.featurize([curr_image], batch=1).unsqueeze(0)
            feat['frames'] = vis_feat
            if model.panoramic:
                panoramic_actions, imgs = get_panoramic_actions(env)
                curr_image_left, curr_image_right, curr_image_up, curr_image_down = imgs
                feat['frames_left'] = resnet.featurize([curr_image_left], batch=1).unsqueeze(0)
                feat['frames_right'] = resnet.featurize([curr_image_right], batch=1).unsqueeze(0)
                feat['frames_up'] = resnet.featurize([curr_image_up], batch=1).unsqueeze(0)
                feat['frames_down'] = resnet.featurize([curr_image_down], batch=1).unsqueeze(0)
                # t += len(panoramic_actions)
                # if t >= args.max_steps:
                #     print("max steps exceeded")
                #     break

            # forward model

            m_out = model.step(feat, lang_index)
            # m_pred = model.extract_preds(m_out, [(traj_data, False)], feat, clean_special_tokens=False)

            m_pred = feat['out_action_low'].max(2)[1].tolist()
            
            # m_pred = list(m_pred.values())[0]

            # action prediction
            # action = m_pred['action_low']

            dist_action = m_out['out_action_low'][0][0].detach().cpu()
            dist_action = F.softmax(dist_action)
            action_mask = torch.ones(len(model.vocab['action_low']), dtype=torch.float)

            action_mask[model.vocab['action_low'].word2index(man_actions)] = -1
            action_mask[model.vocab['action_low'].word2index(cls.STOP_TOKEN)] = -1
            if t<(man_t+3): 
                action_mask[model.vocab['action_low'].word2index(cls.MANIPULATE_TOKEN)] = -1
            
            action = model.vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

            # action_list.append(action)
            # print(action_list, np.array(action_list) == 'LookUp_15', sum(np.array(action_list)=='LookDown_15'))
            # print(action_list)
            if len(action_list) > 20:
                action_list.pop(0)
            #     if (sum(np.array(action_list) == 'MoveAhead_25'))==0:
            #         action = 'MoveAhead_25'
                # print(action_list, (sum(np.array(action_list) == 'LookUp_15') + sum(np.array(action_list)=='LookDown_15') + sum(np.array(action_list)=='Manipulate')), (sum(np.array(action_list) == 'RotateRight_90') + sum(np.array(action_list)=='RotateLeft_90')))
                if (sum(np.array(action_list) == 'LookUp_15') + sum(np.array(action_list)=='LookDown_15') + sum(np.array(action_list)=='Manipulate')) == len(action_list):
                    action_mask[model.vocab['action_low'].word2index(cls.MANIPULATE_TOKEN)] = -1
                    action_mask[model.vocab['action_low'].word2index(['LookDown_15', 'LookUp_15'])] = -1    
                    action = model.vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

                if (sum(np.array(action_list) == 'RotateRight_90') + sum(np.array(action_list)=='RotateLeft_90')) == 20:
                    action_mask[model.vocab['action_low'].word2index(['RotateLeft_90', 'RotateRight_90'])] = -1    
                    action = model.vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

                if (sum(np.array(action_list) == 'Manipulate')) > 3:
                    action_mask[model.vocab['action_low'].word2index(['Manipulate'])] = -1    
                    action = model.vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))
            
            action_list.append(action)
            # if len(action_list) > 20:
            #     action_list.pop(0)



            # action = model.vocab['action_low'].index2word(m_pred)[0][0]

            # if action in man_actions:
            #     dist_action = m_out['out_action_low'][0][0].detach().cpu()
            #     dist_action = F.softmax(dist_action)
            #     action_mask = torch.ones(len(model.vocab['action_low']), dtype=torch.float)
            #     action_mask[model.vocab['action_low'].word2index(man_actions)] = -1
            #     action_mask[model.vocab['action_low'].word2index(cls.STOP_TOKEN)] = -1
            #     action = model.vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))



            # print(action)



            # if action == cls.MANIPULATE_TOKEN and prev_action == cls.MANIPULATE_TOKEN:
            #     dist_action = m_out['out_action_low'][0][0].detach().cpu()
            #     dist_action = F.softmax(dist_action)
            #     action_mask = torch.ones(len(model.vocab['action_low']), dtype=torch.float)
            #     action_mask[model.vocab['action_low'].word2index(man_actions)] = -1
            #     action_mask[model.vocab['action_low'].word2index(cls.MANIPULATE_TOKEN)] = -1
            #     action_mask[model.vocab['action_low'].word2index(cls.STOP_TOKEN)] = -1
            #     # if look_count == 5:
            #     #     action_mask[model.vocab['action_low'].word2index(['LookDown_15', 'LookUp_15'])] = -1
            #     #     look_count=0
            #     # if rotate_count == 5:
            #     #     action_mask[model.vocab['action_low'].word2index(['RotateLeft_90', 'RotateRight_90'])] = -1
            #     #     rotate_count=0
            #     # if move_count == 5:
            #     #     action_mask[model.vocab['action_low'].word2index(['MoveAhead_25'])] = -1
            #     #     move_count=0                    
            #     action = model.vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))
            #     # # print(action, "action after man loop break", look_count, rotate_count, move_count)
            #     # # print("###############3", action)
            #     # if 'Look' in action:
            #     #     look_count+=1
            #     # if 'Rotate' in action:
            #     #     rotate_count+=1
            #     # if 'MoveAhead' in action:
            #     #     move_count+=1
            
            if action == cls.MANIPULATE_TOKEN:
                man_success, t, fails = cls.doManipulation(feat, model, maskrcnn, curr_image, alow_m, alow_mask, lang_index, env, args, t, fails)

                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

                if man_success:
                    lang_index+=1

                if lang_index == max_lang_index:
                    goal_satisfied=True
                    break
                
                prev_action = cls.MANIPULATE_TOKEN
                t+=1
                man_t = t           
                continue

            # elif action == cls.MANIPULATE_TOKEN:
            #     dist_action = m_out['out_action_low'][0][0].detach().cpu()
            #     dist_action = F.softmax(dist_action)
            #     action_mask = torch.ones(len(model.vocab['action_low']), dtype=torch.float)
            #     action_mask[model.vocab['action_low'].word2index(man_actions)] = -1
            #     action_mask[model.vocab['action_low'].word2index(cls.MANIPULATE_TOKEN)] = -1
            #     action_mask[model.vocab['action_low'].word2index(cls.STOP_TOKEN)] = -1
            #     action = model.vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

                # prev_action = cls.MANIPULATE_TOKEN
                # continue        

            # if action == cls.MANIPULATE_TOKEN:
            #     obj_class = feat['objnav'][0][lang_index].unsqueeze(0).mm(model.emb_objnav.weight.t()).max(1)[1].tolist()
            #     obj_name = model.vocab['objnav'].index2word(obj_class)

            #     with torch.no_grad():
            #         out = maskrcnn([to_tensor(curr_image).cuda()])[0]
            #         for k in out:
            #             out[k] = out[k].detach().cpu()

            #     objects_present = [classes[o].lower() for o in out['labels']]
            #     # print("objects_present, obj_nam", objects_present, obj_name)
            #     #print("out['scores']", out['scores'])

            #     if obj_name[0] in objects_present:
            #         posi = objects_present.index(obj_name[0])
            #         scr = out['scores'][posi]
            #         # print("posi, scr", posi, scr)
            #         if scr > 0.5:
            #             man_action_success = []
            #             err_list = []
            #             prev_vis_feat_man = None                        
            #             for man_action, pred_class in zip(alow_m[lang_index], alow_mask[lang_index]):
            #                 # print("man, msk", man_action, classes[pred_class])
            #                 curr_image = Image.fromarray(np.uint8(env.last_event.frame))

            #                 with torch.no_grad():
            #                     out = maskrcnn([to_tensor(curr_image).cuda()])[0]
            #                     for k in out:
            #                         out[k] = out[k].detach().cpu()
                                    
            #                 if sum(out['labels'] == pred_class) == 0:
            #                     mask = np.zeros((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
            #                 else:
            #                     masks = out['masks'][out['labels'] == pred_class].detach().cpu()
            #                     scores = out['scores'][out['labels'] == pred_class].detach().cpu()

            #                 # masks = out['masks'][out['labels'] == pred_class].detach().cpu()
            #                 # scores = out['scores'][out['labels'] == pred_class].detach().cpu()

            #                 # Instance selection based on the minimum distance between the prev. and cur. instance of a same class.
            #                 if prev_class != pred_class:
            #                     scores, indices = scores.sort(descending=True)
            #                     masks = masks[indices]
            #                     prev_class = pred_class
            #                     prev_center = masks[0].squeeze(dim=0).nonzero().double().mean(dim=0)
            #                 else:
            #                     cur_centers = torch.stack([m.nonzero().double().mean(dim=0) for m in masks.squeeze(dim=1)])
            #                     distances = ((cur_centers - prev_center)**2).sum(dim=1)
            #                     distances, indices = distances.sort()
            #                     masks = masks[indices]
            #                     prev_center = cur_centers[0]

            #                 mask = np.squeeze(masks[0].numpy(), axis=0)
            #                 t_success, _, _, err, _ = env.va_interact(man_action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            #                 # inventory_objects = env.last_event.metadata['inventoryObjects'][0]['objectType']
            #                 # if man_action=='PickupObject' and classes[] not in inventory_objects
            #                 err_list.append(err)
            #                 man_action_success.append(t_success)
            #                 # print("t_success, err", t_success, err)
            #                 t+=1

            #                 if np.all(np.array(err_list[-5:] == err_list[-1])):
            #                     print("same error repeating")
            #                     break

                            
            #             if np.all(np.array(man_action_success)):
            #                 lang_index+=1
            #                 # print("\nlang_index", lang_index)                        


                

                # print([classes[o].lower() for o in out['labels']])
                # print(obj_class, obj_name)
                # exit()
                # action_emb_t.mm(self.emb.weight.t())

            if prev_vis_feat != None:
                od_score = ((prev_vis_feat - vis_feat)**2).sum().sqrt()
                epsilon = 1
                if od_score < epsilon:
                    dist_action = m_out['out_action_low'][0][0].detach().cpu()
                    dist_action = F.softmax(dist_action)
                    action_mask = torch.ones(len(model.vocab['action_low']), dtype=torch.float)
                    action_mask[model.vocab['action_low'].word2index(prev_action)] = -1
                    action_mask[model.vocab['action_low'].word2index(man_actions)] = -1
                    action_mask[model.vocab['action_low'].word2index(cls.STOP_TOKEN)] = -1
                    action_mask[model.vocab['action_low'].word2index(cls.MANIPULATE_TOKEN)] = -1
                    action = model.vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

            # if action == cls.STOP_TOKEN:
            #     print("\tpredicted STOP for no reason")
            #     t+=1
            #     continue
                # break

            # mask prediction
            mask = None

            # if model.has_interaction(action):
            #     class_dist = m_pred['action_low_mask'][0]
            #     pred_class = np.argmax(class_dist)

            #     # mask generation
            #     with torch.no_grad():
            #         out = maskrcnn([to_tensor(curr_image).cuda()])[0]
            #         for k in out:
            #             out[k] = out[k].detach().cpu()

            #     if sum(out['labels'] == pred_class) == 0:
            #         mask = np.zeros((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
            #     else:
            #         masks = out['masks'][out['labels'] == pred_class].detach().cpu()
            #         scores = out['scores'][out['labels'] == pred_class].detach().cpu()

            #         # Instance selection based on the minimum distance between the prev. and cur. instance of a same class.
            #         if prev_class != pred_class:
            #             scores, indices = scores.sort(descending=True)
            #             masks = masks[indices]
            #             prev_class = pred_class
            #             prev_center = masks[0].squeeze(dim=0).nonzero().double().mean(dim=0)
            #         else:
            #             cur_centers = torch.stack([m.nonzero().double().mean(dim=0) for m in masks.squeeze(dim=1)])
            #             distances = ((cur_centers - prev_center)**2).sum(dim=1)
            #             distances, indices = distances.sort()
            #             masks = masks[indices]
            #             prev_center = cur_centers[0]

            #         mask = np.squeeze(masks[0].numpy(), axis=0)

            # print action
            if args.debug:
                print(action)

            # use predicted action and mask (if available) to interact with the env
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            # print(action, t_success)
            if not t_success:
                fails += 1
                print(fails)
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

            prev_vis_feat = vis_feat
            prev_action = action

        # check if goal was satisfied
        # goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True

        # goal_conditions
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

        print("-------------")
        print("SR: %d/%d = %.5f" % (results['all']['success']['num_successes'],
                                    results['all']['success']['num_evals'],
                                    results['all']['success']['success_rate']))
        print("PLW SR: %.5f" % (results['all']['path_length_weighted_success_rate']))
        print("GC: %d/%d = %.5f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                    results['all']['goal_condition_success']['total_goal_conditions'],
                                    results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW GC: %.5f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")

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

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, (os.path.basename(self.args.model_path)).split('.')[0] + '_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')        
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

