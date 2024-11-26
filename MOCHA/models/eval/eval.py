import json
import pprint
import random
import time
import torch
import torch.multiprocessing as mp
from models.nn.resnet import Resnet
from data.preprocess import Dataset
from importlib import import_module

class Eval(object):

    # tokens
    STOP_TOKEN = "<<stop>>"
    SEQ_TOKEN = "<<seg>>"
    TERMINAL_TOKENS = [STOP_TOKEN, SEQ_TOKEN]
    MANIPULATE_TOKEN = "Manipulate"

    def __init__(self, args, manager):
        # args and manager
        self.args = args
        self.manager = manager

        # load splits
        with open(self.args.splits) as f:
            self.splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in self.splits.items()})

        print("Loading Nav: ", self.args.nav_model_path)
        M_nav = import_module(self.args.nav_model)
        nav_model, optimizer = M_nav.Module.load(self.args.nav_model_path)
        nav_model.share_memory()
        nav_model.eval()
        nav_model.test_mode = True #Change here

        # updated args
        nav_model.args.dout = self.args.nav_model_path.replace(self.args.nav_model_path.split('/')[-1], '')
        nav_model.args.data = self.args.data if self.args.data else self.nav_model.args.data

        self.model = {'nav':nav_model.cuda()}

        ####################################################################################################################################
        print("Loading Pickup: ", self.args.pickup_model_path)
        M_pickup = import_module(self.args.sub_model)
        pickup_model, optimizer = M_pickup.Module.load(self.args.pickup_model_path)
        pickup_model.share_memory()
        pickup_model.eval()
        pickup_model.test_mode = True #Change here

        # updated args
        pickup_model.args.dout = self.args.pickup_model_path.replace(self.args.pickup_model_path.split('/')[-1], '')
        pickup_model.args.data = self.args.data if self.args.data else self.pickup_model.args.data

        self.model['pickup'] = pickup_model.cuda()

        ####################################################################################################################################

        print("Loading put: ", self.args.put_model_path)
        M_put = import_module(self.args.sub_model)
        put_model, optimizer = M_put.Module.load(self.args.put_model_path)
        put_model.share_memory()
        put_model.eval()
        put_model.test_mode = True #Change here

        # updated args
        put_model.args.dout = self.args.put_model_path.replace(self.args.put_model_path.split('/')[-1], '')
        put_model.args.data = self.args.data if self.args.data else self.put_model.args.data

        self.model['put'] = put_model.cuda()

        ####################################################################################################################################

        print("Loading heat: ", self.args.heat_model_path)
        M_heat = import_module(self.args.sub_model)
        heat_model, optimizer = M_heat.Module.load(self.args.heat_model_path)
        heat_model.share_memory()
        heat_model.eval()
        heat_model.test_mode = True #Change here

        # updated args
        heat_model.args.dout = self.args.heat_model_path.replace(self.args.heat_model_path.split('/')[-1], '')
        heat_model.args.data = self.args.data if self.args.data else self.heat_model.args.data

        self.model['heat'] = heat_model.cuda()

        ####################################################################################################################################

        print("Loading cool: ", self.args.cool_model_path)
        M_cool = import_module(self.args.sub_model)
        cool_model, optimizer = M_cool.Module.load(self.args.cool_model_path)
        cool_model.share_memory()
        cool_model.eval()
        cool_model.test_mode = True #Change here

        # updated args
        cool_model.args.dout = self.args.cool_model_path.replace(self.args.cool_model_path.split('/')[-1], '')
        cool_model.args.data = self.args.data if self.args.data else self.cool_model.args.data

        self.model['cool'] = cool_model.cuda()

        ####################################################################################################################################

        print("Loading clean: ", self.args.clean_model_path)
        M_clean = import_module(self.args.sub_model)
        clean_model, optimizer = M_clean.Module.load(self.args.clean_model_path)
        clean_model.share_memory()
        clean_model.eval()
        clean_model.test_mode = True #Change here

        # updated args
        clean_model.args.dout = self.args.clean_model_path.replace(self.args.clean_model_path.split('/')[-1], '')
        clean_model.args.data = self.args.data if self.args.data else self.clean_model.args.data

        self.model['clean'] = clean_model.cuda()
        
        ####################################################################################################################################

        print("Loading toggle: ", self.args.toggle_model_path)
        M_toggle = import_module(self.args.sub_model)
        toggle_model, optimizer = M_toggle.Module.load(self.args.toggle_model_path)
        toggle_model.share_memory()
        toggle_model.eval()
        toggle_model.test_mode = True #Change here

        # updated args
        toggle_model.args.dout = self.args.toggle_model_path.replace(self.args.toggle_model_path.split('/')[-1], '')
        toggle_model.args.data = self.args.data if self.args.data else self.toggle_model.args.data

        self.model['toggle'] = toggle_model.cuda()

        ####################################################################################################################################

        print("Loading slice: ", self.args.slice_model_path)
        M_slice = import_module(self.args.sub_model)
        slice_model, optimizer = M_slice.Module.load(self.args.slice_model_path)
        slice_model.share_memory()
        slice_model.eval()
        slice_model.test_mode = True #Change here

        # updated args
        slice_model.args.dout = self.args.slice_model_path.replace(self.args.slice_model_path.split('/')[-1], '')
        slice_model.args.data = self.args.data if self.args.data else self.slice_model.args.data

        self.model['slice'] = slice_model.cuda()

        ####################################################################################################################################
        
        print("Loading object model: ", self.args.object_model_path)
        M_object = import_module(self.args.object_model)
        object_model, optimizer = M_object.Module.load(self.args.object_model_path)
        object_model.share_memory()
        object_model.eval()
        object_model.test_mode = True #Change here

        # updated args
        object_model.args.dout = self.args.object_model_path.replace(self.args.object_model_path.split('/')[-1], '')
        object_model.args.data = self.args.data if self.args.data else self.object_model.args.data

        self.model['object'] = object_model.cuda()

        ####################################################################################################################################
        
        print("Loading subgoal prediction model: ", self.args.subgoal_model_path)
        M_subgoal = import_module(self.args.subgoal_pred_model)
        subgoal_model, optimizer = M_subgoal.Module.load(self.args.subgoal_model_path)
        subgoal_model.share_memory()
        subgoal_model.eval()
        subgoal_model.test_mode = True #Change here

        # updated args
        subgoal_model.args.dout = self.args.subgoal_model_path.replace(self.args.subgoal_model_path.split('/')[-1], '')
        subgoal_model.args.data = self.args.data if self.args.data else self.subgoal_model.args.data

        self.model['subgoal'] = subgoal_model.cuda()

        ####################################################################################################################################


        # preprocess and save
        if args.preprocess:
            print("\nPreprocessing dataset and saving to %s folders ... This is will take a while. Do this once as required:" % self.model.args.pp_folder)
            self.model['nav'].args.fast_epoch = self.args.fast_epoch
            dataset = Dataset(self.model['nav'].args, self.model['nav'].vocab)
            dataset.preprocess_splits(self.splits)

        # load resnet
        args.visual_model = 'resnet18'
        self.resnet = Resnet(args, eval=True, share_memory=True, use_conv_feat=True)


        # success and failure lists
        self.create_stats()

        # set random seed for shuffling
        random.seed(int(time.time()))

    
    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        files = self.splits[self.args.eval_split]

        # debugging: fast epoch
        if self.args.fast_epoch:
            files = files[:16]

        if self.args.shuffle:
            random.shuffle(files)
        for traj in files:
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
        for n in range(self.args.num_threads):
            thread = mp.Process(target=self.run, args=(self.model, self.resnet, task_queue, self.args, lock,
                                                       self.successes, self.failures, self.results))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        # save
        self.save_results()

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
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures):
        raise NotImplementedError()

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures):
        raise NotImplementedError()

    def save_results(self):
        raise NotImplementedError()

    def create_stats(self):
        raise NotImplementedError()