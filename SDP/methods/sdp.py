import copy
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.er_baseline import ER
from utils.data_loader import ImageDataset, StreamDataset, cutmix_data

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class SDP(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.memory_size = kwargs["memory_size"]

        self.sdp_mean = kwargs['sdp_mean']
        self.sdp_varcoeff = kwargs['sdp_var']
        assert 0.5 - 1 / self.sdp_mean < self.sdp_varcoeff < 1 - 1 / self.sdp_mean
        self.ema_ratio_1 = (1 - np.sqrt(2 * self.sdp_varcoeff - 1 + 2 / self.sdp_mean)) / (self.sdp_mean - 1 - self.sdp_mean * self.sdp_varcoeff)
        self.ema_ratio_2 = (1 + np.sqrt(2 * self.sdp_varcoeff - 1 + 2 / self.sdp_mean)) / (
                self.sdp_mean - 1 - self.sdp_mean * self.sdp_varcoeff)
        self.cur_time = None
        self.sdp_model = copy.deepcopy(self.model)
        self.ema_model_1 = copy.deepcopy(self.model)
        self.ema_model_2 = copy.deepcopy(self.model)
        self.sdp_updates = 0
        self.num_steps = 0
        self.cls_pred_mean = torch.zeros(1).to(self.device)
        self.temp_ret = None
        self.cls_pred_length = 100
        self.cls_pred = []


    def update_memory(self, sample):
        self.balanced_replace_memory(sample)

    def balanced_replace_memory(self, sample):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.random.choice(
                np.flatnonzero(np.array(label_frequency) == np.array(label_frequency).max()))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace)
        else:
            self.memory.replace_sample(sample)

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        prev_bias = copy.deepcopy(self.model.fc.bias.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
                self.model.fc.bias[:self.num_learned_class - 1] = prev_bias
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.sdp_model.fc = copy.deepcopy(self.model.fc)
        self.ema_model_1.fc = copy.deepcopy(self.model.fc)
        self.ema_model_2.fc = copy.deepcopy(self.model.fc)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
        self.cls_pred.append([])

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        while len(self.temp_batch) > self.batch_size:
            del self.temp_batch[0]
        self.update_memory(sample)
        self.num_updates += self.online_iter
        self.num_steps = sample_num
        self.sample_inference([sample])

        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates))
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
        self.update_schedule()


    def sample_inference(self, sample):
        sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.test_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                       transform_on_gpu=False, use_kornia=False)
        self.sdp_model.eval()
        stream_data = sample_dataset.get_data()
        x = stream_data['image']
        y = stream_data['label']
        x = x.to(self.device)
        logit = self.sdp_model(x)
        prob = F.softmax(logit, dim=1)
        self.cls_pred[y].append(prob[0, y].item())
        if len(self.cls_pred[y]) > self.cls_pred_length:
            del self.cls_pred[y][0]
        self.cls_pred_mean = np.clip(np.mean([np.mean(cls_pred) for cls_pred in self.cls_pred]) - 1/self.num_learned_class, 0, 1) * self.num_learned_class/(self.num_learned_class + 1)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, 0)
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)

            logit, loss = self.model_forward(x, y, use_cutmix=True)

            _, preds = logit.topk(self.topk, 1, True, True)

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.update_schedule()
            self.update_sdp_model(num_updates=1.0)

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    def update_schedule(self, reset=False):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr * (1 - self.cls_pred_mean)

    @torch.no_grad()
    def update_sdp_model(self, num_updates=1.0):
        ema_inv_ratio_1 = (1 - self.ema_ratio_1) ** num_updates
        ema_inv_ratio_2 = (1 - self.ema_ratio_2) ** num_updates
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.sdp_model.named_parameters())
        ema_params_1 = OrderedDict(self.ema_model_1.named_parameters())
        ema_params_2 = OrderedDict(self.ema_model_2.named_parameters())
        assert model_params.keys() == ema_params.keys()
        assert model_params.keys() == ema_params_1.keys()
        assert model_params.keys() == ema_params_2.keys()
        self.sdp_updates += 1
        for name, param in model_params.items():
            ema_params_1[name].sub_((1. - ema_inv_ratio_1) * (ema_params_1[name] - param))
            ema_params_2[name].sub_((1. - ema_inv_ratio_2) * (ema_params_2[name] - param))
            ema_params[name].copy_(
                self.ema_ratio_2 / (self.ema_ratio_2 - self.ema_ratio_1) * ema_params_1[name] - self.ema_ratio_1 / (self.ema_ratio_2 - self.ema_ratio_1) * ema_params_2[
                    name])
            # + ((1. - self.ema_ratio_2)*self.ema_ratio_1**self.ema_updates - (1. - self.ema_ratio_1)*self.ema_ratio_2**self.ema_updates) / (self.ema_ratio_1 - self.ema_ratio_2) * param)
        self.sdp_model.fc = copy.deepcopy(self.model.fc)
        self.ema_model_1.fc = copy.deepcopy(self.model.fc)
        self.ema_model_2.fc = copy.deepcopy(self.model.fc)

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.sdp_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def model_forward(self, x, y, distill=True, use_cutmix=True):
        criterion = nn.CrossEntropyLoss(reduction='none')
        self.sdp_model.train()
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill:
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
                    with torch.no_grad():
                        logit2, feature2 = self.sdp_model(x, get_feature=True)
                    distill_loss = ((feature - feature2.detach()) ** 2).sum(dim=1)
                    sample_weight = self.cls_pred_mean
                    grad = lam * self.get_grad(logit.detach(), labels_a, self.model.fc.weight) + (
                            1 - lam) * self.get_grad(logit.detach(), labels_b, self.model.fc.weight)
                    beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() * 4 + 1e-8)).mean()
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = criterion(logit, y)
                    self.sdp_model.zero_grad()
                    with torch.no_grad():
                        logit2, feature2 = self.sdp_model(x, get_feature=True)
                    distill_loss = ((feature - feature2.detach()) ** 2).sum(dim=1)
                    sample_weight = self.cls_pred_mean
                    grad = self.get_grad(logit.detach(), y, self.model.fc.weight)
                    beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() * 4 + 1e-8)).mean()
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)

    def get_grad(self, logit, label, weight):
        prob = F.softmax(logit)
        oh_label = F.one_hot(label.long(), self.num_learned_class)
        return torch.matmul((prob - oh_label), weight)

    def get_class_weight(self):
        fc_weights = self.model.fc.weight.data.detach().cpu()
        distill_weight = torch.abs(fc_weights - torch.mean(fc_weights, dim=0))
        distill_weight /= torch.sum(distill_weight, dim=1).reshape(-1, 1) + 1e-8
        return distill_weight

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        eval_dict = self.evaluation(test_loader, self.criterion)
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"])
        if sample_num >= self.f_next_time:
            self.get_forgetting(sample_num, test_list, cls_dict, batch_size, n_worker)
            self.f_next_time += self.f_period
            self.f_calculated = True
        else:
            self.f_calculated = False
        return eval_dict

    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        total_correct_fc, total_loss_fc = 0.0, 0.0
        correct_l_fc = torch.zeros(self.n_classes)

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit, feature = self.model(x, get_feature=True)
                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        avg_acc_fc = total_correct_fc / total_num_data
        avg_loss_fc = total_loss_fc / len(test_loader)
        cls_acc_fc = (correct_l_fc / (num_data_l + 1e-5)).numpy().tolist()
        fc_ret = {"avg_loss": avg_loss_fc, "avg_acc": avg_acc_fc, "cls_acc": cls_acc_fc}
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc, "fc_ret": fc_ret}

        return ret

