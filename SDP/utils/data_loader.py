import logging.config
import os
import json
from typing import List

import PIL
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn
from kornia import image_to_tensor
from kornia.geometry.transform import resize

TIMEOUT = 5.0

logger = logging.getLogger()
class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self, input_size=32):
        super().__init__()
        self.input_size = input_size

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x_tmp = np.array(x)  # HxWxC
        x_out = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_out = resize(x_out.float() / 255.0, (self.input_size, self.input_size))
        return x_out

class ImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, dataset: str, transform=None, cls_list=None, data_dir=None,
                 preload=False, device=None, transform_on_gpu=False, use_kornia=False):
        self.use_kornia = use_kornia
        self.data_frame = data_frame
        self.dataset = dataset
        self.transform = transform
        self.cls_list = cls_list
        self.data_dir = data_dir
        self.preload = preload
        self.device = device
        self.transform_on_gpu = transform_on_gpu
        if self.preload:
            mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)
            self.preprocess = Preprocess(input_size=inp_size)
            if self.transform_on_gpu:
                self.transform_cpu = transforms.Compose(
                    [
                        transforms.Resize((inp_size, inp_size)),
                        transforms.PILToTensor()
                    ])
                self.transform_gpu = self.transform
            self.loaded_images = []
            for idx in range(len(self.data_frame)):
                sample = dict()
                try:
                    img_name = self.data_frame.iloc[idx]["file_name"]
                except KeyError:
                    img_name = self.data_frame.iloc[idx]["filepath"]
                if self.cls_list is None:
                    label = self.data_frame.iloc[idx].get("label", -1)
                else:
                    label = self.cls_list.index(self.data_frame.iloc[idx]["klass"])
                if self.data_dir is None:
                    img_path = os.path.join("dataset", self.dataset, img_name)
                else:
                    img_path = os.path.join(self.data_dir, img_name)
                image = PIL.Image.open(img_path).convert("RGB")
                if self.use_kornia:
                    image = self.preprocess(PIL.Image.open(img_path).convert('RGB'))
                elif self.transform_on_gpu:
                    image = self.transform_cpu(PIL.Image.open(img_path).convert('RGB'))
                elif self.transform:
                    image = self.transform(image)
                sample["image"] = image
                sample["label"] = label
                sample["image_name"] = img_name
                self.loaded_images.append(sample)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.preload:
            return self.loaded_images[idx]
        else:
            sample = dict()
            if torch.is_tensor(idx):
                idx = idx.tolist()
            img_name = self.data_frame.iloc[idx]["file_name"]
            if self.cls_list is None:
                label = self.data_frame.iloc[idx].get("label", -1)
            else:
                label = self.cls_list.index(self.data_frame.iloc[idx]["klass"])

            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            image = PIL.Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            sample["image"] = image
            sample["label"] = label
            sample["image_name"] = img_name
            return sample

    def get_image_class(self, y):
        return self.data_frame[self.data_frame["label"] == y]

    def generate_idx(self, batch_size):
        if self.preload:
            arr = np.arange(len(self.loaded_images))
        else:
            arr = np.arange(len(self.data_frame))
        np.random.shuffle(arr)
        if batch_size >= len(arr):
            return [arr]
        else:
            return np.split(arr, np.arange(batch_size, len(arr), batch_size))

    def get_data_gpu(self, indices):
        images = []
        labels = []
        data = {}
        if self.use_kornia:
            images = [self.loaded_images[i]["image"] for i in indices]
            images = torch.stack(images).to(self.device)
            images = self.transform_gpu(images)
            data["image"] = images

            for i in indices:
            # labels
                labels.append(self.loaded_images[i]["label"])
        else:
            for i in indices:
                if self.preload:
                    if self.transform_on_gpu:
                        images.append(self.transform_gpu(self.loaded_images[i]["image"].to(self.device)))
                    else:
                        images.append(self.transform(self.loaded_images[i]["image"]).to(self.device))
                    labels.append(self.loaded_images[i]["label"])
                else:
                    try:
                        img_name = self.data_frame.iloc[i]["file_name"]
                    except KeyError:
                        img_name = self.data_frame.iloc[i]["filepath"]
                    if self.cls_list is None:
                        label = self.data_frame.iloc[i].get("label", -1)
                    else:
                        label = self.cls_list.index(self.data_frame.iloc[i]["klass"])
                    if self.data_dir is None:
                        img_path = os.path.join("dataset", self.dataset, img_name)
                    else:
                        img_path = os.path.join(self.data_dir, img_name)
                    image = PIL.Image.open(img_path).convert("RGB")
                    image = self.transform(image)
                    images.append(image.to(self.device))
                    labels.append(label)
            data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels).to(self.device)
        return data


class StreamDataset(Dataset):
    def __init__(self, datalist, dataset, transform, cls_list, data_dir=None, device=None, transform_on_gpu=False, use_kornia=True):
        self.use_kornia = use_kornia
        self.images = []
        self.labels = []
        self.dataset = dataset
        self.transform = transform
        self.cls_list = cls_list
        self.data_dir = data_dir
        self.device = device

        self.transform_on_gpu = transform_on_gpu
        mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)

        self.preprocess = Preprocess(input_size=inp_size)
        if self.transform_on_gpu:
            self.transform_cpu = transforms.Compose(
                [
                    transforms.Resize((inp_size, inp_size)),
                    transforms.PILToTensor()
                ])
            self.transform_gpu = transform
        for data in datalist:
            try:
                img_name = data['file_name']
            except KeyError:
                img_name = data['filepath']
            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            if self.use_kornia:
                self.images.append(self.preprocess(PIL.Image.open(img_path).convert('RGB')))
            elif self.transform_on_gpu:
                self.images.append(self.transform_cpu(PIL.Image.open(img_path).convert('RGB')))
            else:
                self.images.append(PIL.Image.open(img_path).convert('RGB'))
            self.labels.append(self.cls_list.index(data['klass']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        return sample

    @torch.no_grad()
    def get_data(self):
        data = dict()
        images = []
        labels = []
        if self.use_kornia:
            # images
            images = torch.stack(self.images).to(self.device)
            data['image'] = self.transform_gpu(images)

        if not self.use_kornia:
            for i, image in enumerate(self.images):
                if self.transform_on_gpu:
                    images.append(self.transform_gpu(image.to(self.device)))
                else:
                    images.append(self.transform(image))
            data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(self.labels)
        return data


class MemoryDataset(Dataset):
    def __init__(self, dataset, transform=None, cls_list=None, device=None, test_transform=None,
                 data_dir=None, transform_on_gpu=True, save_test=None, keep_history=False, use_kornia=True):
        self.use_kornia = use_kornia
        self.datalist = []
        self.labels = []
        self.images = []
        self.stream_images = []
        self.stream_labels = []
        self.dataset = dataset
        self.transform = transform
        self.cls_list = []
        self.cls_dict = {cls_list[i]:i for i in range(len(cls_list))}
        self.cls_count = []
        self.cls_idx = []
        self.cls_train_cnt = np.array([])
        self.score = []
        self.others_loss_decrease = np.array([])
        self.previous_idx = np.array([], dtype=int)
        self.device = device
        self.test_transform = test_transform
        self.data_dir = data_dir
        self.keep_history = keep_history
        self.usage_cnt = []
        self.transform_on_gpu = transform_on_gpu
        mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)

        self.preprocess = Preprocess(input_size=inp_size)
        if self.transform_on_gpu:
            self.transform_cpu = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.PILToTensor()
            ])
            self.transform_gpu = transform
            self.test_transform = transforms.Compose([transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std)])
        self.save_test = save_test
        if self.save_test is not None:
            self.device_img = []

    def __len__(self):
        return len(self.images)

    def add_new_class(self, cls_list):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.cls_dict = {self.cls_list[i]:i for i in range(len(self.cls_list))}
        self.cls_train_cnt = np.append(self.cls_train_cnt, 0)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.value()
        label = self.labels[idx]
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        return sample

    def register_stream(self, datalist):
        self.stream_images = []
        self.stream_labels = []
        for data in datalist:
            try:
                img_name = data['file_name']
            except KeyError:
                img_name = data['filepath']
            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            if self.use_kornia:
                self.stream_images.append(self.preprocess(PIL.Image.open(img_path).convert('RGB')))
            elif self.transform_on_gpu:
                self.stream_images.append(self.transform_cpu(PIL.Image.open(img_path).convert('RGB')))
            else:
                self.stream_images.append(PIL.Image.open(img_path).convert('RGB'))
            self.stream_labels.append(self.cls_list.index(data['klass']))

    def update_gss_score(self, score, idx=None):
        if idx is None:
            self.score.append(score)
        else:
            self.score[idx] = score

    def replace_sample(self, sample, idx=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        if idx is None:
            self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.datalist.append(sample)
            self.usage_cnt.append(0)
            try:
                img_name = sample['file_name']
            except KeyError:
                img_name = sample['filepath']
            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            img = PIL.Image.open(img_path).convert('RGB')
            if self.use_kornia:
                img = self.preprocess(img)
            elif self.transform_on_gpu:
                img = self.transform_cpu(img)
            self.images.append(img)
            self.labels.append(self.cls_dict[sample['klass']])
            if self.save_test == 'gpu':
                self.device_img.append(self.test_transform(img).to(self.device).unsqueeze(0))
            elif self.save_test == 'cpu':
                self.device_img.append(self.test_transform(img).unsqueeze(0))
            if self.keep_history:
                if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                    self.others_loss_decrease = np.append(self.others_loss_decrease, 0)
                else:
                    self.others_loss_decrease = np.append(self.others_loss_decrease, np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]]))
        else:
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            self.datalist[idx] = sample
            self.usage_cnt[idx] = 0
            self.cls_idx[self.cls_dict[sample['klass']]].append(idx)
            try:
                img_name = sample['file_name']
            except KeyError:
                img_name = sample['filepath']
            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            img = PIL.Image.open(img_path).convert('RGB')
            if self.use_kornia:
                img = self.preprocess(img)
            elif self.transform_on_gpu:
                img = self.transform_cpu(img)
            self.images[idx] = img
            self.labels[idx] = self.cls_list.index(sample['klass'])
            if self.save_test == 'gpu':
                self.device_img[idx] = self.test_transform(img).to(self.device).unsqueeze(0)
            elif self.save_test == 'cpu':
                self.device_img[idx] = self.test_transform(img).unsqueeze(0)
            if self.keep_history:
                if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                    self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease)
                else:
                    self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]])

    def get_weight(self):
        weight = np.zeros(len(self.images))
        for i, indices in enumerate(self.cls_idx):
            weight[indices] = 1/self.cls_count[i]
        return weight

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=False, transform=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_images))
        batch_size = min(batch_size, stream_batch_size + len(self.images))
        memory_batch_size = batch_size - stream_batch_size
        if memory_batch_size > 0:
            if use_weight:
                weight = self.get_weight()
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, p=weight / np.sum(weight),
                                           replace=False)
            else:
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, replace=False)
        if stream_batch_size > 0:
            if len(self.stream_images) > stream_batch_size:
                stream_indices = np.random.choice(range(len(self.stream_images)), size=stream_batch_size, replace=False)
            else:
                stream_indices = np.arange(len(self.stream_images))

        data = dict()
        images = []
        labels = []
        use_cnt = []
        mean_usage = np.mean(self.usage_cnt)
        if self.use_kornia:
            # images
            if stream_batch_size > 0:
                for i in stream_indices:
                    images.append(self.stream_images[i])
                    labels.append(self.stream_labels[i])
            if memory_batch_size > 0:
                for i in indices:
                    images.append(self.images[i])
                    labels.append(self.labels[i])

            images = torch.stack(images).to(self.device)
            images = self.transform_gpu(images)
        else:
            if stream_batch_size > 0:
                for i in stream_indices:
                    if transform is None:
                        if self.transform_on_gpu:
                            images.append(self.transform_gpu(self.stream_images[i].to(self.device)))
                        else:
                            images.append(self.transform(self.stream_images[i]))
                    else:
                        if self.transform_on_gpu:
                            images.append(transform(self.stream_images[i].to(self.device)))
                        else:
                            images.append(transform(self.stream_images[i]))
                    labels.append(self.stream_labels[i])

            if memory_batch_size > 0:
                for i in indices:
                    if transform is None:
                        if self.transform_on_gpu:
                            images.append(self.transform_gpu(self.images[i].to(self.device)))
                        else:
                            images.append(self.transform(self.images[i]))
                    else:
                        if self.transform_on_gpu:
                            images.append(transform(self.images[i].to(self.device)))
                        else:
                            images.append(transform(self.images[i]))
                    use_cnt.append(self.usage_cnt[i] / mean_usage)
                    labels.append(self.labels[i])
                    self.cls_train_cnt[self.labels[i]] += 1
                    self.usage_cnt[i] += 1

            images = torch.stack(images)
        data['image'] = images
        data['label'] = torch.LongTensor(labels)
        data['usage'] = torch.Tensor(use_cnt)
        if self.keep_history:
            self.previous_idx = np.append(self.previous_idx, indices)
        return data

    def update_loss_history(self, loss, prev_loss, ema_ratio=0.90, dropped_idx=None):
        if dropped_idx is None:
            loss_diff = np.mean(loss - prev_loss)
        elif len(prev_loss) > 0:
            mask = np.ones(len(loss), bool)
            mask[dropped_idx] = False
            loss_diff = np.mean((loss[:len(prev_loss)] - prev_loss)[mask[:len(prev_loss)]])
        else:
            loss_diff = 0
        difference = loss_diff - np.mean(self.others_loss_decrease[self.previous_idx]) / len(self.previous_idx)
        self.others_loss_decrease[self.previous_idx] -= (1 - ema_ratio) * difference
        self.previous_idx = np.array([], dtype=int)

    def get_two_batches(self, batch_size, test_transform):
        indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)
        data_1 = dict()
        data_2 = dict()
        images = []
        labels = []
        if self.use_kornia:
            # images
            for i in indices:
                images.append(self.images[i])
                labels.append(self.labels[i])
            images = torch.stack(images).to(self.device)
            data_1['image'] = self.transform_gpu(images)

        else:
            for i in indices:
                if self.transform_on_gpu:
                    images.append(self.transform_gpu(self.images[i].to(self.device)))
                else:
                    images.append(self.transform(self.images[i]))
                labels.append(self.labels[i])
            data_1['image'] = torch.stack(images)
        data_1['label'] = torch.LongTensor(labels)
        images = []
        labels = []
        for i in indices:
            images.append(self.test_transform(self.images[i]))
            labels.append(self.labels[i])
        data_2['image'] = torch.stack(images)
        data_2['label'] = torch.LongTensor(labels)
        return data_1, data_2

    def make_cls_dist_set(self, labels, transform=None):
        if transform is None:
            transform = self.transform
        indices = []
        for label in labels:
            indices.append(np.random.choice(self.cls_idx[label]))
        indices = np.array(indices)
        data = dict()
        images = []
        labels = []
        for i in indices:
            images.append(transform(self.images[i]))
            labels.append(self.labels[i])
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        return data

    def make_val_set(self, size=None, transform=None):
        if size is None:
            size = int(0.1*len(self.images))
        if transform is None:
            transform = self.transform
        size_per_cls = size//len(self.cls_list)
        indices = []
        for cls_list in self.cls_idx:
            if len(cls_list) >= size_per_cls:
                indices.append(np.random.choice(cls_list, size=size_per_cls, replace=False))
            else:
                indices.append(np.random.choice(cls_list, size=size_per_cls, replace=True))
        indices = np.concatenate(indices)
        data = dict()
        images = []
        labels = []
        for i in indices:
            images.append(transform(self.images[i]))
            labels.append(self.labels[i])
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        return data

    def is_balanced(self):
        mem_per_cls = len(self.images)//len(self.cls_list)
        for cls in self.cls_count:
            if cls < mem_per_cls or cls > mem_per_cls+1:
                return False
        return True

class GdumbMemory(MemoryDataset):
    def __init__(self, dataset, transform=None, cls_list=None, device=None, test_transform=None,
                 data_dir=None, transform_on_gpu=True, save_test=None, keep_history=False, use_kornia=True):
        super().__init__(dataset, transform, cls_list, device, test_transform,
                 data_dir, transform_on_gpu, save_test, keep_history)


    def replace_sample(self, sample, idx=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        if idx is None:
            self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.datalist.append(sample)
            self.usage_cnt.append(0)
            self.images.append(sample)
            self.labels.append(self.cls_dict[sample['klass']])
            if self.keep_history:
                if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                    self.others_loss_decrease = np.append(self.others_loss_decrease, 0)
                else:
                    self.others_loss_decrease = np.append(self.others_loss_decrease, np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]]))
        else:
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            self.datalist[idx] = sample
            self.usage_cnt[idx] = 0
            self.cls_idx[self.cls_dict[sample['klass']]].append(idx)
            self.images[idx] = sample
            self.labels[idx] = self.cls_list.index(sample['klass'])
            if self.keep_history:
                if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                    self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease)
                else:
                    self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]])


class DistillationMemory(MemoryDataset):
    def __init__(self, dataset, transform=None, cls_list=None, device=None, test_transform=None,
                 data_dir=None, transform_on_gpu=True, save_test=None, keep_history=False, use_logit=True, use_feature=False, use_kornia=True):
        super().__init__(dataset, transform, cls_list, device, test_transform,
                 data_dir, transform_on_gpu, save_test, keep_history, use_kornia=use_kornia)
        self.logits = []
        self.features = []
        self.logits_mask = []
        self.use_logit = use_logit
        self.use_feature = use_feature

    def save_logit(self, logit, idx=None):
        if idx is None:
            self.logits.append(logit)
        else:
            self.logits[idx] = logit
        self.logits_mask.append(torch.ones_like(logit))

    def save_feature(self, feature, idx=None):
        if idx is None:
            self.features.append(feature)
        else:
            self.features[idx] = feature

    def add_new_class(self, cls_list):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.cls_dict = {self.cls_list[i]:i for i in range(len(self.cls_list))}
        self.cls_train_cnt = np.append(self.cls_train_cnt, 0)
        for i, logit in enumerate(self.logits):
            self.logits[i] = torch.cat([logit, torch.zeros(1).to(self.device)])
            self.logits_mask[i] = torch.cat([self.logits_mask[i], torch.zeros(1).to(self.device)])

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=False, transform=None):

        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_images))
        batch_size = min(batch_size, stream_batch_size + len(self.images))
        memory_batch_size = batch_size - stream_batch_size
        if memory_batch_size > 0:
            if use_weight:
                weight = self.get_weight()
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, p=weight / np.sum(weight),
                                           replace=False)
            else:
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, replace=False)
        if stream_batch_size > 0:
            if len(self.stream_images) > stream_batch_size:
                stream_indices = np.random.choice(range(len(self.stream_images)), size=stream_batch_size, replace=False)
            else:
                stream_indices = np.arange(len(self.stream_images))

        data = dict()
        images = []
        labels = []
        logits = []
        features = []
        logit_masks = []
        if self.use_kornia:
            # images
            if stream_batch_size > 0:
                for i in stream_indices:
                    images.append(self.stream_images[i])
                    labels.append(self.stream_labels[i])
            if memory_batch_size > 0:
                for i in indices:
                    images.append(self.images[i])
                    labels.append(self.labels[i])
                    if self.use_logit:
                        logits.append(self.logits[i])
                        logit_masks.append(self.logits_mask[i])
                    if self.use_feature:
                        features.append(self.features[i])
            images = torch.stack(images).to(self.device)
            images = self.transform_gpu(images)
        else:
            if stream_batch_size > 0:
                for i in stream_indices:
                    if transform is None:
                        if self.transform_on_gpu:
                            images.append(self.transform_gpu(self.stream_images[i].to(self.device)))
                        else:
                            images.append(self.transform(self.stream_images[i]))
                    else:
                        if self.transform_on_gpu:
                            images.append(transform(self.stream_images[i].to(self.device)))
                        else:
                            images.append(transform(self.stream_images[i]))
                    labels.append(self.stream_labels[i])
            if memory_batch_size > 0:
                for i in indices:
                    if transform is None:
                        if self.transform_on_gpu:
                            images.append(self.transform_gpu(self.images[i].to(self.device)))
                        else:
                            images.append(self.transform(self.images[i]))
                    else:
                        if self.transform_on_gpu:
                            images.append(transform(self.images[i].to(self.device)))
                        else:
                            images.append(transform(self.images[i]))
                    labels.append(self.labels[i])
                    if self.use_logit:
                        logits.append(self.logits[i])
                        logit_masks.append(self.logits_mask[i])
                    if self.use_feature:
                        features.append(self.features[i])

            images = torch.stack(images)
        data['image'] = images
        data['label'] = torch.LongTensor(labels)
        if memory_batch_size > 0:
            if self.use_logit:
                data['logit'] = torch.stack(logits)
                data['logit_mask'] = torch.stack(logit_masks)
            if self.use_feature:
                data['feature'] = torch.stack(features)
        else:
            if self.use_logit:
                data['logit'] = torch.zeros(1)
                data['logit_mask'] = torch.zeros(1)
            if self.use_feature:
                data['feature'] =torch.zeros(1)
        if self.keep_history:
            self.previous_idx = np.append(self.previous_idx, indices)
        return data


def get_train_datalist(dataset, sigma, repeat, init_cls, rnd_seed):
    with open(f"collections/{dataset}/{dataset}_sigma{sigma}_repeat{repeat}_init{init_cls}_seed{rnd_seed}.json") as fp:
        train_list = json.load(fp)
    return train_list['stream'], train_list['cls_dict'], train_list['cls_addition']

def get_test_datalist(dataset) -> List:
    return pd.read_json(f"collections/{dataset}/{dataset}_val.json").to_dict(orient="records")


def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    if dataset == 'imagenet':
        dataset = 'imagenet1000'
    assert dataset in [
        "mnist",
        "KMNIST",
        "EMNIST",
        "FashionMNIST",
        "SVHN",
        "cifar10",
        "cifar100",
        "CINIC10",
        "imagenet100",
        "imagenet1000",
        "tinyimagenet",
    ]
    mean = {
        "mnist": (0.1307,),
        "KMNIST": (0.1307,),
        "EMNIST": (0.1307,),
        "FashionMNIST": (0.1307,),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "CINIC10": (0.47889522, 0.47227842, 0.43047404),
        "tinyimagenet": (0.4802, 0.4481, 0.3975),
        "imagenet100": (0.485, 0.456, 0.406),
        "imagenet1000": (0.485, 0.456, 0.406),
    }

    std = {
        "mnist": (0.3081,),
        "KMNIST": (0.3081,),
        "EMNIST": (0.3081,),
        "FashionMNIST": (0.3081,),
        "SVHN": (0.1969, 0.1999, 0.1958),
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "CINIC10": (0.24205776, 0.23828046, 0.25874835),
        "tinyimagenet": (0.2302, 0.2265, 0.2262),
        "imagenet100": (0.229, 0.224, 0.225),
        "imagenet1000": (0.229, 0.224, 0.225),
    }

    classes = {
        "mnist": 10,
        "KMNIST": 10,
        "EMNIST": 49,
        "FashionMNIST": 10,
        "SVHN": 10,
        "cifar10": 10,
        "cifar100": 100,
        "CINIC10": 10,
        "tinyimagenet": 200,
        "imagenet100": 100,
        "imagenet1000": 1000,
    }

    in_channels = {
        "mnist": 1,
        "KMNIST": 1,
        "EMNIST": 1,
        "FashionMNIST": 1,
        "SVHN": 3,
        "cifar10": 3,
        "cifar100": 3,
        "CINIC10": 3,
        "tinyimagenet": 3,
        "imagenet100": 3,
        "imagenet1000": 3,
    }

    inp_size = {
        "mnist": 28,
        "KMNIST": 28,
        "EMNIST": 28,
        "FashionMNIST": 28,
        "SVHN": 32,
        "cifar10": 32,
        "cifar100": 32,
        "CINIC10": 32,
        "tinyimagenet": 64,
        "imagenet100": 224,
        "imagenet1000": 224,
    }
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        inp_size[dataset],
        in_channels[dataset],
    )


# from https://github.com/drimpossible/GDumb/blob/74a5e814afd89b19476cd0ea4287d09a7df3c7a8/src/utils.py#L102:5
def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5, z=None):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    if z is not None:
        z_a, z_b = z, z[index]
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    if z is None:
        return x, y_a, y_b, lam
    else:
        return x, y_a, y_b, lam, z_a, z_b


def cutmix_feature(x, y, feature, prob, weight, alpha=1.0, cutmix_prob=0.5):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    feature_a, feature_b = feature, feature[index]
    prob_a, prob_b = prob, prob[index]
    weight_a, weight_b = weight, weight[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, feature_a, feature_b, prob_a, prob_b, weight_a, weight_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
