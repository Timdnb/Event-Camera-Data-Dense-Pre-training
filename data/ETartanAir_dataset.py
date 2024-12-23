import os
os.environ["KMP_BLOCKTIME"] = "0"
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
import torch.utils.data.dataset as dataset
from .event_utils import eventsToVoxel
from .file_io import read_event_h5

def isin(i,xs):
    for x in xs:
        if x in i:
            return True
    return False

TRAIN_SCENCE = ['westerndesert', 'seasidetown', 'amusement', 'carwelding', 'seasonsforest', 'office2', 'japanesealley', 'ocean', 'abandonedfactory_night', 'endofworld', 'office', 'soulcity', 'oldtown', 'seasonsforest_winter', 'abandonedfactory']
TEST_SCENCE = ['hospital','gascola', 'neighborhood']

import torch
import torch.nn as nn
import numpy as np
import kornia.augmentation as korniatfm
import math
from functools import partial
import random

class Transform(nn.Module):
    def __init__(self, cfg, cfg_ibot, downsample=32):
        
        super().__init__()
       
        global_crops_scale = cfg.get("global_crops_scale", [0.32,1.0])
        local_crops_scale = cfg.get("local_crops_scale", [0.05,0.32])
        global_crops_size = cfg.get("global_crops_size", 192)
        local_crops_size = cfg.get("local_crops_size", 96) 
        input_size = cfg.get("input_size", 224)
        same_on_batch = False
        
        # self.input = korniatfm.RandomCrop((input_size,input_size)) # -> this crop is now implemented in dataloader
        
        # align_corners=True is important for the grid_sample function
        teacher1 = [korniatfm.RandomResizedCrop((global_crops_size, global_crops_size), scale=global_crops_scale, same_on_batch=same_on_batch,cropping_mode="slice",align_corners=True)] 
        teacher1 += [korniatfm.RandomHorizontalFlip(p=0.5,same_on_batch=same_on_batch)]
        
        teacher2 = [korniatfm.RandomResizedCrop((global_crops_size, global_crops_size), scale=global_crops_scale, same_on_batch=same_on_batch,cropping_mode="slice",align_corners=True)] 
        teacher2 += [korniatfm.RandomHorizontalFlip(p=0.5,same_on_batch=same_on_batch)]
        
        
        student = [korniatfm.RandomResizedCrop((local_crops_size, local_crops_size), scale=local_crops_scale, same_on_batch=same_on_batch,cropping_mode="slice", align_corners=True)] 
        student += [korniatfm.RandomHorizontalFlip(p=0.5,same_on_batch=same_on_batch)]
    
        self.student = student
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.local_crops_size = local_crops_size
        self.global_crops_size = global_crops_size
        self.input_size = input_size
        grid_h, grid_w = np.meshgrid(range(input_size),range(input_size), indexing='ij')
        grid = torch.from_numpy(np.stack([grid_w, grid_h])).float().unsqueeze(0)
        # register_buffers makes sure grid is stored in the state_dict, but not updated by optimizers
        # +1 ensures that indexing starts from 1, e.g. top left is (1, 1)
        self.register_buffer("grid",grid + 1)
        self.downsample = downsample
        self.cfg_ibot = cfg_ibot
        
    @torch.no_grad()
    def forward(self, img):
        # img = self.input(img) # -> this crop is now implemented in dataloader
        assert self.input_size == img.size(2) and self.input_size == img.size(3), f"{img.size()}"
        
        # b = batch
        b = img.size(0)
        # the 2 below defines the number of global crops
        teacher = img.repeat(2,1,1,1) 
        # -> [B*n_global_crops, ebins, 224, 224]
        t_grid = self.grid.expand(b * 2,-1,-1,-1)
        # -> [B*n_global_crops, 2, 224, 224], the 2 is for the grid_w and grid_h
        
        for i, f in enumerate(self.teacher1):
            teacher = f(teacher)
            # use f._params to ensurer that the same augmentation is applied to the grid
            t_grid = f(t_grid,params=f._params)

        # undo index shift  
        t_grid = t_grid - 1
        # normalize to [-1, 1]
        t_grid = 2.0 * t_grid/ max(self.input_size - 1, 1) - 1.0
        # interpolate to downsample size
        t_grid = torch.nn.functional.interpolate(t_grid, self.global_crops_size//self.downsample,mode="bilinear", align_corners=True) #torch.nn.functional.adaptive_avg_pool2d(t1_grid, self.global_crops_size//32)
        # permute to match the format expected by the model
        # (B, C, H, W) -> (B, H, W, C)
        t_grid = t_grid.permute(0, 2, 3, 1)
        

        student = img.repeat(8,1,1,1)
        # the 8 below defines the number of local crops
        s_grid = self.grid.expand(b * 8,-1,-1,-1)
        
        student, s_grid = self.forward_imp(student, s_grid)

        # ----- below is what originally was the collate_data() function
        mask_ratio_tuple = tuple(self.cfg_ibot["mask_ratio_min_max"])
        mask_probability = self.cfg_ibot["mask_sample_probability"]

        device = teacher.device

        B = len(teacher)

        n_tokens=8 # IF CHANING THIS. ALSO CHANGE IN pretrain_model.py line 38
        N = n_tokens**2

        n_samples_masked = int(B * mask_probability)
        probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
        upperbound = 0
        masks_list = []
        for i in range(0, n_samples_masked):
            prob_min = probs[i]
            prob_max = probs[i + 1]
            # single mask is a 12x12 boolean tensor
            masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
            upperbound += int(N * prob_max)
        for i in range(n_samples_masked, B):
            masks_list.append(torch.BoolTensor(mask_generator(0)))

        # masks_list is B*num_global_crops times [12x12]
        random.shuffle(masks_list)

        # collated_masks is [B*num_global_crops times, 144]
        collated_masks = torch.stack(masks_list).flatten(1).to(device)
        # only keep the indices of the masked patches
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

        # masks_weight is used to balance the loss when the number of masked patches is different
        # this ensures that an image with many masked patches does not contribute more to the loss
        # is this desired behavior?
        # calculation comes down to: 1 / number of masked patches for the specific mask, then this weight is assigned 
        # to each patch of that mask
        masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

        return {
            "collated_global_crops": teacher,
            "collated_local_crops": student,
            "s_grid": s_grid,
            "t_grid": t_grid,
            "img": img,
            "collated_masks": collated_masks,
            "mask_indices_list": mask_indices_list,
            "masks_weight": masks_weight,
            "upperbound": upperbound,
            "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long).to(device),
        }
    
    def forward_imp(self, s, g):
        
        downsample = partial(torch.nn.functional.interpolate, size = self.local_crops_size//self.downsample, mode="bilinear", align_corners=True) #partial(torch.nn.functional.adaptive_avg_pool2d, output_size = self.local_crops_size//32)
        
        for f in self.student:
            s = f(s)
            g = f(g,params=f._params)

        g = g - 1
        g = 2.0 * g/ max(self.input_size - 1, 1) - 1.0
        g = downsample(g) 
        g = g.permute(0, 2, 3, 1)

        return s, g

class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    # a __repr__ function is used to print the object in a human readable format
    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            # no clue why the log is undone here
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        return mask

class TartanairPretrainDataset(dataset.Dataset):
    def __init__(self, args, all_args, train = False, aug_params=None):
        super().__init__()
        self.args = args
        self.event_bins = args.event_bins
        self.precomputed = args.precomputed
        self.event_polarity = False if args.no_event_polarity else True
        self.train = train
        
        self.aug_params = aug_params

        self.fetch_valids()
        self.data_length = len(self.data)

        # FROM pretrain_model.py
        aug_downsample = 16
        
        global mask_generator
        N = all_args["network"]["cfg"]["crops"]["global_crops_size"] // aug_downsample
        mask_generator = MaskingGenerator(
            input_size=(N, N),
            max_num_patches=0.5 * N * N,
        )
        self.aug = Transform(all_args["network"]["cfg"]["crops"], all_args["network"]["cfg"]["ibot"], aug_downsample)

    def fetch_valids(self):
        data = [i.strip().split(" ") for i in open(self.args.file, 'r').readlines()]
        scence = TRAIN_SCENCE if self.train else TEST_SCENCE 
        data = [i for i in data if isin(i[1],scence)]
        self.data = data

    def load_data_by_index(self, index):
        event_filename = self.data[index][1]

        if self.precomputed:
            event_filename = event_filename.replace("event_left", "voxels_{}".format(self.event_bins))
            event_filename = event_filename.replace(".hdf5", ".pt")

            return torch.load(event_filename)
        
        if event_filename.endswith(".npy"):
            return np.load(event_filename)
        else:
            return read_event_h5(event_filename)
    
    def crop(self, event):

        crop_size = self.aug_params['crop_size']
        
        height, width = crop_size
        
        if self.train:
            y0 = np.random.randint(0, 480 - crop_size[0]) 
            x0 = np.random.randint(0, 640 - crop_size[1])
        else:
            y0 = (480 - crop_size[0])//2
            x0 = (640 - crop_size[1])//2

        if not self.precomputed:
            if len(event.shape)==2:
                valid_events = (event[:, 0] >= x0) & (event[:, 0] <= x0 + crop_size[1] - 1) &\
                            (event[:, 1] >= y0) & (event[:, 1] <= y0 + crop_size[0] - 1)
        
                event = event[valid_events]
                event[:,0] = event[:,0] - x0
                event[:,1] = event[:,1] - y0
                
                if event.shape[0] < 10:
                    c = 1 + int(self.event_polarity)
                    event  = np.zeros((self.event_bins*c,height,width))
                else:
                    event = eventsToVoxel(event, num_bins=self.event_bins, height=height,
                                                    width=width, event_polarity=self.event_polarity, temporal_bilinear=True)
            else:
                event = event[...,y0:y0+crop_size[0], x0:x0+crop_size[1]]
                
            event = torch.from_numpy(event)
        else:
            event = event[:, y0:y0+crop_size[0], x0:x0+crop_size[1]]
        return event
    
    def batch_aug(self, event):
        events = torch.stack(event)
        return self.aug(events)

    def __getitem__(self, index):
        index = index % self.data_length
        events1_nparray = \
            self.load_data_by_index(index)
        event = self.crop(events1_nparray)
        return event    
    
    def __len__(self):
        return self.data_length