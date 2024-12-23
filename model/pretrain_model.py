import torch
import torch.nn as nn
import numpy as np
import kornia.augmentation as korniatfm
from .network.swin_pretrain import SWIN
from .network.layers import DINOHead
from .network.loss import DINOLoss, iBOTPatchLoss
import math
from omegaconf import OmegaConf
from functools import partial
import random
from torch_scatter import scatter_mean
from copy import deepcopy

class Transform(nn.Module):
    def __init__(self, cfg, downsample=32):
        
        super().__init__()
       
        global_crops_scale = cfg.get("global_crops_scale", [0.32,1.0])
        local_crops_scale = cfg.get("local_crops_scale", [0.05,0.32])
        global_crops_size = cfg.get("global_crops_size", 192)
        local_crops_size = cfg.get("local_crops_size", 96) 
        input_size = cfg.get("input_size", 224)
        same_on_batch = False
        
        self.input = korniatfm.RandomCrop((input_size,input_size))
        
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
        
    @torch.no_grad()
    def forward(self, img):
        
        img = self.input(img)
        
        assert self.input_size == img.size(2) and self.input_size == img.size(3), f"{img.size()}"
        
        # b = batch
        b = img.size(0)
        # the 2 below defines the number of global crops
        teacher = img.repeat(2,1,1,1) 
        # -> [B*n_global_crops, ebins, 224, 224]
        t_grid = self.grid.expand(b * 2,-1,-1,-1)
        # -> [B*n_global_crops, 2, 224, 224], the 2 is for the grid_w and grid_h

        # fill with [top_left, bottom_right] coordinates
        global_crop_pos = []
        
        for i, f in enumerate(self.teacher1):
            teacher = f(teacher)
            # use f._params to ensurer that the same augmentation is applied to the grid
            t_grid = f(t_grid,params=f._params)

            if i == 0:
                global_crop_pos.append([t_grid[:,0,0,0].clone().detach().cpu().numpy(), t_grid[:,1,0,0].clone().detach().cpu().numpy()])
                global_crop_pos.append([t_grid[:,0,-1,-1].clone().detach().cpu().numpy(), t_grid[:,1,-1,-1].clone().detach().cpu().numpy()])

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

        # fill with [top_left, bottom_right] coordinates
        local_crop_pos = []
        
        student, s_grid, local_crop_pos = self.forward_imp(student, s_grid, local_crop_pos)
        output = dict(
            img = img,
            global_crops = teacher,
            local_crops = student,
            s_grid = s_grid,
            t_grid = t_grid,
            )
        
        visualize_crops = False
        if visualize_crops:
            import rerun as rr
            import cv2
            rr.init("assignments")
            rr.connect_tcp('100.120.120.119:9876')

            img1 = img[0,3].clone().detach().cpu().numpy()
            img1 = np.stack([img1, img1, img1], axis=-1)
            img1 = np.where(img1 > 0, 1, 0) * 255
            img1 = img1.astype(np.uint8)
            
            gc_pos = []
            for i in range(2):
                top_left = (int(global_crop_pos[0][0][i]), int(global_crop_pos[0][1][i]))
                bottom_right = (int(global_crop_pos[1][0][i]), int(global_crop_pos[1][1][i]))
                gc_pos.append([top_left, bottom_right])
                img1 = cv2.rectangle(img1, top_left, bottom_right, (0, 255, 0), 1)

            lc_pos = []
            for i in range(8):
                top_left = (int(local_crop_pos[0][0][i]), int(local_crop_pos[0][1][i]))
                bottom_right = (int(local_crop_pos[1][0][i]), int(local_crop_pos[1][1][i]))
                lc_pos.append([top_left, bottom_right])
                img1 = cv2.rectangle(img1, top_left, bottom_right, (255, 0, 0), 1)

            check_overlap = True
            if check_overlap:
                def is_no_overlap(rect1, rect2):
                    # Check if rect1 and rect2 do not overlap
                    return (rect1[1][0] <= rect2[0][0] or  # rect1 is to the left of rect2
                            rect1[0][0] >= rect2[1][0] or  # rect1 is to the right of rect2
                            rect1[1][1] <= rect2[0][1] or  # rect1 is above rect2
                            rect1[0][1] >= rect2[1][1])    # rect1 is below rect2

                def check_no_overlap(main_rects, other_rects):
                    for other in other_rects:
                        if all(is_no_overlap(other, main) for main in main_rects):
                            return True  # Found a rectangle with no overlap
                    return False  # All rectangles overlap with at least one main rectangle
                
                result = check_no_overlap(gc_pos, lc_pos)
                print(f"Overlap between global and local crops: {result}")

                if result:
                    rr.log("crops", rr.Image(img1))
                    input("Press Enter to continue...")
            
            if not check_overlap:
                rr.log("crops", rr.Image(img1))
                input("Press Enter to continue...")

        return collate_data(output)
    
    def forward_imp(self, s, g, local_crop_pos):
        
        downsample = partial(torch.nn.functional.interpolate, size = self.local_crops_size//self.downsample, mode="bilinear", align_corners=True) #partial(torch.nn.functional.adaptive_avg_pool2d, output_size = self.local_crops_size//32)
        
        for i, f in enumerate(self.student):
            s = f(s)
            g = f(g,params=f._params)

            if i == 0:
                local_crop_pos.append([g[:,0,0,0].clone().detach().cpu().numpy(), g[:,1,0,0].clone().detach().cpu().numpy()])
                local_crop_pos.append([g[:,0,-1,-1].clone().detach().cpu().numpy(), g[:,1,-1,-1].clone().detach().cpu().numpy()])
            
        g = g - 1
        g = 2.0 * g/ max(self.input_size - 1, 1) - 1.0
        g = downsample(g) 
        g = g.permute(0, 2, 3, 1)

        return s, g, local_crop_pos

@torch.no_grad()
def batch_cosine_KMeans(X,num_clusters=6,max_iter=10):
    X = X.clone().detach()
    X = torch.nn.functional.normalize(X, dim=2)
    N, L, D = X.shape 
    noise = torch.rand(N, L, device=X.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :num_clusters]
    # ids_keep is num_clusters (=8 by default) random patch indices for each image in the batch
    centroids = torch.gather(X, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    # the centroids are the embeddings of the num_clusters (N_c) random patches for each image in the batch, [B, N_c, 768]
    
    for _ in range(max_iter):
        centroids = torch.nn.functional.normalize(centroids, dim=2)
        # calculate for each path what its distance is to each centroid, [B, N_p, N_c], N_p = number of patches (196 by default)
        dis = 1 - torch.einsum('bij,bkj->bik', X, centroids)
        assignments = torch.argmin(dis, dim=2)
        # scatter_mean averages the embeddings of the patches that are assigned to the same cluster
        # this is the new centroid
        centroids = scatter_mean(X, assignments,dim=1)
    return assignments, centroids


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

class BaseModel(nn.Module):
    def __init__(self, cfg=None, n_tokens=8, *args,**kwargs):
        super().__init__()
        global collate_data
        
        cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        self.n_tokens = n_tokens
        student_model_dict = dict()
        teacher_model_dict = dict()

        t_kwargs = deepcopy(kwargs)
        # teacher model does not have drop path
        # drop path is a regularization technique that randomly drops connections in the network (I think)
        t_kwargs["drop_path_rate"] = 0.0
        student_backbone, teacher_backbone = SWIN(*args,**kwargs), SWIN(*args,**t_kwargs)
        # print(f"Number of parameters in student backbone: {sum(p.numel() for p in student_backbone.parameters())}") # -> 30400794
        # print(f"Number of parameters in teacher backbone: {sum(p.numel() for p in teacher_backbone.parameters())}") # -> 30400794
        # not sure why * 2**3
        embed_dim = kwargs['embed_dim']  * 2**3

            
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        print(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        print("OPTIONS -- DINO")
        
        print(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
        print(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
        print(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
        print(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
        self.dino_loss_weight = cfg.dino.loss_weight
        dino_head = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
            use_attpool = True,
        )
        # use_attpool = True, here means that dino_head.att can be called, but by default it won't (need to pass pool=True)
        self.dino_loss = DINOLoss(self.dino_out_dim)

        student_model_dict["dino_head"] = dino_head()
        teacher_model_dict["dino_head"] = dino_head()

        print("OPTIONS -- IBOT")
        print(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        print(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        print(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")

        self.ibot_loss_weight = cfg.ibot.loss_weight
        assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
        assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
        self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
        self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
        if self.ibot_separate_head:
            print(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
            print(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
            print(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
            print(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
            ibot_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.ibot.head_n_prototypes,
                hidden_dim=cfg.ibot.head_hidden_dim,
                bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                nlayers=cfg.ibot.head_nlayers,
            )
            student_model_dict["ibot_head"] = ibot_head()
            teacher_model_dict["ibot_head"] = ibot_head()
        else:
            print("OPTIONS -- IBOT -- head shared with DINO")

        student_model_dict["cluster_head"] = dino_head()
        teacher_model_dict["cluster_head"] = dino_head()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        for p in self.teacher.parameters():
            p.requires_grad = False
            
        self.prepare_for_distributed_training()
        
        def collate_data(samples, mask_ratio_tuple = cfg.ibot.mask_ratio_min_max, mask_probability=cfg.ibot.mask_sample_probability): 
    
            collated_global_crops = samples["global_crops"]
            collated_local_crops = samples["local_crops"]
            img = samples["img"]
            s_grid = samples["s_grid"]
            t_grid = samples["t_grid"]
            device = collated_global_crops.device

            # B is 'new' batch size, however not the original batch size, but batch size * num_global_crops
            B = len(collated_global_crops)
            # N is not the same as number of patches (that is 12x12, this is 8x8), not sure what it is but probably arbitrary
            # value to limit the number of patches being masked
            N = self.n_tokens**2  
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

            visualize_mask = False
            if visualize_mask:
                import rerun as rr
                rr.init("assignments")
                rr.connect_tcp('100.120.120.119:9876')

                for mask_n in range(len(masks_list)//2):
                    mask = masks_list[mask_n].clone().detach().cpu().numpy()
                    mask_img = np.zeros((12*16, 12*16, 3))

                    for i in range(12):
                        for j in range(12):
                            if mask[i, j]:
                                mask_img[i*16:(i+1)*16, j*16:(j+1)*16] = 1
                    
                    rr.log(f"mask-{mask_n}", rr.Image(mask_img))
                input("Press Enter to continue...")

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
                "collated_global_crops": collated_global_crops,
                "collated_local_crops": collated_local_crops,
                "s_grid": s_grid,
                "t_grid": t_grid,
                "img": img,
                "collated_masks": collated_masks,
                "mask_indices_list": mask_indices_list,
                "masks_weight": masks_weight,
                "upperbound": upperbound,
                "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long).to(device),
            }
                

class PretrainModel(BaseModel):
    def __init__(self, num_clusters = 8, aug_downsample=16, *args, **kwargs):
        global mask_generator
        N = kwargs["cfg"]["crops"]["global_crops_size"] // aug_downsample
        mask_dim = N
        mask_generator = MaskingGenerator(
            input_size=(N, N),
            max_num_patches=0.5 * N * N,
        )
        # initialize the parent class
        # i.e. backbone and 3 heads for both the teacher and student networrk
        super().__init__(mask_dim=mask_dim,*args,**kwargs)
        self.num_clusters = num_clusters
        self.aug = Transform(kwargs["cfg"]["crops"], aug_downsample)

    def forward(self, img, m, teacher_temp):
        self.update_teacher(m)
        images = self.aug(img)
        return self.forward_imp(images,teacher_temp)
        
    def forward_imp(self, images, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"]
        local_crops = images["collated_local_crops"]
        t_grid = images["t_grid"]
        imgs = images["img"]
        
        masks = images["collated_masks"]
        mask_indices_list = images["mask_indices_list"]
        n_masked_patches_tensor = images["n_masked_patches"]
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"]

        n_local_crops_loss_terms =  n_local_crops * n_global_crops 
        n_global_crops_loss_terms = n_global_crops + n_global_crops 

        do_dino = self.do_dino
        do_ibot = self.do_ibot
    
        ibot_loss_scale = 1.0 / n_global_crops

        b = global_crops.size(0)
        
        def tomask(x,maps):
            return x.scatter_(1, maps.unsqueeze(1).long(), 1).bool()
        
        KEY = "x_norm_patchtokens" 
        
        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            global t_assign_map, t_mask, s_assign_map, s_mask, teacher_patchtokens 
            
            # teacher forward pass on input images; shape is [B, ebins, 224, 224]
            teacher_cls_tokens = self.teacher.backbone(imgs, is_training=True)[KEY]
            # teacher_cls_tokens has shape [B, N_p, 768], N_p is number of patches (=14*14=196), each patch is 16x16 pixels
            # 768 is the embedding dimension
            # then for global crops (192x192 pixels), we get 12x12 patches
            wt = int(teacher_cls_tokens.size(1)**0.5)
            
            # assign has shape [B, 196]
            assign, _ = batch_cosine_KMeans(teacher_cls_tokens, num_clusters=self.num_clusters)
            # for future reference: if we want to use a variable number of clusters, it can be changed (per batch) on the fly
            
            visualize_assignments = False
            if visualize_assignments:
                # from this vizualization:
                # patches are somewhat random and not apart from 0-2 clusters per image, the others are bad
                # maybe improve this by only keeping the clusterr which cover
                # 1. many events
                # 2. minimum consecutive area (so not randomly scattered patches)
                import rerun as rr
                rr.init("assignments")
                rr.connect_tcp('100.120.120.119:9876')
                
                ebin = 3
                im_c = imgs[0][ebin].clone().detach().cpu().numpy()
                assign_c = assign[0].clone().detach().cpu().numpy()

                # create the colored image
                colors = [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,255,255)]
                im_color = np.zeros((224,224,3))
                for i in range(14):
                    for j in range(14):
                        cluster = int(assign_c[i*14+j])
                        # cluster = int(assign_c[i+j*14])
                        im_color[i*16:i*16+16,j*16:j*16+16] = colors[cluster]

                # overlay the color pattern with the original image (which is grayscale)
                im_color = im_color/255
                overlay = 0.5*im_color + 0.5*im_c[:,:,None]
                overlay *= 255 

                rr.log(f"ebin{ebin}", rr.Image(overlay))
                input("Press Enter to continue...")

            # convert cluster assignments to boolean mask per cluster, i.e. shape [B, N_c, 196]
            attmask = tomask(torch.zeros((b//2,self.num_clusters, wt**2),  device=teacher_cls_tokens.device),assign)
            # pass though cluster attention head, output shape is [B, N_c, 768]
            teacher_cls_tokens = self.teacher.cluster_head.att(teacher_cls_tokens,self.num_clusters,mask=attmask)
            
            # repeat assignment for number of global crops [B, 196] -> [B*num_global_crops, 196]
            t_assign = assign.repeat(n_global_crops,1)
            # convert cluster assignments to mask per global crop per cluster (0 and 1s), shape [B*num_global_crops, N_c, 14, 14]
            t_assign_map = torch.zeros((b,self.num_clusters, wt, wt),  device=teacher_cls_tokens.device).scatter_(1, t_assign.view(b,wt,wt).long().unsqueeze(1), 1)
            # use grid_sample and t_grid (shape [4, 12, 12, 2]) to interpolate the mask to the global crop size [B*num_global_crops, N_c, 14, 14]
            t_assign_map = nn.functional.grid_sample(t_assign_map, t_grid, align_corners=True)
            # convert all values above 0.25 to 1, else 0, still shape [B*num_global_crops, N_c, 12, 12]
            # I think this means: if the patch is more than 25% covered by the cluster, it is part of the cluster
            t_assign_map = (t_assign_map > .25).to(t_assign_map) 
            # per cluster per global crop, check if the cluster is present in the global crop
            t_mask = t_assign_map.view(b,self.num_clusters,-1).sum(-1) >= .999
            
            # [B, N_c, 768] -> [B*N_c, 768]
            teacher_cls_tokens = teacher_cls_tokens.flatten(0,1)
            n_clusters = teacher_cls_tokens.size(0)
            # global crops are [B*num_global_crops, ebins, 192, 192]
            x = self.teacher.backbone(global_crops, is_training=True)
            # x[KEY] is [B*num_global_crops, N_p, 768], N_p is number of patches (=12*12=144), each patch is 16x16 pixels
            assert n_clusters//x[KEY].size(0) == self.num_clusters // n_global_crops
            # IMPROV?: just get x[KEY] once and then do all operations below
            
            # flattened output from cluster head [B*N_c, 768] is combined with the output from the dino head [B*num_global_crops, 768]
            teacher_cls_tokens = torch.cat((teacher_cls_tokens, self.teacher.dino_head.att(x[KEY])))

            # for the patch loss, we take the embeddings for every patch in the global crops
            ibot_teacher_patch_tokens = x["x_norm_patchtokens"]
            
            _dim = ibot_teacher_patch_tokens.shape[-1] # embedding dimension

            if do_ibot:
                # TODO: why create buffer tensor of size upperbound and slice it to size n_masked_patches 
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                # select only the embeddings of the masked patches, saved to buffer_tensor_teacher [n_masked_patches, 768]
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                # pass the attention pooled context embeddings through the cluster head (now the MLP part)
                teacher_cls_tokens_after_head_1 = self.teacher.cluster_head(teacher_cls_tokens[:n_clusters])
                # -> [B*N_c, 32768], 32768 is the output dimension of the cluster head
                # pass the attention pooled global crop embeddings through the dino head (now the MLP part)
                teacher_cls_tokens_after_head_2 = self.teacher.dino_head(teacher_cls_tokens[n_clusters:])
                # -> [B*num_global_crops, 32768], 32768 is the output dimension of the dino head
                if self.ibot_separate_head:
                    # pass the masked patch embeddings through the ibot head
                    masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                        :n_masked_patches
                    ]
                    # -> [n_masked_patches, 32768]
                else:
                    # otherwise pass the masked patch embeddings through the dino head
                    masked_teacher_patch_tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)[
                        :n_masked_patches
                    ]
                    # -> [n_masked_patches, 32768]
            else:
                raise NotImplementedError

            if self.cfg.train.centering == "centering":
                raise NotImplementedError
                
            # TODO: investigate this sinkhorn_knopp stuff
            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_c = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head_1, teacher_temp=teacher_temp
                )
                # -> [B*N_c, 32768]
                
                teacher_dino_softmaxed_centered_g = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head_2, teacher_temp=teacher_temp
                )
                # -> [B*num_global_crops, 32768]

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )
                    # -> [n_masked_patches, 32768]

            else:
                raise NotImplementedError

            return (teacher_dino_softmaxed_centered_c,teacher_dino_softmaxed_centered_g), masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        # this simply combines the two outputs from the dino heads from the teacher network
        teacher_dino_softmaxed_centered = teacher_dino_softmaxed_centered[0].chunk(self.num_clusters) + teacher_dino_softmaxed_centered[1].chunk(n_global_crops)
        # -> [num_clusters + n_global_crops, B, 32768]

        # TEACHER is done, now STUDENT
        loss_dict = {}

        loss_accumulator = 0 
        # global crops = [B*num_global_crops, ebins, 192, 192]; masks = [B*num_global_crops, 144]; local crops = [B*num_local_crops, ebins, 96, 96]
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        student_local_cls_tokens = student_local_backbone_output_dict[KEY]
        # -> [B*num_local_crops, N_p, 768], N_p is number of patches (=6*6=36), each patch is 16x16 pixels
        student_global_cls_tokens = student_global_backbone_output_dict[KEY]
        # -> [B*num_global_crops, 144, 768]
        
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_patchtokens"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                student_global_masked_patch_tokens_after_head = self.student.dino_head(buffer_tensor_patch_tokens.unsqueeze(0)).squeeze(0)[:n_masked_patches]
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]
                # -> [n_masked_patches, 32768]

        # get the global crop embeddings after the dino head from the teacher network
        teacher_global_base = torch.cat((teacher_dino_softmaxed_centered[self.num_clusters:][::-1]))
        # -> [B*num_global_crops, 32768]
        
        if n_local_crops > 0:
            # loss weight, probably to account for the way the loss term is calculated
            w = 1 / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            student_local_base = self.student.dino_head(student_local_cls_tokens,pool=True)
            
            # LOSS: this is the local/global loss (not mentioned in paper)
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_base.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_global_base.chunk(n_global_crops), 
            ) 
            dino_local_crops_loss = dino_local_crops_loss * w
            
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss
            
        loss_scales = 2

        if do_dino:
            
            w = 1 / (n_global_crops_loss_terms + n_local_crops_loss_terms)
           
            student_global_base = self.student.dino_head(student_global_cls_tokens,pool=True)
            
            # get context embeddings after the dino head from the teacher network (repeat n_global_crops times) and only keep them if the cluster is 
            # actually present in the global crop
            teacher_center = torch.cat((teacher_dino_softmaxed_centered[:self.num_clusters])).repeat(n_global_crops,1)[t_mask.flatten(0,1)]
            # -> [B*num_global_crops*N_c, 32768]
            # convert to bool (originally in 0s and 1s)
            attmask = (t_assign_map.flatten(2) > 0.5).bool()
            # -> [B*num_global_crops, N_c, 144]
        
            student_global_cls_tokens_after_head = self.student.cluster_head.att(student_global_cls_tokens,self.num_clusters,mask=attmask)[t_mask]
            student_global_cls_tokens_after_head = self.student.cluster_head(student_global_cls_tokens_after_head)
            
            # LOSS: this is both the context loss and the image loss
            dino_global_crops_loss = self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_center
                    ],  
                ) * n_global_crops + self.dino_loss(
                    student_output_list=[student_global_base],
                    teacher_out_softmaxed_centered_list=[
                        teacher_global_base
                    ],  
                ) * n_global_crops
            
            dino_global_crops_loss = dino_global_crops_loss * w

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss
            loss_dict["koleo_loss"] = dino_global_crops_loss * 0

        if do_ibot:
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # division by 2 -> counters multiplication of loss_scales, which is equal to 2
            loss_dict["ibot_loss"] = ibot_patch_loss / 2
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss
        else:
            loss_dict["ibot_loss"] = torch.zeros(1).sum().to(student_global_base)
        return loss_accumulator, loss_dict
    
    def update_teacher(self, m):
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(self.student[k].parameters(), self.teacher[k].parameters()):
                    mt.data = mt.data * m + ms.data * (1.0 - m)   
                    
    def train(self,*args,**kwargs):
        super().train(*args,**kwargs)
        self.teacher.eval()

    def prepare_for_distributed_training(self):
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())