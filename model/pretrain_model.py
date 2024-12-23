import torch
import torch.nn as nn
from .network.swin_pretrain import SWIN
from .network.layers import DINOHead
from .network.loss import DINOLoss, iBOTPatchLoss
from omegaconf import OmegaConf
from functools import partial
from torch_scatter import scatter_mean
from copy import deepcopy

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

class BaseModel(nn.Module):
    def __init__(self, cfg=None, n_tokens=8, *args,**kwargs): # WHEN CHANGING n_tokens, ALSO CHANGE IN ETartanAir_dataset.py line 139
        super().__init__()
        
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
                

class PretrainModel(BaseModel):
    def __init__(self, num_clusters = 8, aug_downsample=16, *args, **kwargs): # WHEN CHANING aug_downsample, ALSO CHANGE IN pretrain_model.py line 262
        N = kwargs["cfg"]["crops"]["global_crops_size"] // aug_downsample
        mask_dim = N
        super().__init__(mask_dim=mask_dim,*args,**kwargs)
        self.num_clusters = num_clusters

    def forward(self, img, m, teacher_temp):
        self.update_teacher(m)
        return self.forward_imp(img,teacher_temp)
        
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