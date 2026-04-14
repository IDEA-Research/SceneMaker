from dataclasses import dataclass, field
import numpy as np
import copy
import torch
import torch.nn.functional as F
from PIL import Image
import trimesh

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.typing import *
from craftsman.systems.utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, get_sigmas, flow_sample_pose_direct
from craftsman.utils.bg_processing import RMBG
from utils.transforms import *

import pdb

@craftsman.register("direct-unify-flow-system")
class UnifyFlowSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        compute_metric: bool = True
        visualize_mesh: bool = True
        val_samples_json: Optional[str] = None
        extract_mesh_func: str = "mc"
        vis_training: bool = False
        remove_bg: bool = False
        octree_depth: int = 7
        max_objs: int = 5
        min_pcd: int = 1000
        uncond_ratio: float = 0.1
        pretrain_pcd: str = "pcd-encoder"
        lambda_rot: float = 100.0
        lambda_trans: float = 1.0
        lambda_size: float = 1.0
        lambda_kl: float = 0.0
        lambda_cd: float = 0.0
        lambda_vae: float = 0.0
        sup_latents: bool = False
        loss_mode: str = "object"
        use_scene_img: bool = False
        use_scene_mask: bool = False
        use_scene_mask_rectify: bool = False
        use_scene_mask_img: bool = False
        use_scene_pcd: bool = False
        use_deocc_image: bool = False
        use_caption: bool = False
        freeze_pose_enc: bool = False
        freeze_pcd_model: bool = False

        # diffusion config
        z_scale_factor: float = 1.0
        guidance_scale: float = 7.5
        num_inference_steps: int = 50
        eta: float = 0.0
        snr_gamma: float = 5.0
        # flow
        weighting_scheme: str = "logit_normal"
        logit_mean: float = 0
        logit_std: float = 1.0
        mode_scale: float = 1.29
        precondition_outputs: bool = True
        precondition_t: int = 1000
        pretrained_model_name_or_path: str = ""

        # shape vae model
        shape_model_type: Optional[str] = None
        shape_model: dict = field(default_factory=dict)
        
        # pose model
        pose_model_type: Optional[str] = None
        pose_model: dict = field(default_factory=dict)
        
        # pcd model
        pcd_model_type: Optional[str] = None
        pcd_model: dict = field(default_factory=dict)

        # condition model
        condition_model_type: Optional[str] = None
        condition_model: dict = field(default_factory=dict)
        
        # caption model
        caption_condition_type: Optional[str] = None
        caption_condition: dict = field(default_factory=dict)

        # diffusion model
        denoiser_model_type: Optional[str] = None
        denoiser_model: dict = field(default_factory=dict)

        # noise scheduler
        noise_scheduler_type: Optional[str] = None
        noise_scheduler: dict = field(default_factory=dict)

        # denoise scheduler
        denoise_scheduler_type: Optional[str] = None
        denoise_scheduler: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        super().configure()

        # shape model
        self.shape_model = craftsman.find(self.cfg.shape_model_type)(self.cfg.shape_model)
        self.shape_model.eval()
        self.shape_model.requires_grad_(False)
        
        # pose model
        self.pose_model = craftsman.find(self.cfg.pose_model_type)(self.cfg.pose_model)
        self.pose_emb = torch.nn.Linear(self.cfg.pose_model.embed_dim, self.cfg.denoiser_model.context_dim)
        
        # pcd model
        if self.cfg.pretrain_pcd == "shape2vec":
            self.pcd_model = craftsman.find(self.cfg.pcd_model_type)(self.cfg.pcd_model)
            if self.cfg.freeze_pcd_model:
                self.pcd_model.eval()
                self.pcd_model.requires_grad_(False)
            self.pcd_emb = torch.nn.Linear(self.cfg.pcd_model.embed_dim, self.cfg.denoiser_model.context_dim, bias=True)
        else:
            self.pcd_model = craftsman.find(self.cfg.pcd_model_type)(self.cfg.pcd_model)

        # visual model
        self.condition = craftsman.find(self.cfg.condition_model_type)(self.cfg.condition_model)
        self.condition.eval()
        self.condition.requires_grad_(False)
        
        # caption model
        if self.cfg.use_caption:
            self.caption_condition = craftsman.find(self.cfg.caption_condition_type)(self.cfg.caption_condition)
            self.caption_condition.eval()
            self.caption_condition.requires_grad_(False)
            self.caption_emb = torch.nn.Linear(self.cfg.caption_condition.caption_condition_dim, self.cfg.denoiser_model.context_dim)
        
        # denoiser model
        self.denoiser_model = craftsman.find(self.cfg.denoiser_model_type)(self.cfg.denoiser_model)
        self.noise_scheduler = craftsman.find(self.cfg.noise_scheduler_type)(**self.cfg.noise_scheduler)
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.denoise_scheduler = craftsman.find(self.cfg.denoise_scheduler_type)(**self.cfg.denoise_scheduler)
        
        # metrics
        self.metrics_list = []
        
        if self.cfg.pretrained_model_name_or_path != "":
            print(f"Loading pretrained flow model from {self.cfg.pretrained_model_name_or_path}")
            pretrained_ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if 'state_dict' in pretrained_ckpt:
                pretrained_ckpt = pretrained_ckpt['state_dict']
            self.load_state_dict(pretrained_ckpt, strict=False)

    def forward(self, batch: Dict[str, Any], skip_noise=False) -> Dict[str, Any]:
        # 1. encode shape latents
        # reshape (B, T, N, 3) -> (B*T, N, 3)
        bs = batch["surface"].shape[0]
        if "sharp_surface" in batch.keys():
            sharp_surface = batch["sharp_surface"].view(-1, *batch["sharp_surface"].shape[2:])
            sharp_surface = sharp_surface[..., :3 + self.cfg.shape_model.point_feats]
        else:
            sharp_surface = None
        surface = batch["surface"].view(-1, *batch["surface"].shape[2:])
        shape_embeds, kl_embed, _ = self.shape_model.encode(
            surface[..., :3 + self.cfg.shape_model.point_feats], 
            sample_posterior=True,
            sharp_surface=sharp_surface, 
        )
        shape_latents = kl_embed * self.cfg.z_scale_factor
        shape_latents = shape_latents.view(bs, self.cfg.max_objs, *shape_latents.shape[-2:]) # [B, T, N, C]

        # 2. gain visual condition
        if "image" in batch and batch['image'].dim() == 5:
            if self.training:
                bs, n_images = batch['image'].shape[:2]
                batch['image'] = batch['image'].view(bs*n_images, *batch['image'].shape[-3:])
            else:
                batch['image'] = batch['image'][:, 0, ...]
                n_images = 1
                bs = batch['image'].shape[0]
            cond_latents = self.condition.forward_visual_embeds(batch).to(shape_latents)
        else:
            cond_latents = self.condition.forward_visual_embeds(batch).to(shape_latents)
        cond_latents = cond_latents.reshape(bs, n_images, *cond_latents.shape[-2:]) # [B, T, N, C]
        
        # scene image latents
        if self.cfg.use_scene_img:
            scene_img_latents = self.condition.encode_image(batch["whole_img"]).to(shape_latents).reshape(bs, 1, *cond_latents.shape[-2:]).repeat(1,n_images,1,1)
            if self.cfg.use_scene_mask:
                scene_mask = batch["masks"].reshape(bs*n_images,*batch["masks"].shape[-2:],1).repeat(1,1,1,3)
                scene_mask_latents = self.condition.encode_image(scene_mask).to(shape_latents).reshape(bs, n_images, *cond_latents.shape[-2:])
                scene_img_latents = torch.cat([scene_img_latents, scene_mask_latents], dim=-2) # concat
            elif self.cfg.use_scene_mask_img:
                # apply mask on scene image
                scene_mask = batch["masks"].reshape(bs*n_images,*batch["masks"].shape[-2:],1).repeat(1,1,1,3)
                scene_obj_img = batch["whole_img"].repeat_interleave(n_images, dim=0) * scene_mask
                scene_mask_latents = self.condition.encode_image(scene_obj_img).to(shape_latents).reshape(bs, n_images, *cond_latents.shape[-2:])
                scene_img_latents = torch.cat([scene_img_latents, scene_mask_latents], dim=-2) # concat
            elif self.cfg.use_deocc_image:
                # deocc image
                deocc_image = batch["deocc_images"].reshape(bs*n_images,*batch["deocc_images"].shape[-3:])
                scene_mask_latents = self.condition.encode_image(deocc_image).to(shape_latents).reshape(bs, n_images, *cond_latents.shape[-2:])
                scene_img_latents = torch.cat([scene_img_latents, scene_mask_latents], dim=-2)
        
        # scene pcd latents
        if self.cfg.use_scene_pcd:
            scene_pcds = batch["scene_pcds"].reshape(bs*self.cfg.max_objs,-1,3).to(shape_latents)
            if self.cfg.pretrain_pcd == "shape2vec":
                scene_pcd_kl_embed, _ = self.pcd_model.encode(scene_pcds, sample_posterior=True)
                scene_pcd_latents = self.pcd_emb(scene_pcd_kl_embed)
            else:
                scene_pcd_latents = self.pcd_model.encode(scene_pcds)
            scene_pcd_latents = scene_pcd_latents.view(bs, -1, *scene_pcd_latents.shape[-2:]) # [B, T, N, C]
            
        ## 2.1 text condition if provided
        if "caption" in batch and self.cfg.use_caption:
            assert "caption" in batch.keys(), "caption is required for caption encoder"
            assert bs == len(batch["caption"]), "Batch size must be the same as the caption length."
            caption_list = [item for sublist in batch["caption"] for item in sublist]
            caption_latents = self.caption_condition.encode_text(caption_list).to(shape_latents)
            caption_latents = caption_latents.reshape(bs, -1, *caption_latents.shape[-2:])
            caption_latents = self.caption_emb(caption_latents)
        
        # pcd condition
        if self.cfg.pretrain_pcd == "shape2vec":
            pcd = batch["obj_pcds"].reshape(bs*self.cfg.max_objs,-1,3).to(shape_latents)
            pcd_kl_embed, _ = self.pcd_model.encode(pcd, sample_posterior=True)
            pcd_latents = self.pcd_emb(pcd_kl_embed)
        else:
            pcd_latents = self.pcd_model.encode(batch["obj_pcds"].reshape(bs*self.cfg.max_objs,-1,3).to(shape_latents))
        pcd_latents = pcd_latents.view(bs, -1, *pcd_latents.shape[-2:]) # [B, T, N, C]
        
        # pcd size and translation
        pcd_pose_latents = self.pose_model.encode_size_translation(size=batch["pcd_sizes"]-1, translation=batch["pcd_trans"]).to(shape_latents)
        pcd_pose_latents = self.pose_emb(pcd_pose_latents)
     
        # pose latents
        rotation, translation, size = batch["pose"], batch["translation"], batch["size"]
        size = size - 1 # normalize size to [-1, 1]

        # denoised latents
        pose_latents = torch.cat([rotation, translation, size], dim=-1)  # [B, T, N, 6+3+1]
        latents = pose_latents

        # 3. sample noise that we"ll add to the latents
        noise = torch.randn_like(latents).to(latents) # [batch_size, n_objs, 1, latent_dim]
        
        # 4. Sample a random timestep for each motion
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.cfg.weighting_scheme,
            batch_size=bs,
            logit_mean=self.cfg.logit_mean,
            logit_std=self.cfg.logit_std,
            mode_scale=self.cfg.mode_scale,
        )
        indices = (u * self.cfg.noise_scheduler.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=latents.device)
        # timesteps = torch.ones_like(timesteps) * 1000

        # 5. add noise
        sigmas = get_sigmas(self.noise_scheduler_copy, timesteps, n_dim=3, dtype=latents.dtype)
        noisy_z = (1.0 - sigmas) * latents + sigmas * noise   
        
        # encode latents
        noisy_latents = self.pose_model.encode(rotation=noisy_z[...,:6], translation=noisy_z[...,6:9], size=noisy_z[...,9:]).to(shape_latents)
        
        # concat shape latents
        noisy_latents = torch.cat([noisy_latents, shape_latents], dim=-2)        
        
        # conditional latents
        cond_latents = torch.cat([cond_latents, pcd_latents, pcd_pose_latents], dim=-2)
        if self.cfg.use_scene_img:
            cond_latents = torch.cat([cond_latents, scene_img_latents], dim=-2)
        if self.cfg.use_scene_pcd:
            cond_latents = torch.cat([cond_latents, scene_pcd_latents], dim=-2)
        if self.cfg.use_caption:
            cond_latents = torch.cat([cond_latents, caption_latents], dim=-2)
        
        # random mask based on uncond_ratio
        if self.training:
            uncond_mask = (torch.rand((bs,)) < self.cfg.uncond_ratio).to(shape_latents).bool()
            cond_latents = torch.where(uncond_mask[:,None,None,None], torch.zeros_like(cond_latents).to(shape_latents), cond_latents)
        
        # if use scene-level self-attn apply mask
        attention_kwargs = {"mask_legal": batch["mask_legal"]} if self.denoiser_model.cfg.attn_mode == "scene" else {"mask_legal": torch.ones_like(batch["mask_legal"])}
        
        # 6. diffusion model forward
        output_latents = self.denoiser_model(noisy_latents, timesteps.long(), cond_latents, attention_kwargs=attention_kwargs)
        # split condtion
        output, _ = output_latents.split([3, shape_latents.shape[-2]], dim=-2)
        
        # decode outputs
        output = self.pose_model.decode(output)

        # 7. compute loss
        if self.cfg.precondition_outputs:
            output = output * (-sigmas) + noisy_z
       
        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.cfg.weighting_scheme, sigmas=sigmas)
        
        # flow matching loss
        target, target_trans, target_size = batch["pose"], batch["translation"], batch["size"]
        
        # supervise on pose values
        rot_out, trans_out, size_out = output.split([6, 3, 1], dim=-1)  # [B, T, N, 6+3+1]
        size_out = size_out + 1  # normalize size to [0, 2]
        
        # Compute regular loss.
        loss_rot = (rot_out.float() - target.float()) ** 2
        loss_trans = (trans_out.float() - target_trans.float()) ** 2
        loss_size = (size_out.float() - target_size.float()) ** 2
        
        # weighting
        loss_rot = weighting * loss_rot
        loss_trans = weighting * loss_trans
        loss_size = weighting * loss_size
        
        # mask
        mask_legal = batch["mask_legal"].unsqueeze(-1).float()
        num_legal = torch.sum(batch["mask_legal"].float(), dim=-1)
        loss_rot = torch.sum((loss_rot * mask_legal).mean(-1), dim=-1) / (num_legal+1e-4)
        loss_trans = torch.sum((loss_trans * mask_legal).mean(-1), dim=-1) / (num_legal+1e-4)
        loss_size = torch.sum((loss_size * mask_legal).mean(-1), dim=-1) / (num_legal+1e-4)
        loss = self.cfg.lambda_rot * loss_rot + loss_trans * self.cfg.lambda_trans + loss_size * self.cfg.lambda_size
    
            
        # cd loss
        if self.cfg.lambda_cd != 0.0:
            # whole obj
            with torch.no_grad():
                pts = batch["surface"][...,:3].reshape(bs*self.cfg.max_objs, -1, 3).to(output)
                # random sample points
                if pts.shape[1] >= self.cfg.min_pcd:
                    idx = torch.randperm(pts.shape[1], device=pts.device)[:self.cfg.min_pcd]
                    pts_sampled = pts[:, idx, :]  # [B*T, min_pcd, 3]
                else:
                    idx = torch.randint(0, pts.shape[1], (pts.shape[0], self.min_pcd), device=pts.device)
                    pts_sampled = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, 3))  # [B*T, min_pcd, 3]
            
            # apply on all
            gt_pts = self.apply_transform(pts_sampled, target.reshape(bs*self.cfg.max_objs,6), target_trans.reshape(bs*self.cfg.max_objs,3), target_size.reshape(bs*self.cfg.max_objs,1))
            pred_pts = self.apply_transform(pts_sampled, rot_out.reshape(bs*self.cfg.max_objs,6), trans_out.reshape(bs*self.cfg.max_objs,3), size_out.reshape(bs*self.cfg.max_objs,1))
            
            # check nan
            gt_pts[torch.isnan(gt_pts)] = 0.0
            pred_pts[torch.isnan(pred_pts)] = 0.0
            
            # loss map one by one
            cd_loss = torch.mean((gt_pts - pred_pts) ** 2, dim=(-2,-1))  # [B*T, min_pcd]
            
            # reshape to [B, T]
            cd_loss = cd_loss.reshape(bs, self.cfg.max_objs)  # [B, T]
            cd_loss[torch.isnan(cd_loss)] = 0.0 # check nan
            cd_loss = torch.sum((cd_loss * batch["mask_legal"].float()), dim=-1) / (num_legal+1e-4)
            loss = loss + cd_loss * self.cfg.lambda_cd
        
        # check nan
        nan_mask = ~torch.isnan(loss)
        if nan_mask.sum() == 0:
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
            print("Warning: loss is all nan, set to 0.0")
        else:
            loss = loss[nan_mask]
            loss = loss.mean()        
        
        # log
        self.log("train/loss_rot", loss_rot.mean())
        self.log("train/loss_trans", loss_trans.mean())
        self.log("train/loss_size", loss_size.mean())
        if self.cfg.lambda_cd != 0.0:
            self.log("train/loss_cd", cd_loss.mean())

        return {
            "loss_diffusion": loss,
            "output": output,
            "target": target,
            "timesteps": timesteps,
            "shape_latents": shape_latents,
            }
    
    def apply_transform(self, points, rotation_6d, translation, scale=None):
        """
        Args:
            points: [N, 3] or [B, N, 3] point cloud
            rotation_6d: [6] or [B, 6] 6D rotation
            translation: [3] or [B, 3] translation
            scale: [3] or [B, 3] scale (optional)
        Returns:
            points_transformed: transformed point cloud
        """
        points = points * scale.unsqueeze(-2) / 2
        transform_matrix = self.recover_transform_matrix(rotation_6d, translation)
        points_trans = torch.matmul(points.float(), transform_matrix[:,:3, :3].transpose(-1,-2)) + transform_matrix[:,:3, 3].unsqueeze(-2)

        return points_trans
    
    @torch.no_grad()
    def viz_train(self, batch, shape_latents, pred_pose, pred_trans, pred_size):
        # gt
        target, target_trans, target_size = batch["pose"], batch["translation"], batch["size"]
        bs = target.shape[0]
        
        # latent
        shape_latents = shape_latents.reshape(bs*self.cfg.max_objs, -1, shape_latents.shape[-1])
        shape = self.shape_model.decode(shape_latents)
        mesh_v_f, has_surface = self.shape_model.extract_geometry(latents=shape, extract_mesh_func=self.cfg.extract_mesh_func)
        
        # replace pred values with target values if lambda is 0
        if self.cfg.lambda_trans == 0.0:
            pred_trans = target_trans
        if self.cfg.lambda_size == 0.0:
            pred_size = target_size
        if self.cfg.lambda_rot == 0.0:
            pred_pose = target
        
        # reconstruct mesh
        for scene_idx, mask_legal in enumerate(batch["mask_legal"].float().cpu().numpy()):
            pred_mesh_list, gt_mesh_list = [], []
            for obj_idx in range(mask_legal.shape[0]):
                # only visualize legal objects
                if not mask_legal[obj_idx]:
                    continue
                else:  
                    # recover transform matrix from 6-D rotation and translation
                    transform_matrix_gt = self.recover_transform_matrix(target[scene_idx, obj_idx], target_trans[scene_idx, obj_idx])
                    transform_matrix_pred = self.recover_transform_matrix(pred_pose[scene_idx, obj_idx], pred_trans[scene_idx, obj_idx])
                    # size
                    bbox_size = target_size[scene_idx, obj_idx].detach().float().cpu().numpy() / 2
                    bbox_size_pred = pred_size[scene_idx, obj_idx].detach().float().cpu().numpy() / 2
                    # load mesh
                    mesh = mesh_v_f[scene_idx*self.cfg.max_objs + obj_idx]
                    mesh_pred = trimesh.Trimesh(vertices=mesh[0], faces=mesh[1])  # Create a Trimesh object
                    mesh_pred.apply_scale(bbox_size_pred) # apply size
                    mesh_pred.apply_transform(transform_matrix_pred.detach().float().cpu().numpy())  # Apply the transformation matrix
                    pred_mesh_list.append(mesh_pred)
                    # gt mesh
                    mesh_gt = trimesh.Trimesh(vertices=mesh[0], faces=mesh[1])
                    mesh_gt.apply_scale(bbox_size)
                    mesh_gt.apply_transform(transform_matrix_gt.detach().float().cpu().numpy())
                    gt_mesh_list.append(mesh_gt)
            
            # reconstruct a scene
            scene_gt = trimesh.Scene(gt_mesh_list)
            scene_pred = trimesh.Scene(pred_mesh_list)
            # save mesh
            scene_gt.export(self.get_save_path(f"it{self.true_global_step}_train/{batch['taskid'][scene_idx]}_gt.obj"))
            scene_pred.export(self.get_save_path(f"it{self.true_global_step}_train/{batch['taskid'][scene_idx]}_pred.obj"))
            # save image
            img = (batch["whole_img"][scene_idx]*255).float().cpu().numpy().astype(np.uint8)
            Image.fromarray(img).save(self.get_save_path(f"it{self.true_global_step}_train/{batch['taskid'][scene_idx]}.png"))

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            if name.startswith("lambda_"):
                self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}
    

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):    
        self.eval()
        
        # predict pose
        pred_pose_latents = self.sample(batch)
            
        # encode shape
        bs = batch["surface"].shape[0]
        if "sharp_surface" in batch.keys():
            sharp_surface = batch["sharp_surface"].view(-1, *batch["sharp_surface"].shape[2:])
            sharp_surface = sharp_surface[..., :3 + self.cfg.shape_model.point_feats]
        else:
            sharp_surface = None
        surface = batch["surface"].view(-1, *batch["surface"].shape[2:])
        shape_embeds, kl_embed, _ = self.shape_model.encode(
            surface[..., :3 + self.cfg.shape_model.point_feats], 
            sample_posterior=True,
            sharp_surface=sharp_surface, 
        )
        shape_latents = kl_embed * self.cfg.z_scale_factor
        shape_latents = shape_latents.view(bs, self.cfg.max_objs, *shape_latents.shape[-2:]) # [B, T, N, C]
    
            
        # val loss
        if "pose" in batch.keys():                       
            # supervise on pose values
            target, target_trans, target_size = batch["pose"], batch["translation"], batch["size"]
            pred_pose, pred_trans, pred_size = pred_pose_latents.split([6, 3, 1], dim=-1)  # [B, T, N, 6+3+1]
            pred_size = pred_size + 1  # normalize size to [0, 2]

            # Compute regular loss.
            loss_rot = ((pred_pose.float() - target.float()) ** 2)
            loss_trans = ((pred_trans.float() - target_trans.float()) ** 2)
            loss_size = ((pred_size.float() - target_size.float()) ** 2)
            
            if self.cfg.sup_latents:
                loss_rot = torch.mean(loss_rot, dim=-1)
                loss_trans = torch.mean(loss_trans, dim=-1)
                loss_size = torch.mean(loss_size, dim=-1)

            # mask
            mask_legal = batch["mask_legal"].unsqueeze(-1).float()
            num_legal = torch.sum(batch["mask_legal"].float(), dim=-1)
            loss_rot = torch.sum((loss_rot * mask_legal).mean(-1), dim=-1) / (num_legal+1e-4)
            loss_trans = torch.sum((loss_trans * mask_legal).mean(-1), dim=-1) / (num_legal+1e-4)
            loss_size = torch.sum((loss_size * mask_legal).mean(-1), dim=-1) / (num_legal+1e-4)
            loss = loss_rot + loss_trans + loss_size
            loss = loss.mean()
        
        else:
            loss = 0.0
        
        # log
        self.log("val/loss_rot", loss_rot.mean())
        self.log("val/loss_trans", loss_trans.mean())
        self.log("val/loss_size", loss_size.mean())
        self.log("val/loss", loss)
        print(f"val/loss: {loss.item()}")

        # viz
        # decode shape
        shape_latents = shape_latents.reshape(bs*self.cfg.max_objs, -1, shape_latents.shape[-1])
        shape = self.shape_model.decode(shape_latents)
        # logits = self.shape_model.extract_logits(shape)
        mesh_v_f, has_surface = self.shape_model.extract_geometry(latents=shape, extract_mesh_func=self.cfg.extract_mesh_func)
        
        # decode pose
        if self.cfg.sup_latents:
            target, target_trans, target_size = batch["pose"], batch["translation"], batch["size"]
            pred_pose, pred_trans, pred_size = self.pose_model.decode(pred_pose_latents)
            
        # replace pred values with target values if lambda is 0
        if self.cfg.lambda_trans == 0.0:
            pred_trans = batch["translation"]
        if self.cfg.lambda_size == 0.0:
            pred_size = batch["size"]
        if self.cfg.lambda_rot == 0.0:
            pred_pose = target

        # reconstruct mesh
        for scene_idx, mask_legal in enumerate(batch["mask_legal"].float().cpu().numpy()):
            # skip if no legal objects
            if mask_legal.sum() == 0:
                continue
            pred_mesh_list, gt_mesh_list = [], []
            for obj_idx in range(mask_legal.shape[0]):
                # only visualize legal objects
                if not mask_legal[obj_idx]:
                    continue
                else:  
                    # recover transform matrix from 6-D rotation and translation
                    transform_matrix_gt = self.recover_transform_matrix(target[scene_idx, obj_idx], batch["translation"][scene_idx, obj_idx])
                    transform_matrix_pred = self.recover_transform_matrix(pred_pose[scene_idx, obj_idx], pred_trans[scene_idx, obj_idx])
                    # size
                    bbox_size = batch["size"][scene_idx, obj_idx].float().cpu().numpy() / 2
                    bbox_size_pred = pred_size[scene_idx, obj_idx].float().cpu().numpy() / 2
                    # load mesh
                    mesh = mesh_v_f[scene_idx*self.cfg.max_objs + obj_idx]
                    mesh_pred = trimesh.Trimesh(vertices=mesh[0], faces=mesh[1])  # Create a Trimesh object
                    mesh_pred.apply_scale(bbox_size_pred) # apply size
                    mesh_pred.apply_transform(transform_matrix_pred.float().cpu().numpy())  # Apply the transformation matrix
                    pred_mesh_list.append(mesh_pred)
                    # gt mesh
                    mesh_gt = trimesh.Trimesh(vertices=mesh[0], faces=mesh[1])
                    mesh_gt.apply_scale(bbox_size)
                    mesh_gt.apply_transform(transform_matrix_gt.float().cpu().numpy())
                    gt_mesh_list.append(mesh_gt)
            
            # reconstruct a scene
            scene_gt = trimesh.Scene(gt_mesh_list)
            scene_pred = trimesh.Scene(pred_mesh_list)
            # save mesh
            scene_gt.export(self.get_save_path(f"it{self.true_global_step}/{batch['taskid'][scene_idx]}_gt.obj"))
            scene_pred.export(self.get_save_path(f"it{self.true_global_step}/{batch['taskid'][scene_idx]}_pred.obj"))
            # save image
            img = (batch["whole_img"][scene_idx]*255).float().cpu().numpy().astype(np.uint8)
            Image.fromarray(img).save(self.get_save_path(f"it{self.true_global_step}/{batch['taskid'][scene_idx]}.png"))
            # calculate metrics
            metrics = self.compute_metrics(vertices_pred=pred_mesh_list, vertices_gt=gt_mesh_list)
            self.metrics_list.append(metrics)

        return {"val/loss": loss}
    
    
    @torch.no_grad()
    def sample(self, batch, guidance_scale=None):
        self.eval()
        # encode shape
        bs = batch["surface"].shape[0]
        if "sharp_surface" in batch.keys():
            sharp_surface = batch["sharp_surface"].view(-1, *batch["sharp_surface"].shape[2:])
            sharp_surface = sharp_surface[..., :3 + self.cfg.shape_model.point_feats]
        else:
            sharp_surface = None
        surface = batch["surface"].view(-1, *batch["surface"].shape[2:])
        shape_embeds, kl_embed, _ = self.shape_model.encode(
            surface[..., :3 + self.cfg.shape_model.point_feats], 
            sample_posterior=True,
            sharp_surface=sharp_surface, 
        )
        shape_latents = kl_embed * self.cfg.z_scale_factor
        shape_latents = shape_latents.view(bs, self.cfg.max_objs, *shape_latents.shape[-2:]) # [B, T, N, C]

        # 2. gain visual condition
        if "image" in batch and batch['image'].dim() == 5:
            bs, n_images = batch['image'].shape[:2]
            batch['image'] = batch['image'].view(bs*n_images, *batch['image'].shape[-3:])
            cond_latents = self.condition.forward_visual_embeds(batch).to(shape_latents)
        else:
            cond_latents = self.condition.forward_visual_embeds(batch).to(shape_latents)
        cond_latents = cond_latents.reshape(bs, n_images, *cond_latents.shape[-2:]) # [B, T, N, C]
        uncond_latents = torch.zeros_like(cond_latents)
        
        # scene image latents
        if self.cfg.use_scene_img:
            scene_img_latents = self.condition.encode_image(batch["whole_img"]).to(shape_latents).reshape(bs, 1, *cond_latents.shape[-2:]).repeat(1,n_images,1,1)
            if self.cfg.use_scene_mask:
                scene_mask = batch["masks"].reshape(bs*n_images,*batch["masks"].shape[-2:],1).repeat(1,1,1,3)
                scene_mask_latents = self.condition.encode_image(scene_mask).to(shape_latents).reshape(bs, n_images, *cond_latents.shape[-2:])
                scene_img_latents = torch.cat([scene_img_latents, scene_mask_latents], dim=-2) # concat
            elif self.cfg.use_scene_mask_img:
                # apply mask on scene image
                scene_mask = batch["masks"].reshape(bs*n_images,*batch["masks"].shape[-2:],1).repeat(1,1,1,3)
                scene_obj_img = batch["whole_img"].repeat_interleave(n_images, dim=0) * scene_mask
                scene_mask_latents = self.condition.encode_image(scene_obj_img).to(shape_latents).reshape(bs, n_images, *cond_latents.shape[-2:])
                scene_img_latents = torch.cat([scene_img_latents, scene_mask_latents], dim=-2) # concat
            elif self.cfg.use_deocc_image:
                # deocc image
                deocc_image = batch["deocc_images"].reshape(bs*n_images,*batch["deocc_images"].shape[-3:])
                scene_mask_latents = self.condition.encode_image(deocc_image).to(shape_latents).reshape(bs, n_images, *cond_latents.shape[-2:])
                scene_img_latents = torch.cat([scene_img_latents, scene_mask_latents], dim=-2)
            uncond_scene_img_latents = torch.zeros_like(scene_img_latents)
        
        # scene pcd latents
        if self.cfg.use_scene_pcd:
            scene_pcds = batch["scene_pcds"].reshape(bs*self.cfg.max_objs,-1,3).to(shape_latents)
            if self.cfg.pretrain_pcd == "shape2vec":
                scene_pcd_kl_embed, _ = self.pcd_model.encode(scene_pcds, sample_posterior=True)
                scene_pcd_latents = self.pcd_emb(scene_pcd_kl_embed)
            else:
                scene_pcd_latents = self.pcd_model.encode(scene_pcds)
            scene_pcd_latents = scene_pcd_latents.view(bs, -1, *scene_pcd_latents.shape[-2:]) # [B, T, N, C]
            uncond_scene_pcd_latents = torch.zeros_like(scene_pcd_latents)
            
        ## 2.1 text condition if provided
        if "caption" in batch and self.cfg.use_caption:
            assert "caption" in batch.keys(), "caption is required for caption encoder"
            assert bs == len(batch["caption"]), "Batch size must be the same as the caption length."
            caption_list = [item for sublist in batch["caption"] for item in sublist]
            caption_latents = self.caption_condition.encode_text(caption_list).to(shape_latents)
            caption_latents = caption_latents.reshape(bs, -1, *caption_latents.shape[-2:])
            caption_latents = self.caption_emb(caption_latents)
            uncond_caption_latents = torch.zeros_like(caption_latents)
        
        # pcd condition
        if self.cfg.pretrain_pcd == "shape2vec":
            pcd = batch["obj_pcds"].reshape(bs*self.cfg.max_objs,-1,3).to(shape_latents)
            pcd_kl_embed, _ = self.pcd_model.encode(pcd, sample_posterior=True)
            pcd_latents = self.pcd_emb(pcd_kl_embed)
        else:
            pcd_latents = self.pcd_model.encode(batch["obj_pcds"].reshape(bs*self.cfg.max_objs,-1,3).to(shape_latents))
        pcd_latents = pcd_latents.view(bs, -1, *pcd_latents.shape[-2:]) # [B, T, N, C]
        uncond_pcd_latents = torch.zeros_like(pcd_latents)

        # pcd size and translation
        pcd_pose_latents = self.pose_model.encode_size_translation(size=batch["pcd_sizes"]-1, translation=batch["pcd_trans"]).to(shape_latents)
        pcd_pose_latents = self.pose_emb(pcd_pose_latents)
        uncond_pcd_pose_latents = torch.zeros_like(pcd_pose_latents)
     
        # conditional latents
        cond_latents = torch.cat([cond_latents, pcd_latents, pcd_pose_latents], dim=-2)
        uncond_latents = torch.cat([uncond_latents, uncond_pcd_latents, uncond_pcd_pose_latents], dim=-2)
        if self.cfg.use_scene_img:
            cond_latents = torch.cat([cond_latents, scene_img_latents], dim=-2)
            uncond_latents = torch.cat([uncond_latents, uncond_scene_img_latents], dim=-2)
        if self.cfg.use_scene_pcd:
            cond_latents = torch.cat([cond_latents, scene_pcd_latents], dim=-2)
            uncond_latents = torch.cat([uncond_latents, uncond_scene_pcd_latents], dim=-2)
        if self.cfg.use_caption:
            cond_latents = torch.cat([cond_latents, caption_latents], dim=-2)
            uncond_latents = torch.cat([uncond_latents, uncond_caption_latents], dim=-2)
        
        # cfg
        guidance_scale = guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        do_classifier_free_guidance = guidance_scale != 1.0
        if do_classifier_free_guidance:
            cond_latents = torch.cat([uncond_latents, cond_latents], dim=0)
        
        # attn mask
        attn_mask_legal = batch["mask_legal"].repeat(2,1) if do_classifier_free_guidance else batch["mask_legal"]
        attention_kwargs = {"mask_legal": attn_mask_legal} if self.denoiser_model.cfg.attn_mode == "scene" else {"mask_legal": torch.ones_like(attn_mask_legal)}
    
        # loop
        output = []
        sample_loop = flow_sample_pose_direct(
            self.denoise_scheduler,
            self.denoiser_model.eval(),
            pose_ae=self.pose_model.eval(),
            shape=(self.cfg.max_objs, 6+3+1),
            cond=cond_latents,
            cond_cat=shape_latents,
            steps=self.cfg.num_inference_steps,
            guidance_scale=guidance_scale,
            do_classifier_free_guidance=do_classifier_free_guidance,
            device=self.device,
            eta=0.0,
            disable_prog=False,
            generator=None,
            attention_kwargs=attention_kwargs,
        )
        for sample, t in sample_loop:
            output.append(sample)

        return output[-1]

    
    def recover_transform_matrix(self, rotation_6d, translation):
        """
        Recover 4x4 transformation matrix from 6D rotation representation and 3D translation.
        
        Args:
            rotation_6d: Tensor of shape (..., 6), representing 6D rotation.
            translation: Tensor of shape (..., 3), representing 3D translation.
        
        Returns:
            transformation: Tensor of shape (..., 4, 4)
        """
        R = repr6d2mat(rotation_6d)
        T = torch.eye(4, device=rotation_6d.device).expand(*rotation_6d.shape[:-1], 4, 4).clone()
        T[..., :3, :3] = R
        T[..., :3, 3] = translation
        
        return T
        
    def normalize_vector(self, v):
        return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)

    def on_validation_epoch_end(self):
        if len(self.metrics_list) == 0:
            return
        all_metrics = {}
        for k in self.metrics_list[0].keys():
            all_metrics[k] = np.mean([float(m[k]) for m in self.metrics_list if m is not None])
            self.log(f"val/{k}_mean", all_metrics[k])
            print(f"val/{k}_mean: {all_metrics[k]:.6f}")
        self.metrics_list = []
    
    def compute_metrics(self, vertices_pred, vertices_gt, use_icp_obj=False, use_icp_scene=False):
        from einops import rearrange
        from craftsman.utils.metrics import (
            compute_chamfer_distance,
            compute_fscore,
            compute_volume_iou,
            icp,
        )

        def normalize_(tensor):
            min_vals = tensor.min(dim=1, keepdim=True)[0]
            max_vals = tensor.max(dim=1, keepdim=True)[0]
            ranges = max_vals - min_vals
            ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)
            # respectively nrom
            normalized_tensor = 1.9 * (tensor - min_vals) / ranges.max() - 0.95
            return normalized_tensor

        cd_scene, cd_scene_1, cd_scene_2, fscore_scene = [], [], [], []
        cd_object, fscore_object, iou_bbox = [], [], []

        # preprocess: fps sampling
        vertices_pred = [torch.from_numpy(trimesh.sample.sample_surface(mesh, 20480)[0]) for mesh in vertices_pred]
        vertices_pred = torch.stack(vertices_pred).float().to(self.device)
        vertices_gt = [torch.from_numpy(trimesh.sample.sample_surface(mesh, 20480)[0]) for mesh in vertices_gt]
        vertices_gt = torch.stack(vertices_gt).float().to(self.device)

        # 1. scene
        B, N, C = vertices_pred.shape
        vertices_scene_pred = rearrange(vertices_pred, "B N C -> (B N) C").unsqueeze(0)
        vertices_scene_gt = rearrange(vertices_gt, "B N C -> (B N) C").unsqueeze(0)
        
        # normalize
        vertices_scene_pred = normalize_(vertices_scene_pred)
        vertices_scene_gt = normalize_(vertices_scene_gt)
        
        # object
        vertices_object_pred = vertices_scene_pred.reshape(B,N,C).clone().detach()
        vertices_object_gt = vertices_scene_gt.reshape(B,N,C).clone().detach()

        ## 1.1 icp
        if use_icp_scene:
            vertices_scene_pred, R, t = icp(
                vertices_scene_pred, vertices_scene_gt, max_iterations=50
            )
        
        ## 1.2 metrics
        cds = compute_chamfer_distance(vertices_scene_pred, vertices_scene_gt)
        cd_scene.append(cds[0])
        cd_scene_1.append(cds[1])
        cd_scene_2.append(cds[2])
        fscore_scene.append(compute_fscore(vertices_scene_pred, vertices_scene_gt))

        # # 2. object
        # vertices_object_pred = vertices_pred.reshape(B,N,C)
        # vertices_object_gt = vertices_gt.reshape(B,N,C)

        # 2.1. object iou in global scene
        iou_bbox.append(compute_volume_iou(vertices_object_pred, vertices_object_gt, mode="bbox"))

        # 2.2 object quality
        vertices_object_pred = normalize_(vertices_object_pred)
        vertices_object_gt = normalize_(vertices_object_gt)

        ## 2.2.1 icp
        if use_icp_obj:
            vertices_object_pred, _, _ = icp(
                vertices_object_pred, vertices_object_gt, max_iterations=50
            )

        ## 2.2.2 metrics
        cd_object.append(
            compute_chamfer_distance(vertices_object_pred, vertices_object_gt)[0]
        )
        fscore_object.append(
            compute_fscore(vertices_object_pred, vertices_object_gt)
        )

        for item in [
            cd_scene,
            cd_scene_1,
            cd_scene_2,
            fscore_scene,
            cd_object,
            fscore_object,
            iou_bbox,
        ]:
            if len(item) == 0:
                return None

        mean_acc = lambda x: torch.cat(x).mean().cpu()

        return {
            "scene_cd": mean_acc(cd_scene),
            "scene_cd_1": mean_acc(cd_scene_1),
            "scene_cd_2": mean_acc(cd_scene_2),
            "scene_fscore": mean_acc(fscore_scene),
            "object_cd": mean_acc(cd_object),
            "object_fscore": mean_acc(fscore_object),
            "iou_bbox": mean_acc(iou_bbox),
        }

