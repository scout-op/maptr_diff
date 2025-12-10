import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmcv.ops import Voxelization, DynamicScatter

# In MMDetection3D 1.x, builder module is removed. Use MODELS.build() instead.
@MODELS.register_module()
class MapTR(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 modality='vision',
                 lidar_encoder=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 ):

        super(MapTR, self).__init__(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            img_neck=img_neck,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.modality = modality
        if self.modality == 'fusion' and lidar_encoder is not None :
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": MODELS.build(lidar_encoder["backbone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          lidar_feat,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, lidar_feat, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Compatibility forward wrapper.

        This method adapts the original MapTR forward interface to the
        MMEngine/MMDetection3D 2.x calling convention.

        Two calling patterns are supported:

        1. Legacy pattern (MMDet3D 1.x):

           ``model.forward(return_loss=True, img=..., img_metas=..., ...)``

        2. New MMEngine pattern (MMDet3D 2.x via Base3DDetector):

           ``model(inputs=..., data_samples=..., mode='loss')``

        For the new pattern we:

        - Extract ``imgs`` and metadata from ``inputs`` and ``data_samples``.
        - Build the legacy ``img`` tensor with shape (B, T, N, C, H, W), where
          T is the queue length and N is the number of cameras.
        - Build ``img_metas`` as a list[list[dict]] of per-frame metas,
          matching the expectations of ``forward_train`` / ``forward_test``.
        """

        mode = kwargs.pop('mode', None)
        inputs = kwargs.pop('inputs', None)
        data_samples = kwargs.pop('data_samples', None)
        
        # Pop legacy keys - save them as they may contain the correctly 
        # structured temporal data from union2one
        legacy_img = kwargs.pop('img', None)
        legacy_img_metas = kwargs.pop('img_metas', None)

        # Helper to unwrap DC objects recursively
        def unwrap_dc(obj):
            if hasattr(obj, 'data'):
                return unwrap_dc(obj.data)
            elif hasattr(obj, '_data'):
                return unwrap_dc(obj._data)
            elif isinstance(obj, (list, tuple)):
                return [unwrap_dc(item) for item in obj]
            return obj

        # New-style entry: inputs + data_samples
        if inputs is not None and data_samples is not None:
            # Training / loss path
            if mode == 'loss' or return_loss:
                # Det3DDataPreprocessor packs images into inputs['imgs']
                if 'imgs' in inputs:
                    imgs = inputs['imgs']  # Tensor or list of tensors
                elif 'img' in inputs:
                    imgs = inputs['img']
                else:
                    raise KeyError("MapTR expects 'imgs' or 'img' in inputs for vision modality")

                # Handle case where preprocessor returns list of tensors
                # (happens with pseudo_collate or when stacking fails)
                if isinstance(imgs, (list, tuple)):
                    # Stack list of tensors into batch tensor
                    # Each element is typically [N, C, H, W] for multi-view
                    imgs = torch.stack(imgs, dim=0)

                # MMDet3D 2.x Det3DDataPreprocessor already stacks to tensor.
                # Our legacy MapTR.forward_train expects img shaped as
                # [B, T, N, C, H, W]. For queue_length=1 (your config), T=1
                # so we can add a singleton queue dimension.
                if imgs.dim() == 5:
                    # imgs: [B, N, C, H, W] -> [B, 1, N, C, H, W]
                    img = imgs.unsqueeze(1)
                elif imgs.dim() == 6:
                    # Already [B, T, N, C, H, W]
                    img = imgs
                else:
                    raise ValueError(f"Unexpected imgs dim {imgs.dim()} for MapTR")
                
                # Ensure img is on the same device as the model
                device = next(self.parameters()).device
                if img.device != device:
                    img = img.to(device)

                # Use legacy img_metas if available (from union2one with temporal info)
                # Otherwise build from data_samples
                if legacy_img_metas is not None:
                    # Extract from DC wrapper recursively
                    img_metas = unwrap_dc(legacy_img_metas)
                    # img_metas from union2one is a dict {0: meta0, 1: meta1, ...}
                    # wrapped in a list for batch. Convert to list[dict] format.
                    if isinstance(img_metas, dict):
                        # Single sample case - wrap in list for batch dim
                        img_metas = [img_metas]
                else:
                    # Fallback: build from data_samples (may lack temporal info)
                    img_metas = []
                    for ds in data_samples:
                        if hasattr(ds, 'metainfo'):
                            meta = dict(ds.metainfo)
                            # Ensure prev_bev_exists is set
                            meta.setdefault('prev_bev_exists', False)
                        else:
                            meta = {'prev_bev_exists': False}
                        # Wrap in dict with temporal index for compatibility
                        img_metas.append({0: meta})

                return self.forward_train(img=img, img_metas=img_metas, **kwargs)

            # Prediction / inference path
            else:
                # For predict/tensor we follow a similar mapping but call
                # forward_test/simple_test. Here we assume no temporal queue
                # (queue_length=1) and single frame per sample.
                if 'imgs' in inputs:
                    imgs = inputs['imgs']
                elif 'img' in inputs:
                    imgs = inputs['img']
                else:
                    raise KeyError("MapTR expects 'imgs' or 'img' in inputs for vision modality")

                # Handle case where preprocessor returns list of tensors
                if isinstance(imgs, (list, tuple)):
                    imgs = torch.stack(imgs, dim=0)

                # Ensure imgs is on the same device as the model
                device = next(self.parameters()).device
                if imgs.device != device:
                    imgs = imgs.to(device)
                
                if imgs.dim() == 5:
                    # [B, N, C, H, W]
                    img = [imgs]
                elif imgs.dim() == 6:
                    # [B, T, N, C, H, W] -> use last frame
                    img = [imgs[:, -1, ...]]
                else:
                    raise ValueError(f"Unexpected imgs dim {imgs.dim()} for MapTR inference")

                # Use legacy img_metas if available
                if legacy_img_metas is not None:
                    img_metas = unwrap_dc(legacy_img_metas)
                    if isinstance(img_metas, dict):
                        img_metas = [img_metas]
                    # Convert {0: meta, ...} to [[meta]] for forward_test
                    img_metas = [[m.get(0, m) if isinstance(m, dict) else m] for m in img_metas]
                else:
                    img_metas = []
                    for ds in data_samples:
                        if hasattr(ds, 'metainfo'):
                            meta = dict(ds.metainfo)
                            meta.setdefault('scene_token', None)
                            meta.setdefault('can_bus', [0]*18)
                        else:
                            meta = {'scene_token': None, 'can_bus': [0]*18}
                        img_metas.append([meta])

                return self.forward_test(img_metas=img_metas, img=img, **kwargs)

        # Legacy entry: direct kwargs (MMDet3D 1.x style)
        # Re-add the legacy keys we popped
        if legacy_img is not None:
            kwargs['img'] = legacy_img
        if legacy_img_metas is not None:
            kwargs['img_metas'] = legacy_img_metas
            
        if mode is not None:
            if mode == 'loss':
                return_loss = True
            elif mode in ('predict', 'tensor'):
                return_loss = False

        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, None, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    def extract_lidar_feat(self,points):
        feats, coords, sizes = self.voxelize(points)
        # voxel_features = self.lidar_modal_extractor["voxel_encoder"](feats, sizes, coords)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)
        
        return lidar_feat

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        lidar_feat = None
        if self.modality == 'fusion':
            lidar_feat = self.extract_lidar_feat(points)
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue>1 else None
        
        img_metas = [each[len_queue-1] for each in img_metas]
        
        # Ensure required keys exist in img_metas
        for meta in img_metas:
            if 'can_bus' not in meta:
                meta['can_bus'] = np.zeros(18)
            elif not isinstance(meta['can_bus'], np.ndarray):
                meta['can_bus'] = np.array(meta['can_bus'])
            meta.setdefault('prev_bev_exists', False)
            meta.setdefault('scene_token', 'unknown')
            # lidar2img: 6 cameras x 4x4 transformation matrices (identity as default)
            if 'lidar2img' not in meta or len(meta['lidar2img']) == 0:
                meta['lidar2img'] = [np.eye(4).astype(np.float32) for _ in range(6)]
            meta.setdefault('img_shape', [(450, 800, 3)] * 6)
            meta.setdefault('pad_shape', [(480, 800, 3)] * 6)
        
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, lidar_feat, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None,points=None,  **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        
        # Ensure required keys exist in img_metas
        for batch_metas in img_metas:
            for meta in batch_metas:
                if isinstance(meta, dict):
                    if 'can_bus' not in meta:
                        meta['can_bus'] = np.zeros(18)
                    elif not isinstance(meta['can_bus'], np.ndarray):
                        meta['can_bus'] = np.array(meta['can_bus'])
                    meta.setdefault('scene_token', 'unknown')
                    # lidar2img: 6 cameras x 4x4 transformation matrices
                    if 'lidar2img' not in meta or len(meta['lidar2img']) == 0:
                        meta['lidar2img'] = [np.eye(4).astype(np.float32) for _ in range(6)]
                    meta.setdefault('img_shape', [(450, 800, 3)] * 6)
                    meta.setdefault('pad_shape', [(480, 800, 3)] * 6)
        
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], points[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()

        return result_dict
    def simple_test_pts(self, x, lidar_feat, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        
        bbox_results = [
            self.pred2result(bboxes, scores, labels, pts)
            for bboxes, scores, labels, pts in bbox_list
        ]
        # import pdb;pdb.set_trace()
        return outs['bev_embed'], bbox_results
    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        lidar_feat = None
        if self.modality =='fusion':
            lidar_feat = self.extract_lidar_feat(points)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, lidar_feat, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list


@MODELS.register_module()
class MapTR_fp16(MapTR):
    """
    The default version BEVFormer currently can not support FP16. 
    We provide this version to resolve this issue.
    """
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      prev_bev=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # import pdb;pdb.set_trace()
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev=prev_bev)
        losses.update(losses_pts)
        return losses


    def val_step(self, data, optimizer):
        """
        In BEVFormer_fp16, we use this `val_step` function to inference the `prev_pev`.
        This is not the standard function of `val_step`.
        """

        img = data['img']
        img_metas = data['img_metas']
        img_feats = self.extract_feat(img=img,  img_metas=img_metas)
        prev_bev = data.get('prev_bev', None)
        prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        return prev_bev
