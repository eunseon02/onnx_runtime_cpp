#!/usr/bin/env python
import copy
import os
import sys
from typing import Any, Dict

import torch
from einops.einops import rearrange

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "aspanformer"))
from src.ASpanFormer import ASpanFormer
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config



config = get_cfg_defaults()
config.merge_from_file(    os.path.join(
        _CURRENT_DIR,
        "aspanformer", 
        "configs",
        "aspan",
        "outdoor",
        "aspan_test.py"
    ))
_config = lower_config(config)

DEFAULT_CFG = copy.deepcopy(_config['aspan'])
DEFAULT_CFG['coarse']['d_model'] = 256

# DEFAULT_CFG['coarse']['test_res'] = [779, 1054] 
# DEFAULT_CFG['coarse']['train_res'] = [779, 1054] 
# DEFAULT_CFG['coarse']['d_flow'] = 128
# DEFAULT_CFG['coarse']['ini_layer_num'] = 2
# DEFAULT_CFG['coarse']['radius_scale'] = 5
# DEFAULT_CFG['coarse']['nsample'] = [2, 8]


class ASpanFormerWrapper(ASpanFormer):
    def __init__(
        self,
        config: Dict[str, Any] = DEFAULT_CFG,
    ):
        ASpanFormer.__init__(self, config)

    def forward(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        online_resize: bool = False,
    ) -> Dict[str, torch.Tensor]:
        data = {
            "image0": image0,
            "image1": image1,
        }
        del image0, image1

        if online_resize:
            assert data['image0'].shape[0]==1 and data['image1'].shape[1]==1
            self.resize_input(data,self.config['coarse']['train_res'])
        else:
            data['pos_scale0'],data['pos_scale1']=None,None

        # 1. Local Feature CNN
        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )
        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(
                torch.cat([data["image0"], data["image1"]], dim=0)
            )
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data["bs"]), feats_f.split(data["bs"])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(
                data['image0']), self.backbone(data['image1'])

        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        [feat_c0, pos_encoding0], [feat_c1, pos_encoding1] = self.pos_encoding(feat_c0,data['pos_scale0']), self.pos_encoding(feat_c1,data['pos_scale1'])
        # print("feat_c0.shape:", feat_c0.shape)
        # print("pos_encoding0.shape:", pos_encoding0.shape)

        feat_c0 = rearrange(feat_c0, 'n c h w -> n c h w ')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n c h w ')

        #TODO:adjust ds 
        ds0=[int(data['hw0_c'][0]/self.coarsest_level[0]),int(data['hw0_c'][1]/self.coarsest_level[1])]
        ds1=[int(data['hw1_c'][0]/self.coarsest_level[0]),int(data['hw1_c'][1]/self.coarsest_level[1])]
        if online_resize:
            ds0,ds1=[4,4],[4,4]

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(
                -2), data['mask1'].flatten(-2)
        feat_c0, feat_c1, flow_list = self.loftr_coarse(
            feat_c0, feat_c1,pos_encoding0,pos_encoding1,mask_c0,mask_c1,ds0,ds1)

        # 3. match coarse-level and register predicted offset
        self.coarse_matching(feat_c0, feat_c1, flow_list,data,
                             mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(
            feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(
                feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        # 6. resize match coordinates back to input resolution
        if online_resize:
            data['mkpts0_f']*=data['online_resize_scale0']
            data['mkpts1_f']*=data['online_resize_scale1']


        rename_keys: Dict[str, str] = {
            "mkpts0_f": "keypoints0",
            "mkpts1_f": "keypoints1",
            "mconf": "confidence",
        }
        out: Dict[str, torch.Tensor] = {}
        for k, v in rename_keys.items():
            _d = data[k]
            if isinstance(_d, torch.Tensor):
                out[v] = _d
            else:
                raise TypeError(
                    f"Expected torch.Tensor for item `{k}`. Gotcha {type(_d)}"
                )
        del data

        return out
