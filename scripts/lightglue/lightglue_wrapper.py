import torch

from LightGlue.lightglue.lightglue_onnx import (
    LightGlue,
    pad_to_length,
    filter_matches,
    normalize_keypoints,
)

from torch.onnx.operators import shape_as_tensor

class LightGlueWrapper(LightGlue):
    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.95,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }


    def __init__(self, features: str = "superpoint", **conf):
        super().__init__(features=features, **conf)

    def forward(
        self,
        image_size0: torch.Tensor,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        image_size1: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
    ):
        data = {
            # "image0": image0,
            "image_size0": image_size0.to(torch.int64),
            "keypoints0": keypoints0,
            "descriptors0": descriptors0,
            # "image1": image1,
            "image_size1": image_size1.to(torch.int64),
            "keypoints1": keypoints1,
            "descriptors1": descriptors1,
        }
        """Run LightGlue on a pair of keypoints and descriptors"""

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        # b, m, _ = kpts0.shape
        # b, n, _ = kpts1.shape
        b = kpts0.size(0)
        m = kpts0.size(1)
        n = kpts1.size(1)
        nl = self.conf.n_layers 

        device = kpts0.device
        size0, size1 = data.get("image_size0"), data.get("image_size1")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        # if self.conf.add_scale_ori:
        #     kpts0 = torch.cat(
        #         [kpts0] + [data0[k].unsqueeze(-1) for k in ("scales", "oris")], -1
        #     )
        #     kpts1 = torch.cat(
        #         [kpts1] + [data1[k].unsqueeze(-1) for k in ("scales", "oris")], -1
        #     )

        desc0 = data["descriptors0"].detach().contiguous()
        desc1 = data["descriptors1"].detach().contiguous()

        # assert desc0.shape[-1] == self.conf.input_dim
        # assert desc1.shape[-1] == self.conf.input_dim

        # if torch.is_autocast_enabled():
        #     desc0 = desc0.half()
        #     desc1 = desc1.half()

        mask0, mask1 = None, None
        # m = int(desc0.size(1))
        # n = int(desc1.size(1))
        c = m if m > n else n

        do_compile = self.static_lengths and c <= max(self.static_lengths)
        if do_compile:
            kn = min([k for k in self.static_lengths if k >= c])
            desc0, mask0 = pad_to_length(desc0, kn)
            desc1, mask1 = pad_to_length(desc1, kn)
            kpts0, _ = pad_to_length(kpts0, kn)
            kpts1, _ = pad_to_length(kpts1, kn)
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0
        do_point_pruning = self.conf.width_confidence > 0 and not do_compile
        pruning_th = self.pruning_min_kpts(device)
        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            # if desc0.shape[1] == 0 or desc1.shape[1] == 0:  # no keypoints
            #     break

            desc0, desc1 = self.transformers[i](
                desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1
            )
            if i == self.conf.n_layers - 1:
                continue  # no early stopping or adaptive width at last layer

            if do_early_stop:
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                    break
            # if do_point_pruning and desc0.shape[-2] > pruning_th:
            #     scores0 = self.log_assignment[i].get_matchability(desc0)
            #     prunemask0 = self.get_pruning_mask(token0, scores0, i)
            #     keep0 = torch.where(prunemask0)[1]
            #     ind0 = ind0.index_select(1, keep0)
            #     desc0 = desc0.index_select(1, keep0)
            #     encoding0 = encoding0.index_select(-2, keep0)
            #     prune0[:, ind0] += 1
            # if do_point_pruning and desc1.shape[-2] > pruning_th:
            #     scores1 = self.log_assignment[i].get_matchability(desc1)
            #     prunemask1 = self.get_pruning_mask(token1, scores1, i)
            #     keep1 = torch.where(prunemask1)[1]
            #     ind1 = ind1.index_select(1, keep1)
            #     desc1 = desc1.index_select(1, keep1)
            #     encoding1 = encoding1.index_select(-2, keep1)
            #     prune1[:, ind1] += 1
            scores0    = self.log_assignment[i].get_matchability(desc0)  # [b, m]
            prunemask0 = self.get_pruning_mask(token0, scores0, i)       # [b, m] bool

            scores1    = self.log_assignment[i].get_matchability(desc1)  # [b, n]
            prunemask1 = self.get_pruning_mask(token1, scores1, i)       # [b, n] bool

            # 3) prune count 업데이트 (살아남은 포인트만 +1)
            prune0 = prune0 + prunemask0.long()                          # [b, m]
            prune1 = prune1 + prunemask1.long()                          # [b, n]

            # 4) 실제 인덱스 추출 (NonZero → Gather)
            # torch.where(prunemask0) → (coords, keep0), keep0=[b, k0]
            keep0 = torch.where(prunemask0)  # tuple (batch_idx, kp_idx)
            batch_idx0, kp_idx0 = keep0
            # Gather를 쓰려면, idx tensor는 [b, k0] 형태여야 함
            # batch 축마다 다른 개수라면 pad/gather_all 형태가 필요하지만,
            # LightGlue는 batch independent이므로 아래와 같이 배치별 동작으로 가정:
            keep0_flat = kp_idx0  # shape = [total_kept], we’ll use gather with batch offsets if needed

            # 동일한 방식으로 desc1
            keep1 = torch.where(prunemask1)
            batch_idx1, kp_idx1 = keep1
            keep1_flat = kp_idx1

            # 5) Gather for desc / encoding / prune
            #   reshape desc0: [b, m, D] → [b*m, D] , then index_select, reshape back
            D0 = desc0.size(2)
            desc0_flat = desc0.view(-1, D0)                             # [b*m, D0]
            # linear index로 변환: idx = batch_idx * m + kp_idx
            lin_idx0 = batch_idx0 * m + kp_idx0                         # [total_kept]
            desc0 = desc0_flat.index_select(0, lin_idx0).view(b, -1, D0)  # [b, k0, D0]

            # encoding0: 예를 들어 [b, T, m, C] 라면, 비슷하게 처리
            T, C = encoding0.size(1), encoding0.size(-1)
            enc0_flat = encoding0.permute(0, 2, 1, 3).contiguous().view(-1, T, C)  
            # permute to [b, m, T, C] → flat [b*m, T, C]
            encoding0 = enc0_flat.index_select(0, lin_idx0).view(b, -1, T, C).permute(0, 2, 1, 3)
            # → back to [b, T, k0, C]

            # prune0: [b, m] → flat [b*m] → gather → [b, k0]
            prune0_flat = prune0.view(-1)
            prune0 = prune0_flat.index_select(0, lin_idx0).view(b, -1)

            # 6) 동일하게 desc1, encoding1, prune1 처리
            D1 = desc1.size(2)
            desc1_flat = desc1.view(-1, D1)
            lin_idx1   = batch_idx1 * n + kp_idx1
            desc1      = desc1_flat.index_select(0, lin_idx1).view(b, -1, D1)

            T1, C1     = encoding1.size(1), encoding1.size(-1)
            enc1_flat  = encoding1.permute(0, 2, 1, 3).contiguous().view(-1, T1, C1)
            encoding1  = enc1_flat.index_select(0, lin_idx1).view(b, -1, T1, C1).permute(0, 2, 1, 3)

            prune1_flat = prune1.view(-1)
            prune1      = prune1_flat.index_select(0, lin_idx1).view(b, -1)     


        # Case1 : If we have no keypoints, return empty matches.
        empty_m0 = desc0.new_full((b, m), -1, dtype=torch.long)
        empty_m1 = desc1.new_full((b, n), -1, dtype=torch.long)
        empty_ms0 = desc0.new_zeros((b, m))
        empty_ms1 = desc1.new_zeros((b, n))
        empty_matches = desc0.new_empty((b, 0, 2), dtype=torch.long)
        empty_mscores = desc0.new_empty((b, 0))
        if not do_point_pruning:
            empty_prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            empty_prune1 = torch.ones_like(mscores1) * self.conf.n_layers
        else:
            nl = self.conf.n_layers
            empty_prune0 = desc0.new_ones((b, m), dtype=torch.long) * nl
            empty_prune1 = desc1.new_ones((b, n), dtype=torch.long) * nl
  

        # Case2 : If we have keypoints, compute matches.
        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]  # remove padding
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            if do_point_pruning:
                m_indices_0 = ind0[k, m_indices_0]
                m_indices_1 = ind1[k, m_indices_1]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])


        
        # TODO: Remove when hloc switches to the compact format.
        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers
        
        # Select Case1 or Case2
        b = desc0.size(0)
        shape0 = shape_as_tensor(desc0)
        shape1 = shape_as_tensor(desc1)
        m_dim  = shape0[1]  
        n_dim  = shape1[1]

        empty0 = torch.eq(m_dim, 0)
        empty1 = torch.eq(n_dim, 0)
        empty_flag = torch.logical_or(empty0, empty1)
        is_empty = empty_flag.unsqueeze(0).expand(b).unsqueeze(1)


        m0 = torch.where(is_empty, empty_m0, m0)
        m1 = torch.where(is_empty, empty_m1, m1)
        mscores0  = torch.where(is_empty, empty_ms0, mscores0)
        mscores1  = torch.where(is_empty, empty_ms1, mscores1)
        prune0   = torch.where(is_empty, empty_prune0, prune0)
        prune1   = torch.where(is_empty, empty_prune1, prune1)

        # for k in data:
        #     print(f"{k}: {data[k].shape}")
        # print("prune0", prune0.shape, "prune1",prune1.shape)

        return {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "prune0": prune0,
            "prune1": prune1,
        }

