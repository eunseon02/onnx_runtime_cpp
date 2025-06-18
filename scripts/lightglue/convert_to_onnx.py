#!/usr/bin/env python
import os

import torch

import onnxruntime
from lightglue_wrapper import LightGlueWrapper as LightGlue

import torch

def main():
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


    model = LightGlue(**default_conf)
    model.eval()

    batch_size = 1
    height = 480
    width = 640
    num_keypoints = 382
    data = {}
    for i in range(2):
        data[f"image{i}"] = torch.randn(batch_size, 3, height, width)
        data[f"keypoints{i}"] = torch.randn(batch_size, num_keypoints, 2)
        # data[f"keypoint_scores{i}"] = torch.randn(batch_size, num_keypoints)
        data[f"descriptors{i}"] = torch.randn(batch_size, num_keypoints, 256)

        data[f"image_size{i}"] = torch.tensor(
            [[height, width]] * batch_size,
        )

    # scripted = torch.jit.script(model)  
    example_args = (
        torch.randn(1,1,480,640),  # image0
        torch.tensor([[480,640]]), # image_size0
        torch.randn(1,382,2),      # kpts0
        torch.randn(1,382,256),    # desc0
        torch.randn(1,1,480,640),  # image1
        torch.tensor([[480,640]]), # image_size1
        torch.randn(1,382,2),      # kpts1
        torch.randn(1,382,256),    # desc1
    )

        

    torch.onnx.export(
        model,
        example_args,
        "light_glue.onnx",
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=list(data.keys()),
        output_names=["matches0", "matches1", "matching_scores0", "matching_scores1", "prune0", "prune1"],
        dynamic_axes={
            "keypoints0": {0: "batch_size", 1: "num_keypoints0"},
            # "keypoint_scores0": {0: "batch_size", 1: "num_keypoints0"},
            "descriptors0": {0: "batch_size", 1: "num_keypoints0"},
            "keypoints1": {0: "batch_size", 1: "num_keypoints1"},
            # "keypoint_scores1": {0: "batch_size", 1: "num_keypoints1"},
            "descriptors1": {0: "batch_size", 1: "num_keypoints1"},
            "matches0": {0: "batch_size", 1: "num_keypoints0"},
            "matches1": {0: "batch_size", 1: "num_keypoints1"},
            "matching_scores0": {0: "batch_size", 1: "num_keypoints0"},
            "matching_scores1": {0: "batch_size", 1: "num_keypoints1"},
            "prune0": {0: "batch_size", 1: "num_keypoints0"},
            "prune1": {0: "batch_size", 1: "num_keypoints1"},
        },
    )
    print(f"\nonnx model is saved to: {os.getcwd()}/light_glue.onnx")

    print("\ntest inference using onnxruntime")
    sess = onnxruntime.InferenceSession("light_glue.onnx")
    for input in sess.get_inputs():
        print("input: ", input)

    print("\n")
    for output in sess.get_outputs():
        print("output: ", output)


if __name__ == "__main__":
    main()
