#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:10:25 2026

@author: alex
"""


import sys
from collections import defaultdict

import torch


def sizeof_fmt(num):
    """Human readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:3.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PB"

short_mode = False

def cprint(astring):
    global short_mode
    if not short_mode:
        print(astring)

def describemodel(path):

    cprint(f"Loading: {path}")

    obj = torch.load(path, map_location="cpu")

    # handle common checkpoint formats
    if isinstance(obj, dict):
        if "state_dict" in obj:
            state_dict = obj["state_dict"]
            cprint("Found checkpoint with key: state_dict")
        elif "model" in obj and isinstance(obj["model"], dict):
            state_dict = obj["model"]
            cprint("Found checkpoint with key: model")
        else:
            state_dict = obj
            cprint("Assuming object itself is a state_dict")
    else:
        print(f"Unsupported object type: {type(obj)}")
        sys.exit(1)

    total_params = 0
    total_bytes = 0

    dtype_count = defaultdict(int)

    cprint("\n=== TENSORS ===")

    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue

        numel = tensor.numel()
        bytes_ = numel * tensor.element_size()

        total_params += numel
        total_bytes += bytes_

        dtype_count[str(tensor.dtype)] += numel

        cprint(
            f"{name:60s} "
            f"shape={str(tuple(tensor.shape)):20s} "
            f"dtype={str(tensor.dtype):15s} "
            f"params={numel:12,d} "
            f"size={sizeof_fmt(bytes_)}"
        )

    print(f"\n=== SUMMARY === [{path}]")
    print(f"Total parameters : {total_params:,}")
    print(f"Total size       : {sizeof_fmt(total_bytes)}")

    cprint("\n=== PARAMS BY DTYPE ===")
    for dtype, count in dtype_count.items():
        print(f"{dtype:15s}: {count:,}")

    cprint("\n=== TOP-LEVEL KEYS ===")
    if isinstance(obj, dict):
        for k in obj.keys():
            cprint(f"- {k}")

    # optional metadata
    for key in ["epoch", "step", "global_step"]:
        if key in obj:
            cprint(f"\n{key}: {obj[key]}")


def main():
    global short_mode

  
    modelname=""
    
    if len(sys.argv)>2:
        if sys.argv[1] == '--short':
            short_mode=True    
            for modelname in sys.argv[2:]:
                describemodel(modelname)
 

    elif len(sys.argv) < 2:
        pass
    else:
        describemodel(modelname)
        
    if modelname == "":
        print(f"Usage: {sys.argv[0]} state.pth")
        sys.exit(1)
    



if __name__ == "__main__":
    main()