#!/usr/bin/env python

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# This script extracts fp32 consolidated weights from a zero 1, 2 and 3 DeepSpeed checkpoints. It gets
# copied into the top level checkpoint dir, so the user can easily do the conversion at any point in
# the future. Once extracted, the weights don't require DeepSpeed and can be used in any
# application.
#
# example: python zero_to_fp32.py . pytorch_model.bin

import argparse
import torch

# load to cpu
device = torch.device('cpu')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_ref",
                        type=str,
                        help="path to the desired checkpoint folder, e.g., path/checkpoint-12")
    parser.add_argument("checkpoint_dir",
                        type=str,
                        help="path to the desired checkpoint folder, e.g., path/checkpoint-12")
    parser.add_argument(
        "output_dir",
        type=str,
        help="path to the pytorch fp32 state_dict output file (e.g. path/checkpoint-12/pytorch_model.bin)")
    parser.add_argument("-t",
                        "--tag",
                        type=str,
                        default=None,
                        help="checkpoint tag used as a unique identifier for checkpoint. e.g., global_step1")
    parser.add_argument("-d", "--debug", action='store_true', help="enable debug")
    args = parser.parse_args()

    debug = args.debug

    from transformers import LlamaTokenizer, LlamaForCausalLM

    tokenizer = LlamaTokenizer.from_pretrained(args.model_ref)
    tokenizer.save_pretrained(args.output_dir)

    model = LlamaForCausalLM.from_pretrained(args.model_ref)
    ref_keys = model.state_dict().keys()

    state_dict = torch.load(f'{args.checkpoint_dir}/pytorch_model.bin')
    new_keys = state_dict.keys()

    for key in [
        'v_head.summary.bias',
        'v_head.summary.weight'
    ]:
        del state_dict[key]

    model.load_state_dict(state_dict)
    model.half()
    model.save_pretrained(args.output_dir)

