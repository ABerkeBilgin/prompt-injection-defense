# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys


parser = argparse.ArgumentParser(prog='')
parser.add_argument('-m', '--model_name_or_path', type=str, nargs='+')
parser.add_argument('--lora_alpha', type=float, default=[8.0], nargs='+')
parser.add_argument('--norun', action='store_false')
args = parser.parse_args()
args.run = args.norun

cmds = [
    [sys.executable, '-m', 'src.official_stacks.meta_secalign.test', '--attack', 'none', 'straightforward', 'straightforward_before', 'ignore', 'ignore_before', 'completion', 'completion_ignore', 'completion_llama33_70B', 'completion_ignore_llama33_70B', '--defense', 'none', '--test_data', 'data/davinci_003_outputs.json', '--lora_alpha', '{lora_alpha}', '-m', '{model_name_or_path}'],
    [sys.executable, '-m', 'src.official_stacks.meta_secalign.test', '--attack', 'none', 'straightforward', 'straightforward_before', 'ignore', 'ignore_before', 'completion', 'completion_ignore', 'completion_llama33_70B', 'completion_ignore_llama33_70B', '--defense', 'none', '--test_data', 'data/SEP_dataset_test.json', '--lora_alpha', '{lora_alpha}', '-m', '{model_name_or_path}'],
    [sys.executable, '-m', 'src.official_stacks.meta_secalign.test_lm_eval', '--lora_alpha', '{lora_alpha}', '-m', '{model_name_or_path}'],
    [sys.executable, '-m', 'src.official_stacks.meta_secalign.test_agentdojo', '-a', 'none', '-d', 'repeat_user_prompt', '-m', '{model_name_or_path}', '--lora_alpha', '{lora_alpha}'],
    [sys.executable, '-m', 'src.official_stacks.meta_secalign.test_agentdojo', '-a', 'important_instructions', '-d', 'repeat_user_prompt', '-m', '{model_name_or_path}', '--lora_alpha', '{lora_alpha}'],
    [sys.executable, '-m', 'src.official_stacks.meta_secalign.test_injecagent', '--defense', 'sandwich', '--lora_alpha', '{lora_alpha}', '-m', '{model_name_or_path}'],
    [sys.executable, '-m', 'src.official_stacks.meta_secalign.test', '--attack', 'straightforward', '--defense', 'none', '--test_data', 'data/TaskTracker_dataset_test.json', '--lora_alpha', '{lora_alpha}', '-m', '{model_name_or_path}'],
]

actual_cmds = []
for model_name_or_path in args.model_name_or_path:
    for lora_alpha in args.lora_alpha:
        for cmd in cmds:
            rendered_lora_alpha = lora_alpha
            if 'gpt' in model_name_or_path or 'gemini' in model_name_or_path:
                if 'test_lm_eval' in ' '.join(cmd):
                    continue
                rendered_lora_alpha = -1
            if 'Llama-3.1-8B' in model_name_or_path and 'test_agentdojo' in ' '.join(cmd):
                continue
            rendered = [part.format(model_name_or_path=model_name_or_path, lora_alpha=rendered_lora_alpha) for part in cmd]
            if '70B' in model_name_or_path:
                rendered.extend(['--tensor_parallel_size', '4'])
            actual_cmds.append(rendered)

for cmd in actual_cmds:
    print(' '.join(cmd) + '\n')
if not args.run:
    print('Run flag is set to False. Exiting without executing commands.')
    raise SystemExit(0)
for cmd in actual_cmds:
    print('\nExecuting...\n' + ' '.join(cmd) + '\n')
    os.spawnvp(os.P_WAIT, cmd[0], cmd)
