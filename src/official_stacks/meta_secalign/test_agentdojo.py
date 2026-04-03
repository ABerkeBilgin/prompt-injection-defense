# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import subprocess
import time
from datetime import datetime

import yaml

from .paths import AGENTDOJO_SRC_DIR, add_vendor_paths

add_vendor_paths()

from .utils import load_vllm_model_with_changed_lora_alpha, model_uses_lora, resolve_base_model_path, summary_results, test_parser

args = test_parser()
if args.defense != 'repeat_user_prompt' or ('important_instructions' not in args.attack and 'none' not in args.attack):
    print('Warning: not using attack=important_instructions/none with defense=repeat_user_prompt as in Meta-SecAlign paper.')


def print_while_saving(command: str, log_path: str, env: dict | None = None) -> str:
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=str(AGENTDOJO_SRC_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    lines = []
    with open(log_path, 'w', encoding='utf-8', newline='\n') as handle:
        for line in proc.stdout:
            print(line, end='')
            handle.write(line)
            handle.flush()
            lines.append(line)
    proc.wait()
    return ''.join(lines)


def parse_percentages(output: str, prefix: str) -> float | None:
    matches = re.findall(rf"{re.escape(prefix)}\s*([0-9]+(?:\.[0-9]+)?)%", output)
    if len(matches) == 0:
        return None
    values = [float(item) for item in matches]
    return sum(values) / len(values)


for model_name_or_path in args.model_name_or_path:
    env = os.environ.copy()
    env['PYTHONPATH'] = str(AGENTDOJO_SRC_DIR) + os.pathsep + env.get('PYTHONPATH', '')
    if 'gpt' not in model_name_or_path and 'gemini' not in model_name_or_path:
        base_model_path = resolve_base_model_path(model_name_or_path)
        command = 'vllm serve %s --dtype auto --host 0.0.0.0 --tensor-parallel-size %d --max-model-len 24576' % (base_model_path, args.tensor_parallel_size)
        if model_uses_lora(model_name_or_path):
            model_name_or_path = load_vllm_model_with_changed_lora_alpha(model_name_or_path, args.lora_alpha)
            command += ' --enable-lora --max-lora-rank 64 --lora-modules %s=%s' % (model_name_or_path, model_name_or_path)
        log_dir = os.getcwd() + '/' + model_name_or_path
        if not os.path.exists(log_dir):
            log_dir += '-log'
        log_dir += '/agentdojo'
        os.makedirs(log_dir, exist_ok=True)
        summary_path = os.path.join(os.path.dirname(log_dir), 'summary.tsv')
        server_log = log_dir + '/vllm_server_%s.out' % datetime.now().strftime('%Y%m%d_%H%M%S')
        command = 'nohup ' + command + ' > ' + server_log + ' 2>&1 &'
        os.system(command)
        time.sleep(30)

        while True:
            with open(server_log, 'r', encoding='utf-8', errors='ignore') as handle:
                txt = handle.read()
                pids = re.findall(r'(?<=pid=).*?(?=\))', txt)
                if 'Application startup complete' in txt:
                    break
            print('Waiting another 30s for vLLM server to start...')
            time.sleep(30)
        print('Evaluating AgentDojo on', model_name_or_path, 'with attacks', args.attack, 'and defense', args.defense, end='\n\n\n')
        command = 'python -m agentdojo.scripts.benchmark --model local --logdir %s --model-id %s --tool-delimiter input' % (log_dir, model_name_or_path)
    else:
        log_dir = os.getcwd() + '/' + model_name_or_path + '-log/agentdojo'
        os.makedirs(log_dir, exist_ok=True)
        summary_path = os.path.join(os.path.dirname(log_dir), 'summary.tsv')
        command = 'python -m agentdojo.scripts.benchmark --model %s --logdir %s' % (model_name_or_path, log_dir)
        if 'gpt-5' in model_name_or_path:
            command += ' --reasoning-effort %s' % args.gpt5_reasoning_effort
        pids = []
        if 'gpt' in model_name_or_path:
            with open(args.openai_config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)['default']
            usable_keys = []
            for item in config:
                if item.get('azure_deployment', model_name_or_path) == model_name_or_path:
                    item = dict(item)
                    item.pop('azure_deployment', None)
                    usable_keys.append(item)
            if len(usable_keys) == 0:
                print('No usable OpenAI or Azure key found for model', model_name_or_path)
                exit()
            env['AZURE_API_KEY'] = usable_keys[0]['api_key']
            env['AZURE_API_ENDPOINT'] = usable_keys[0]['azure_endpoint']
            env['AZURE_API_VERSION'] = usable_keys[0]['api_version']
        elif 'gemini' in model_name_or_path:
            with open(args.gemini_config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)['default']
            env['GOOGLE_API_KEY'] = config[1]['api_key']

    if args.defense != 'none':
        command += ' --defense %s' % args.defense
    try:
        for attack in args.attack:
            attack_log = os.path.join(log_dir, f'summary_{attack}.log')
            if attack == 'none':
                output = print_while_saving(command, attack_log, env=env)
                utility = parse_percentages(output, 'Average utility:')
                if utility is not None:
                    summary_results(summary_path, {
                        'attack': 'AgentDojo-none',
                        'ASR/Utility': f'{utility:.2f}%',
                        'defense': args.defense,
                        'instruction_hierarchy': args.instruction_hierarchy,
                        'lora_alpha': args.lora_alpha,
                        'test_data': 'AgentDojo',
                    })
            else:
                output = print_while_saving(command + ' --attack %s' % attack, attack_log, env=env)
                security = parse_percentages(output, 'Average security:')
                if security is not None:
                    summary_results(summary_path, {
                        'attack': f'AgentDojo-{attack}',
                        'ASR/Utility': f'{100 - security:.2f}%',
                        'defense': args.defense,
                        'instruction_hierarchy': args.instruction_hierarchy,
                        'lora_alpha': args.lora_alpha,
                        'test_data': 'AgentDojo',
                    })
    except Exception as exc:
        print('Error occurred:', exc)
        break
    if len(pids):
        os.system('kill -9 %s' % ' '.join(set(pids)))
