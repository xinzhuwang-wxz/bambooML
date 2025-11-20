import os

def run(args):
    if args.system == 'local':
        content = f"#!/bin/bash\n{args.cmdline}\n"
    else:
        content = """#!/bin/bash
#SBATCH --job-name=bambooml
#SBATCH --output=bambooml.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
""" + f"\n{args.cmdline}\n"
    os.makedirs(os.path.dirname(args.script), exist_ok=True)
    with open(args.script, 'w') as f:
        f.write(content)
    print('Script written to', args.script)
