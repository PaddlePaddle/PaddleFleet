#!/usr/bin/env python3
"""Simple script to run the failing test in distributed mode."""
import os
import subprocess
import sys

# Clear paddlejob cluster env
for var in ['PADDLE_TRAINERS', 'PADDLE_TRAINERS_NUM', 'PADDLE_WORKERS_IP_PORT_LIST',
              'PADDLE_TRAINER_ENDPOINTS', 'DISTRIBUTED_TRAINER_ENDPOINTS',
              'PADDLE_CURRENT_ENDPOINT', 'PADDLE_NUM_GRADIENT_SERVERS',
              'POD_INDEX', 'TRAINER_PORTS', 'TRAINER_HOSTS_NUM',
              'PADDLE_TRAINING_ROLE', 'PADDLE_TRAINER_COUNT', 'PADDLE_PORT']:
    if var in os.environ:
        del os.environ[var]

# Use specified Python environment
venv_python = "/root/paddlejob/share-storage/gpfs/system-public/liyamei/PaddleFleet/.venv/bin/python"
if not os.path.exists(venv_python):
    print(f"Error: Python not found at {venv_python}")
    sys.exit(1)

print(f"Using Python: {venv_python}")
print(f"Python version: {subprocess.check_output([venv_python, '-c', 'import paddle; print(paddle.__version__)']).decode().strip()}")

# Build command
cmd = [
    venv_python, "-m", "paddle.distributed.launch",
    "--devices", "0,1,2,3",
    "--run_script", "tests/multi_card_tests/test_subbatch.py",
    "--run_script_args", "TestGPTSubbatchTP.test_gpt_loss_subbatch_equals_no_subbatch",
]

print("Running command:", " ".join(cmd))
sys.stdout.flush()

# Run the test
result = subprocess.run(cmd, env=os.environ.copy())

print("\n" + "="*60)
print("Test completed with return code:", result.returncode)
if result.returncode != 0:
    print("\nTest output:")
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("\nStderr:")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
