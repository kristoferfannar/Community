import subprocess
from torch import nn, save
import random
import os


def run(task_model: nn.Module, rest_model: nn.Module, turns: int, civilians: int):
    prefix = random.randint(0, 100000000)
    task_file = f"{prefix}_task_weights.pth"
    rest_file = f"{prefix}_rest_weights.pth"
    save(task_model.state_dict(), task_file)
    save(rest_model.state_dict(), rest_file)

    result = subprocess.run(
        [
            "python",
            "community.py",
            "--num_members",
            f"{civilians}",
            "--g2",
            f"{civilians}",
            "--num_turns",
            f"{turns}",
            "--seed",
            f"{prefix}",  # use new seeds every time
            f"prefix={prefix}",
        ],
        capture_output=True,
        text=True,
    )

    os.remove(task_file)
    os.remove(rest_file)

    if result.stderr:
        print(result.stderr)

    lines = result.stdout.strip().split("\n")
    for line in lines:
        searchstr = "Total tasks completed: "
        if line.startswith(searchstr):
            completed_tasks = int(line[len(searchstr) :])
            return completed_tasks

    raise Exception("something broke")
