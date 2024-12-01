import subprocess
from torch import nn, save


def run(task_model: nn.Module, turns, civilians):
    save(task_model.state_dict(), "task_weights.pth")

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
        ],
        capture_output=True,
        text=True,
    )

    if result.stderr:
        print(result.stderr)

    lines = result.stdout.strip().split("\n")
    for line in lines:
        searchstr = "Total tasks completed: "
        if line.startswith(searchstr):
            completed_tasks = int(line[len(searchstr) :])
            return completed_tasks

    raise Exception("something broke")
