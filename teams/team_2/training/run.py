import subprocess
from torch import nn, save


def run(task_model: nn.Module):
    save(task_model.state_dict(), "task_weights.pth")

    result = subprocess.run(
        [
            "python",
            "community.py",
            "--num_members",
            "10",
            "--g2",
            "10",
            "--num_turns",
            "20",
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
            print(f"result: {completed_tasks}")
            return completed_tasks

    raise Exception("something broke")
