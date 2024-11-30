import numpy as np
import matplotlib.pyplot as plt
import random
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_GENERATIONS = 10
POP_SIZE = 100
HIDDEN_SIZE = 64
TASK_FEATURE_SIZE = 6
PLAYER_STATE_SIZE = 9


if __name__ == "__main__":
    from run import run
else:
    from teams.team_2.training.run import run


class TaskScorerNN(nn.Module):
    def __init__(self, task_feature_size, player_state_size, hidden_size):
        super(TaskScorerNN, self).__init__()
        input_size = task_feature_size + player_state_size

        # Define layers with specified sizes
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.output_layer = nn.Linear(
            hidden_size // 4, 1
        )  # Outputs a single score for a task

    def forward(self, task_features, player_state):
        # Concatenate task features and player state
        combined = torch.cat(
            [
                task_features if task_features.ndim > 1 else task_features.view(-1),
                player_state,
            ],
            dim=-1,
        )

        # Apply layers with ReLU activation
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        score = self.output_layer(x)  # Final score output
        return score


def evaluate_fitness(task_model: nn.Module):
    # for task in task_environment:
    # action = model(task.features, task.state)
    result = run(task_model)
    # result = task_environment.run(action)  # Simulates environment behavior
    return result


def crossover(parent1, parent2, is_task):
    child = TaskScorerNN(TASK_FEATURE_SIZE, PLAYER_STATE_SIZE, HIDDEN_SIZE)

    for param1, param2, child_param in zip(
        parent1.parameters(), parent2.parameters(), child.parameters()
    ):
        child_param.data.copy_((param1.data + param2.data) / 2)  # Weighted average
    return child


def mutate(model, mutation_rate=0.1, mutation_noise=0.01):
    for param in model.parameters():
        if torch.rand(1).item() < mutation_rate:
            param.data += (
                torch.randn_like(param.data) * mutation_noise
            )  # Small random noise


def select_parents(population, fitness_scores):
    best = sorted(
        list(zip(population, fitness_scores)), key=lambda x: x[1], reverse=True
    )

    return [b[0] for b in best][: len(population) // 2]


def task_scorer():
    pass


def rest_scorer():
    pass


class Task:
    def __init__(self, features: torch.Tensor, state=None):
        self.features = features
        self.state = state
        if not state:
            self.state = torch.zeros(features.shape[0])


population = [
    TaskScorerNN(TASK_FEATURE_SIZE, PLAYER_STATE_SIZE, HIDDEN_SIZE)
    for _ in range(POP_SIZE)
]

avg_scores = []
max_scores = []

for generation in range(MAX_GENERATIONS):
    print(f"[*] Generation - {generation}")
    # Evaluate fitness
    fitness_scores = [evaluate_fitness(task_model) for task_model in population]
    avg_scores.append(np.mean(fitness_scores))
    max_scores.append(max(fitness_scores))

    # Select parents
    parents = select_parents(population, fitness_scores)

    # Generate offspring
    offspring = []
    # for now, only create offspring from TaskScorerNN
    for _ in range(len(parents) // 2):
        parent1, parent2 = random.sample(parents, 2)
        child_task = crossover(parent1, parent2, is_task=True)
        mutate(child_task, 0.1, 0.05)
        offspring.append(child_task)

        parent1, parent2 = random.sample(parents, 2)
        child_task = crossover(parent1, parent2, is_task=True)
        mutate(child_task, 0.1, 0.05)
        offspring.append(child_task)

    # Replace population
    [(mutate(ptask)) for ptask in parents]
    population = parents + offspring
    print(f"{len(population)} members")

    print(f"{generation}: fitness scores: {sorted(fitness_scores, reverse=True)[:3]}")


if __name__ == "__main__":
    best_model = select_parents(population, fitness_scores)[0]

    torch.save(best_model.state_dict(), "best_weigths.pth")
    print('best model weights saved in "best_weights.pth"')

    # Get the current working directory (runfolder)
    runfolder = os.getcwd()

    # Plot both curves
    plt.plot(
        np.arange(len(avg_scores)), avg_scores, label="Average Score", color="blue"
    )
    plt.plot(np.arange(len(max_scores)), max_scores, label="Max Score", color="red")

    # Add labels and title
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title(
        f"Average and Max Fitness Scores Over {MAX_GENERATIONS} Generations\nInitial population size = {POP_SIZE}"
    )

    # Add a legend
    plt.legend()

    # Save the plot to the current directory
    plt.savefig(os.path.join(runfolder, "fitness_scores.png"))
    print("path", os.path.join(runfolder, "fitness_scores.png"))

    # task_scores = [task_scorer(task, player_state) for task in tasks]
    # rest_score = rest_scorer(player_state)
    # combined_scores = torch.cat([task_scores, rest_score.unsqueeze(0)])
    # action = torch.argmax(combined_scores).item()
