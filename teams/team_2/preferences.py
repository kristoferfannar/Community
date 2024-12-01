import numpy as np
from scipy.optimize import linear_sum_assignment
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def decide_action(task_features, player_state, task_scorer, k, max_tasks=100):
    """
    Decides whether to perform a task or rest.

    Args:
        task_features (list): List of features for each task.
        player_state (object): Current state of the player.
        task_scorer (nn.Module): Neural network scoring tasks.
        k (int): Number of top tasks to consider.
        max_tasks (int): Maximum number of tasks the model can handle.

    Returns:
        action (int): 0 to len(task_features)-1 for tasks, len(task_features) for rest.
    """
    if len(task_features) == 0:
        return []
    player_state_tensor = torch.tensor(player_state, dtype=torch.float32)

    task_vectors = [torch.tensor(task, dtype=torch.float32) for task in task_features]

    task_scores = torch.tensor(
        [task_scorer(task_vector, player_state_tensor) for task_vector in task_vectors]
    )

    # Pad task scores to max_tasks with 0 if fewer tasks are available
    # if len(task_scores) <= max_tasks:
    #     # 1 for rest
    #     padding = torch.zeros(max_tasks - len(task_scores) + 1)
    #     task_scores = torch.cat([task_scores, padding])

    # Select top k scores (valid tasks only)
    num_available_tasks = len(task_features)
    # top_k_scores, top_k_indices = torch.topk(
    #     task_scores[:num_available_tasks], min(k, num_available_tasks)
    # )

    top_k_scores, top_k_indices = torch.topk(task_scores, min(k, num_available_tasks))

    # Combine task scores and rest score
    num_available_tasks = task_scores.size(0)

    # Find the action with the highest score
    try:
        action = torch.argmax(task_scores).item()
    except Exception:
        pass

    # If rest is chosen, return []
    if action == 0:
        return []
    # Exclude rest and pick the argmax among the top k tasks
    # top_k_scores, top_k_indices = torch.topk(task_scores, k=k)
    # Making sure we dont select any tasks already done
    return [i for i in top_k_indices.tolist()]


def count_tired_exhausted(community):
    tired = len([m for m in community.members if -10 < m.energy < 0])
    exh = len([m for m in community.members if -10 > m.energy])
    return tired, exh


def rest_energy_gain(energy):
    if abs(energy) == 10:
        return 0
    if energy < 0:
        return 0.5
    return 1


def create_cost_matrix(player, community):
    cost_matrix = []
    for task in community.tasks:
        task_costs = []
        for member in community.members:
            # Compute the element-wise maximum of abilities
            max_abilities = [
                max(i, j) if member.energy >= 0 else float("inf")
                for i, j in zip(player.abilities, member.abilities)
            ]
            # Compute the delta and absolute values
            delta = [abs(max_val - req) for max_val, req in zip(max_abilities, task)]
            # Total cost is the sum of deltas
            total_cost = sum(delta)
            task_costs.append(total_cost)
        cost_matrix.append(task_costs)
    cost_matrix = np.array(cost_matrix)
    return cost_matrix


def create_cost_matrix_raw(community):
    cost_matrix = []
    for task in community.tasks:
        task_costs = []
        for member in community.members:
            # Compute the delta and absolute values
            delta = [max(val - req, 0) for val, req in zip(member.abilities, task)]
            # Total cost is the sum of deltas
            total_cost = sum(delta)
            if member.energy <= -10:
                total_cost = float("inf")
            task_costs.append(total_cost)
        cost_matrix.append(task_costs)
    cost_matrix = np.array(cost_matrix)
    return cost_matrix


def create_cost_matrix_would_exhaust(community):
    cost_matrix = []
    for task in community.tasks:
        task_costs = []
        for member in community.members:
            # Compute the delta and absolute values
            delta = [max(val - req, 0) for val, req in zip(member.abilities, task)]
            # Total cost is the sum of deltas
            total_cost = sum(delta)
            if member.energy - total_cost <= -10:
                total_cost = float("inf")
            task_costs.append(total_cost)
        cost_matrix.append(task_costs)
    cost_matrix = np.array(cost_matrix)
    return cost_matrix


def create_cost_matrix_would_tire(community):
    cost_matrix = []
    for task in community.tasks:
        task_costs = []
        for member in community.members:
            # Compute the delta and absolute values
            delta = [max(val - req, 0) for val, req in zip(member.abilities, task)]
            # Total cost is the sum of deltas
            total_cost = sum(delta)
            if member.energy - total_cost < 0:
                total_cost = float("inf")
            task_costs.append(total_cost)
        cost_matrix.append(task_costs)
    cost_matrix = np.array(cost_matrix)
    return cost_matrix


def count_lower_cost_players(player_cost_array, cost_matrix):
    """
    Count the number of players with lower costs than the given player for each task.

    Args:
        player_cost_array (np.ndarray): 1D array of the player's costs for each task.
        cost_matrix (np.ndarray): 2D array of shape (num_tasks, num_members), where each entry is the cost for a member to perform a task.

    Returns:
        list: A list where each element is the count of players with lower costs for the corresponding task.
    """
    # Ensure the player's cost array is an array
    player_cost_array = np.array(player_cost_array)

    # Compare the player's costs with all members' costs for each task
    lower_cost_counts = np.sum(cost_matrix < player_cost_array[:, None], axis=1)

    return lower_cost_counts.tolist()


def best_partner(task: np.ndarray):
    for partner_id in range(len(task)):
        if task[partner_id] == task.min():
            return partner_id

    raise Exception("All arrays have a minimum value")


def create_tasks_feature_vector(player, community):

    player_cost_array = []

    num_abilities = len(player.abilities)
    for i, task in enumerate(community.tasks):
        energy_cost = sum(
            [max(task[j] - player.abilities[j], 0) for j in range(num_abilities)]
        )
        # if player.energy - energy_cost >= 0:
        player_cost_array.append(energy_cost)

    mat_raw = create_cost_matrix_raw(community)
    mat_tire = create_cost_matrix_would_tire(community)
    mat_exhaust = create_cost_matrix_would_exhaust(community)

    tasks_lower_raw = count_lower_cost_players(player_cost_array, mat_raw)
    tasks_lower_tire = count_lower_cost_players(player_cost_array, mat_tire)
    tasks_lower_exhaust = count_lower_cost_players(player_cost_array, mat_exhaust)

    task_costs = []
    # Rest is option 0
    subvector = []
    gain = 0
    cost = rest_energy_gain(player.energy)
    task_difficulty = 0
    subvector.append(gain)
    subvector.append(cost)
    subvector.append(task_difficulty)
    subvector.append(0)
    subvector.append(0)
    subvector.append(0)

    for i, task in enumerate(community.tasks):
        subvector = []
        gain = 1
        task_difficulty = sum(task) / len(task)
        cost = player_cost_array[i]
        subvector.append(gain)
        subvector.append(cost)
        subvector.append(task_difficulty)
        subvector.append(tasks_lower_raw[i] / len(community.members))
        subvector.append(tasks_lower_tire[i] / len(community.members))
        subvector.append(tasks_lower_exhaust[i] / len(community.members))

        task_costs.append(subvector)
    task_costs = np.array(task_costs)
    return task_costs


def phaseIpreferences(player, community, global_random):
    """Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id
    """
    return []


def phaseIIpreferences(player, community, global_random):
    """Return a list of tasks for the particular player to do individually"""
    try:

        # bids.sort(key=lambda x: (x[1], -sum(community.tasks[x[0]])))
        # return [b[0] for b in bids[:3]]

        # NN part
        # Initialize

        # Hardcoded as 1, to be only the cost of the task - this can be changed.
        task_feature_size = 6
        player_params_size = 9
        hidden_size = 64

        if not hasattr(player, "turn"):
            player.taskNN = TaskScorerNN(
                task_feature_size=task_feature_size,
                player_state_size=player_params_size,
                hidden_size=hidden_size,
            )
            player.taskNN.load_state_dict(
                torch.load("task_weights.pth", weights_only=True)
            )
            player.turn = 1
            player.num_tasks = len(community.members) * 2
            # This should contain the params for decision, such as player.energy, etc
            player.params = [
                len(community.members),
                len(community.tasks),
                len(community.members) / (len(community.tasks) + 1),
                player.turn,
                player.energy,
                min(player.energy, 0) ** 2,
                0,  # Energy to gain from resting
                0,  # Num tired
                0,  # Num exhausted
            ]
        else:
            player.turn += 1
            tired, exh = count_tired_exhausted(community)
            player.params = [
                len(community.members),
                len(community.tasks),
                len(community.members) / (len(community.tasks) + 1),
                player.turn,
                player.energy,
                min(player.energy, 0) ** 2,
                rest_energy_gain(player.energy),
                tired,
                exh,
            ]

        task_features = create_tasks_feature_vector(player, community)
        action = decide_action(
            task_features,
            player.params,
            player.taskNN,
            k=min(3, len(community.tasks)),
            max_tasks=player.num_tasks,
        )
        return action
    except Exception as e:
        print(f"CRASH: {e}")
        traceback.print_exc()
