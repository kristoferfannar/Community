import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

import traceback
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# check if we're running a train simulation
# or running for real
training = False
for args in sys.argv:
    if args == "train=true":
        training = True
        break

class PartnershipNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(PartnershipNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class TaskScorerNN(nn.Module):
    def __init__(self, task_feature_size, player_state_size, hidden_size):
        super(TaskScorerNN, self).__init__()
        self.fc1 = nn.Linear(task_feature_size + player_state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)  # Outputs a single score for a task

    def forward(self, task_features, player_state):
        # Concatenate task features and player state
        combined = torch.cat(
            [
                task_features if task_features.ndim > 1 else task_features.view(-1),
                player_state,
            ],
            dim=-1,
        )
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        score = self.fc3(x)  # Outputs score
        return score


class RestDecisionNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RestDecisionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)  # Single output for rest score

    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = self.fc2(x)
        score = self.fc3(x)
        return score


def decide_action(
    task_features, player_state, task_scorer, rest_decider, k, max_tasks=100
):
    """
    Decides whether to perform a task or rest.

    Args:
        task_features (list): List of features for each task.
        player_state (object): Current state of the player.
        task_scorer (nn.Module): Neural network scoring tasks.
        rest_decider (nn.Module): Neural network deciding rest.
        k (int): Number of top tasks to consider.
        max_tasks (int): Maximum number of tasks the model can handle.

    Returns:
        action (int): 0 to len(task_features)-1 for tasks, len(task_features) for rest.
    """
    player_state_tensor = torch.tensor(player_state, dtype=torch.float32)

    task_vectors = [torch.tensor(task, dtype=torch.float32) for task in task_features]

    task_scores = torch.tensor(
        [task_scorer(task_vector, player_state_tensor) for task_vector in task_vectors]
    )

    # Pad task scores to max_tasks with 0 if fewer tasks are available
    if len(task_scores) <= max_tasks:
        # 1 for rest
        padding = torch.zeros(max_tasks - len(task_scores) + 1)
        task_scores = torch.cat([task_scores, padding])

    # Select top k scores (valid tasks only)
    num_available_tasks = len(task_features)
    top_k_scores, top_k_indices = torch.topk(
        task_scores[:num_available_tasks], min(k, num_available_tasks)
    )

    # Aggregate features for rest decision
    rest_input = torch.cat(
        [
            torch.mean(top_k_scores).unsqueeze(0),  # Mean score of top k tasks
            torch.tensor(player_state),  # Add player and community features
        ]
    )

    # Score rest action
    rest_score = rest_decider(rest_input).item()

    # Combine task scores and rest score
    combined_scores = task_scores.clone()
    combined_scores[max_tasks] = rest_score  # Append rest score after valid tasks
    num_available_tasks = task_scores.size(0)

    # Find the action with the highest score
    action = torch.argmax(combined_scores).item()

    # If rest is chosen, return []
    if action == max_tasks:
        return []
    # Exclude rest and pick the argmax among the top k tasks
    top_k_scores, top_k_indices = torch.topk(task_scores, k=k)
    # Making sure we dont select any tasks already done
    return [i for i in top_k_indices.tolist() if i < len(task_vectors)]


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


# def create_cost_matrix_raw(community):
#     cost_matrix = []
#     for task in community.tasks:
#         task_costs = []
#         for member in community.members:
#             # Compute the delta and absolute values
#             delta = [max(val - req, 0) for val, req in zip(member.abilities, task)]
#             # Total cost is the sum of deltas
#             total_cost = sum(delta)
#             if member.energy <= -10:
#                 total_cost = float("inf")
#             task_costs.append(total_cost)
#         cost_matrix.append(task_costs)
#     cost_matrix = np.array(cost_matrix)
#     return cost_matrix


# def create_cost_matrix_would_exhaust(community):
#     cost_matrix = []
#     for task in community.tasks:
#         task_costs = []
#         for member in community.members:
#             # Compute the delta and absolute values
#             delta = [max(val - req, 0) for val, req in zip(member.abilities, task)]
#             # Total cost is the sum of deltas
#             total_cost = sum(delta)
#             if member.energy - total_cost <= -10:
#                 total_cost = float("inf")
#             task_costs.append(total_cost)
#         cost_matrix.append(task_costs)
#     cost_matrix = np.array(cost_matrix)
#     return cost_matrix


# def create_cost_matrix_would_tire(community):
#     cost_matrix = []
#     for task in community.tasks:
#         task_costs = []
#         for member in community.members:
#             # Compute the delta and absolute values
#             delta = [max(val - req, 0) for val, req in zip(member.abilities, task)]
#             # Total cost is the sum of deltas
#             total_cost = sum(delta)
#             if member.energy - total_cost < 0:
#                 total_cost = float("inf")
#             task_costs.append(total_cost)
#         cost_matrix.append(task_costs)
#     cost_matrix = np.array(cost_matrix)
#     return cost_matrix


def create_combined_cost_matrix(community):
    num_tasks = len(community.tasks)
    num_members = len(community.members)

    # Initialize a single matrix with an extra dimension for raw, exhaust, and tire
    cost_matrix = np.zeros((num_tasks, num_members, 3))

    for task_idx, task in enumerate(community.tasks):
        for member_idx, member in enumerate(community.members):
            # Compute the delta and total cost
            delta = [max(val - req, 0) for val, req in zip(member.abilities, task)]
            total_cost = sum(delta)

            # Raw cost
            cost_matrix[task_idx, member_idx, 0] = (
                total_cost if member.energy > -10 else float("inf")
            )
            # Tire cost
            cost_matrix[task_idx, member_idx, 1] = (
                total_cost if member.energy - total_cost >= 0 else float("inf")
            )
            # Exhaust cost
            cost_matrix[task_idx, member_idx, 2] = (
                total_cost if member.energy - total_cost > -10 else float("inf")
            )

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



# def create_task_partner_features(player, community):
#     """
#     Create a matrix containing features for each (task, partner) pairing:
#     cost, cosine similarity, and difficulty.
#     """
#     features_matrix = []
#     for task in community.tasks:
#         task_features = []
#         for member in community.members:
#             # Cost
#             # Compute the element-wise maximum of abilities
#             max_abilities = [max(i, j) for i, j in zip(player.abilities, member.abilities)]
#             # Compute the delta and absolute values
#             delta = [abs(max_val - req) for max_val, req in zip(max_abilities, task)]
#             # Total cost is the sum of deltas
#             total_cost = sum(delta)

#             # Cosine Similarity
#             # Combine abilities and task requirements for the cosine similarity
#             combined_abilities = np.array(max_abilities)
#             task_vector = np.array(task)
#             # Avoid division by zero for trivial tasks
#             if np.linalg.norm(combined_abilities) == 0 or np.linalg.norm(task_vector) == 0:
#                 cosine_sim = 0
#             else:
#                 cosine_sim = 1 - cosine(combined_abilities, task_vector)

#             # Difficulty
#             # Difficulty is the sum of task requirements
#             task_difficulty = sum(task) / len(task)

#             # Append the features
#             task_features.append([total_cost, cosine_sim, task_difficulty])
        
#         features_matrix.append(task_features)
    
#     return np.array(features_matrix)


def create_task_partner_features(player, community):
    """
    Create a feature matrix for PartnershipNN.
    Each row includes task features, player parameters, and partner parameters.
    """
    features_matrix = []
    tired, exh = count_tired_exhausted(community)
    player.params = get_player_params(player, community, player.turn, player.num_tasks, tired, exh)
    player_params = np.array(player.params)  # Player's parameters (same for all rows)

    for task in community.tasks:
        task_features = []  # Features for all partners for a single task

        for member in community.members:
            # --- Task Features ---
            # Compute cost
            max_abilities = [max(i, j) for i, j in zip(player.abilities, member.abilities)]
            delta = [abs(max_val - req) for max_val, req in zip(max_abilities, task)]
            total_cost = sum(delta)

            # Compute cosine similarity
            combined_abilities = np.array(max_abilities)
            task_vector = np.array(task)
            if np.linalg.norm(combined_abilities) == 0 or np.linalg.norm(task_vector) == 0:
                cosine_sim = 0
            else:
                cosine_sim = 1 - cosine(combined_abilities, task_vector)

            # Compute difficulty
            task_difficulty = sum(task)

            # --- Partner Parameters ---
            member.params = get_player_params(member, community, player.turn, player.num_tasks, tired, exh)
            partner_params = np.array(member.params)  # Partner-specific parameters

            # Combine features into a single row
            task_features.append(
                np.concatenate(
                    ([total_cost, cosine_sim, task_difficulty], player_params, partner_params)
                )
            )

        features_matrix.append(task_features)

    return np.array(features_matrix)


def create_tasks_feature_vector(player, community):

    player_cost_array = []
    cosine_similarities = []

    num_abilities = len(player.abilities)
    for i, task in enumerate(community.tasks):
        energy_cost = sum(
            [max(task[j] - player.abilities[j], 0) for j in range(num_abilities)]
        )
        # if player.energy - energy_cost >= 0:
        player_cost_array.append(energy_cost)

        dot_product = np.dot(player.abilities, task)
        player_magnitude = np.linalg.norm(player.abilities)
        task_magnitude = np.linalg.norm(task) 
        if player_magnitude == 0 or task_magnitude == 0:
            similarity = 0  # Handle cases where the magnitude is zero
        else:
            similarity = dot_product / (player_magnitude * task_magnitude)
        cosine_similarities.append(similarity)


    combined_cost_matrix = create_combined_cost_matrix(community)

    mat_raw = combined_cost_matrix[:, :, 0]

    mat_tire = combined_cost_matrix[:, :, 1]

    mat_exhaust = combined_cost_matrix[:, :, 2]

    # mat_raw = create_cost_matrix_raw(community)
    # mat_tire = create_cost_matrix_would_tire(community)
    # mat_exhaust = create_cost_matrix_would_exhaust(community)

    tasks_lower_raw = count_lower_cost_players(player_cost_array, mat_raw)
    tasks_lower_tire = count_lower_cost_players(player_cost_array, mat_tire)
    tasks_lower_exhaust = count_lower_cost_players(player_cost_array, mat_exhaust)

    task_costs = []
    for i, task in enumerate(community.tasks):
        subvector = []
        task_difficulty = sum(task) / len(task)
        subvector.append(task_difficulty)
        subvector.append(cosine_similarities[i])
        subvector.append(player_cost_array[i])
        subvector.append(player.energy - player_cost_array[i])

        subvector.append(tasks_lower_raw[i] / len(community.members))
        subvector.append(tasks_lower_tire[i] / len(community.members))
        subvector.append(tasks_lower_exhaust[i] / len(community.members))

        task_costs.append(subvector)
    task_costs = np.array(task_costs)
    return task_costs


def create_partnership_input(player, partner, task_features, task_id):
    player_features = np.array(player.params)
    partner_features = np.array(partner.params)
    task_features = np.array(task_features[task_id])
    return np.concatenate([player_features, partner_features, task_features])


PARTNER_REQUEST_AMOUNT = 5
PARTNERSHIP_TASK_FEATURE_SIZE = 3
P1_HIDDEN_SIZE = 64

def select_top_k_partnerships(score_matrix, k):
    # Flatten the matrix into (task, partner, score) tuples
    flattened = [
        (task_idx, partner_idx, score)
        for task_idx, row in enumerate(score_matrix)
        for partner_idx, score in enumerate(row)
    ]

    # Sort the tuples by score in descending order
    sorted_scores = sorted(flattened, key=lambda x: x[2], reverse=True)

    # Initialize tracking sets
    assigned_tasks = set()
    assigned_partners = set()
    selected_partnerships = []

    # Select top k partnerships
    for task, partner, score in sorted_scores:
        if task not in assigned_tasks and partner not in assigned_partners:
            selected_partnerships.append((task, partner))
            assigned_tasks.add(task)
            assigned_partners.add(partner)
        if len(selected_partnerships) == k:
            break

    return selected_partnerships

def phaseIpreferences(player, community, global_random):
    """Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id
    """
    # if we're training (phase 2), we don't want to do any pairing
    if training:
        return []

    list_choices = []
    # if player.energy < 0:
    #     return list_choices

    # cost_matrix = create_cost_matrix(player, community)


    if not hasattr(player, "turn"):
        player.turn = 1
        player.num_tasks = len(community.members) * 2
        for member in community.members:
            member.num_tasks = len(community.members) * 2
        # Initialize paths and model prefixes
        prefix = "teams/team_2/"
        for arg in sys.argv:
            if arg.startswith("prefix="):
                prefix += "models/" + arg[len("prefix="):] + "_"
                break

        # Load task and rest NNs
        player.taskNN = TaskScorerNN(
            task_feature_size=TASK_FEATURE_SIZE,
            player_state_size=PLAYER_PARAM_SIZE,
            hidden_size=HIDDEN_SIZE,
        )
        player.taskNN.load_state_dict(
            torch.load(prefix + "task_weights.pth", weights_only=True)
        )

        player.restNN = RestDecisionNN(
            input_size=PLAYER_PARAM_SIZE + 1,  # Mean of task scores
            hidden_size=HIDDEN_SIZE,
        )
        player.restNN.load_state_dict(
            torch.load(prefix + "rest_weights.pth", weights_only=True)
        )

        # Initialize PartnershipNN
        player.partnershipNN = PartnershipNN(
            input_size=PLAYER_PARAM_SIZE * 2 + PARTNERSHIP_TASK_FEATURE_SIZE,
            hidden_size=P1_HIDDEN_SIZE,
        )
        player.partnershipNN.load_state_dict(
            torch.load(prefix + "partnership_weights.pth", weights_only=True)
        )

        # Initialize other parameters
        
    else:
        player.turn += 1

    # Update player parameters
    tired, exh = count_tired_exhausted(community)
    features_matrix = create_task_partner_features(player, community)
    flat_features = features_matrix.reshape(-1, features_matrix.shape[-1])
    scores = player.partnershipNN(torch.tensor(flat_features, dtype=torch.float32))
    score_matrix = scores.detach().numpy().reshape(len(community.tasks), len(community.members))

    return select_top_k_partnerships(score_matrix, PARTNER_REQUEST_AMOUNT)
    # player.params = get_player_params(player, community, player.turn, player.num_tasks, tired, exh)
    # task_features = create_tasks_feature_vector(player, community)

    # partnership_scores = []
    # for task_id, task in enumerate(community.tasks):
    #     for partner_id, partner in enumerate(community.members):
    #         input_vector = create_partnership_input(player, partner, task_features, task_id)
    #         input_tensor = torch.tensor(input_vector, dtype=torch.float32)
    #         score = player.partnershipNN(input_tensor).item()
    #         partnership_scores.append((task_id, partner_id, score))

    # partnership_scores.sort(key=lambda x: -x[2])  # Higher scores first

    # selected_partnerships = partnership_scores[:PARTNER_REQUEST_AMOUNT]
    # for task_id, partner_id, score in selected_partnerships:
    #     list_choices.append([task_id, partner_id])

    # best_partner_for_task = [
    #     (task_id, best_partner(cost_matrix[task_id]), cost_matrix[task_id].min())
    #     for task_id in range(len(cost_matrix))
    # ]
    # best_partner_for_task.sort(key=lambda x: x[2])

    # requested_partners = []

    # # to incentivize players to not request pairing up with the best member in the community,
    # # we require that they at least request 5 different partners
    
    # potential_partners = set()
    # curr_idx = 0
    # while len(potential_partners) < PARTNER_REQUEST_AMOUNT and curr_idx < len(
    #     best_partner_for_task
    # ):
    #     task_id, partner_id, cost = best_partner_for_task[curr_idx]
    #     if partner_id not in potential_partners:
    #         requested_partners.append([task_id, partner_id])
    #         potential_partners.add(partner_id)

    #     curr_idx += 1

    # return requested_partners

TASK_FEATURE_SIZE = 7
PLAYER_PARAM_SIZE = 11
HIDDEN_SIZE = 40

def get_player_params(player, community, turn, num_tasks, tired, exh):
    
    return [
                len(community.members),
                len(community.tasks) / num_tasks,
                (1 - len(community.tasks) / num_tasks) ** 2,
                len(community.members) / (len(community.tasks) + 1),
                turn,
                player.energy,
                min(player.energy, 0) ** 2,
                player.energy**3,
                rest_energy_gain(player.energy),  # Energy to gain from resting
                tired,  # Num tired
                exh,  # Num exhausted
            ]


def phaseIIpreferences(player, community, global_random):
    """Return a list of tasks for the particular player to do individually"""
    try:

        # NN part
        # Initialize

        # Hardcoded as 1, to be only the cost of the task - this can be changed.
        

        # if not hasattr(player, "turn"):
        #     prefix = "teams/team_2/"
        #     for arg in sys.argv:
        #         if arg.startswith("prefix="):
        #             prefix += "models/" + arg[len("prefix=") :] + "_"
        #             break

        #     player.taskNN = TaskScorerNN(
        #         task_feature_size=TASK_FEATURE_SIZE,
        #         player_state_size=PLAYER_PARAM_SIZE,
        #         hidden_size=HIDDEN_SIZE,
        #     )
        #     player.taskNN.load_state_dict(
        #         torch.load(prefix + "task_weights.pth", weights_only=True)
        #     )
        #     player.restNN = RestDecisionNN(
        #         # The 1 here is hardcoded because we get a mean of the task scores
        #         input_size=PLAYER_PARAM_SIZE + 1,
        #         hidden_size=HIDDEN_SIZE,
        #     )
        #     player.restNN.load_state_dict(
        #         torch.load(prefix + "rest_weights.pth", weights_only=True)
        #     )

        #     player.turn = 1
        #     player.num_tasks = len(community.members) * 2
            # This should contain the params for decision, such as player.energy, etc
        # else:
        #     player.turn += 1

        tired, exh = count_tired_exhausted(community)
        player.params = get_player_params(player, community, player.turn, player.num_tasks, tired, exh)
        task_features = create_tasks_feature_vector(player, community)
        action = decide_action(
            task_features,
            player.params,
            player.taskNN,
            player.restNN,
            k=min(3, len(community.tasks)),
            max_tasks=player.num_tasks,
        )
        return action
    except Exception as e:
        print(f"CRASH: {e}")
        traceback.print_exc()
