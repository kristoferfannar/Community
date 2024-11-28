import numpy as np
import csv

tracking_data = []

# Partnership Round
def phaseIpreferences(player, community, global_random):
    if player.energy <= 5:
        return []

    partner_bids = get_partner_bids(player, community)
    return partner_bids


# Individual Round
def phaseIIpreferences(player, community, global_random):
    solo_bids = get_all_possible_tasks(community, player)
    return solo_bids

def get_partner_bids(player, community):
    """
    This function calculates a person's cost matrix for performing every task alone or partnered and then returns a preference to work alone or in partnerships.

    :return solo_preferences:
            partnered_preferences:
    """
    partnered_abilities = get_possible_partnerships(player, community)
    penalty_matrix = calculate_penalty_matrix(partnered_abilities, community.tasks)
    partner_bids = []

    # Iterate over each column (penalties for single task)
    do_task_alone_cutoff = min(2, player.energy)
    for task_index, column in enumerate(penalty_matrix.T):
        solo_penalty = column[0]

        # Do not partner if the task alone has (1) no penalty alone or (2) is "easy enough".
        if (solo_penalty == 0) or (solo_penalty <= do_task_alone_cutoff):
            continue

        # Partner with anyone where the task penalty is bottom 5% of penalty difficulty for that task.
        do_task_partnered_cutoff = round(np.percentile(np.array(column[1:]), 5))
        for offset_player_index, penalty in enumerate(column[1:]):
            if (penalty == 0) or (penalty <= do_task_partnered_cutoff):
                # Remember that the player_abilities index for partnered abilities started at 1 not 0.
                # To get the original partner id we need to reset adn use the community members object.
                partner_id = community.members[offset_player_index - 1].id
                partner_bids.append([task_index, partner_id])

    return partner_bids


def get_possible_partnerships(player, community):
    """
    :return: [2D array] containing a partnership's combined ability level.
        Row 0 is for the player's solo abilities.
        Row 1:N is for the combo player/partner abilities.
    """
    partnered_abilities = np.array(
        [player.abilities] + [[int(max(p, q)) for p, q in zip(player.abilities, partner.abilities)] for
                              partner in community.members])
    return partnered_abilities


def calculate_penalty_matrix(partnered_abilities, tasks):
    """
    Calculate the penalty matrix for given players and tasks.

    Args:
    - team_abilities: A 2D list or array where each row represents a player, or partnerships, abilities. Index 0 is the individual player. Index 1 to N represents the partnerships between the player and
    - tasks: A 2D list or array where each row represents a task's requirements.

    Returns:
    - A 2D numpy array containing the penalties for each partner-task pair.
    """

    penalty_matrix = []
    for i, abilities in enumerate(partnered_abilities):
        row_penalties = []
        for task in tasks:
            # Calculate energy expended as the sum of positive differences split between two partners
            penalty = np.sum(np.maximum(0, task - abilities)) / 2
            # First value is individual completing the task so it is not shared
            if i == 0:
                penalty = penalty * 2
            row_penalties.append(penalty)
        penalty_matrix.append(row_penalties)

    return np.array(penalty_matrix)


def get_all_possible_tasks(community, player):
    """
    Volunteer for all tasks that keep your energy level above 0.
    """
    solo_bids = []
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    for task_index, task in sorted_tasks:
        # Volunteer for all tasks that will keep your energy above 0.
        energy_cost = sum(max(task[i] - player.abilities[i], 0) for i in range(len(task)))
        if energy_cost <= player.energy:
            solo_bids.append(task_index)

    return solo_bids


def log_turn_data(turn, community, tasks_completed):
    """
    Logs the state of the community for a single turn.
    """
    global tracking_data

    energy_levels = [player.energy for player in community.members]
    median_energy = np.median(energy_levels)
    exhausted_count = sum(1 for energy in energy_levels if energy < 0)

    tracking_data.append({
        "Turn": turn,
        "Tasks Completed": tasks_completed,
        "Median Energy": median_energy,
        "Exhausted Players": exhausted_count,
    })


def export_csv(filename="simulation_data.csv"):
    """
    Exports the logged data to a CSV file.
    """
    global tracking_data
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Turn", "Tasks Completed", "Median Energy", "Exhausted Players"])
        writer.writeheader()
        writer.writerows(tracking_data)

"""
Retired Helper Functions 
"""


def get_best_partner(preferences, tasks, task_index, player, community):
    """
    Volunteer for every task with the minimum penalty partner.
    """

    # Returns a list of [task_index, partner_id]
    preferences = []
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    # Iterate over all tasks
    for task_index, task in sorted_tasks:

        # Find a partner with complementary skills and sufficient energy
        best_partner = None
        min_energy_cost = float('inf')

        for partner in community.members:
            # Skip invalid partners
            if partner.id == player.id or partner.energy < 0:
                continue

            # Compute combined abilities
            combined_abilities = [
                max(player.abilities[i], partner.abilities[i]) for i in range(len(task))
            ]

            # Compute energy cost per partner for the task
            energy_cost = sum(
                max(task[i] - combined_abilities[i], 0) for i in range(len(task))
            ) / 2

            # Finding best partner for the task.
            if (
                    energy_cost < min_energy_cost
                    and energy_cost <= player.energy
                    and energy_cost <= partner.energy
            ):
                min_energy_cost = energy_cost
                best_partner = partner.id

        if best_partner is not None:
            preferences.append([task_index, best_partner])

    return preferences
