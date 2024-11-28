from scipy.optimize import linear_sum_assignment
import numpy as np


PHASE_1_ASSIGNMENTS = True


def exists_good_match(
    tasks,
    player_abilities,
    each_difference=1,
    total_difference=None,
    penalty_tolerance=0,
    return_multiple_tasks=False,
):
    # go through all the tasks and check if there is a task whose difficulty in each category is less than the player's ability in that category by each_tolerance
    no_of_abilities = len(player_abilities)
    current_index = -1
    penalty_suffered = -1
    if total_difference == None:
        total_difference = no_of_abilities * each_difference

    returned_tasks = []
    for task in tasks:
        if (
            all(
                abs(player_abilities[i] - task[i]) <= each_difference
                for i in range(no_of_abilities)
            )
            and sum(
                [abs(player_abilities[i] - task[i]) for i in range(no_of_abilities)]
            )
            <= total_difference
        ):
            penalty = 0
            for i in range(no_of_abilities):
                if player_abilities[i] < task[i]:
                    penalty += task[i] - player_abilities[i]
            if penalty <= penalty_tolerance:
                returned_tasks.append(task)
                if penalty_suffered == -1 or penalty < penalty_suffered:
                    penalty_suffered = penalty
                    current_index = tasks.index(task)

    if current_index == -1:
        return False, penalty_suffered, current_index, []
    else:
        # print("TASKS: ", returned_tasks)
        return True, penalty_suffered, current_index, returned_tasks


def phaseIpreferences(player, community, global_random):
    """Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id
    """
    list_choices = []

    try:
        if PHASE_1_ASSIGNMENTS:
            assignments, total_cost = assign_phase1(community.tasks, community.members)

            for assignment in assignments:
                if len(assignment[0]) == 2 and player.id in assignment[0]:
                    for task in community.tasks:
                        if task == assignment[1]:
                            if player.id == assignment[0][0]:
                                list_choices.append(
                                    [community.tasks.index(task), assignment[0][1]]
                                )
                            else:
                                list_choices.append(
                                    [community.tasks.index(task), assignment[0][0]]
                                )
        else:
            perfect_match, _, _, _ = exists_good_match(
                community.tasks,
                player.abilities,
                each_difference=1,
                penalty_tolerance=2,
            )
            if perfect_match:
                # print("PERFECT MATCH")
                return []
            else:
                for member in community.members:
                    max_abilities = [
                        max(player.abilities[i], member.abilities[i])
                        for i in range(len(player.abilities))
                    ]
                    _, _, _, matching_tasks = exists_good_match(
                        community.tasks,
                        max_abilities,
                        each_difference=3,
                        penalty_tolerance=max(len(player.abilities), 6),
                        return_multiple_tasks=True,
                    )
                    for matching_task in matching_tasks:
                        list_choices.append(
                            [community.tasks.index(matching_task), member.id]
                        )

                threshold = 3
                if len(list_choices) < threshold:
                    for member in community.members:
                        max_abilities = [
                            max(player.abilities[i], member.abilities[i])
                            for i in range(len(player.abilities))
                        ]
                        _, _, _, matching_tasks = exists_good_match(
                            community.tasks,
                            max_abilities,
                            each_difference=4,
                            penalty_tolerance=8,
                            return_multiple_tasks=True,
                        )
                        for matching_task in matching_tasks:
                            list_choices.append(
                                [community.tasks.index(matching_task), member.id]
                            )

    except Exception as e:
        print(e)

    # print(list_choices)
    return list_choices


def phaseIIpreferences(player, community, global_random):
    """Return a list of tasks for the particular player to do individually"""
    bids = []
    if player.energy < 0:
        return bids

    try:
        wait_energy_threshold = 0
        player_index = community.members.index(player)
        assignments, total_cost = assign_phase2(community.tasks, community.members)

        best_task = assignments.get(player_index)
        if best_task is None:
            return []

        best_task_cost = loss_phase2(
            community.tasks[best_task], player.abilities, player.energy
        )
        if player.energy - best_task_cost < wait_energy_threshold:
            return []

        return [best_task]
    except Exception as e:
        print(e)
        return bids


def assign_phase1(tasks, members):
    num_tasks = len(tasks)
    num_members = len(members)

    # Generate all possible partnerships
    partnerships = []
    for i in range(num_members):
        for j in range(i + 1, num_members):
            partnerships.append((i, j))
    num_partnerships = len(partnerships)

    # Create cost matrix with columns for both partnerships and individual assignments
    cost_matrix = np.zeros((num_tasks, num_partnerships + num_members))

    # Fill partnership costs (first num_partnerships columns)
    for i, task in enumerate(tasks):
        for j, (member1_idx, member2_idx) in enumerate(partnerships):
            member1 = members[member1_idx]
            member2 = members[member2_idx]
            cost_matrix[i][j] = loss_phase1(task, member1, member2)

    # Fill individual costs (remaining columns)
    for i, task in enumerate(tasks):
        for j, member in enumerate(members):
            cost_matrix[i][num_partnerships + j] = loss_phase2(
                task, member.abilities, member.energy
            )

    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Convert results to assignments
    assignments = []
    for task_idx, col_idx in zip(row_indices, col_indices):
        task = tasks[task_idx]
        # If column index is less than num_partnerships, it's a partnership
        if col_idx < num_partnerships:
            member1_idx, member2_idx = partnerships[col_idx]
            member1 = members[member1_idx]
            member2 = members[member2_idx]
            loss = cost_matrix[task_idx, col_idx]
            assignments.append(([member1.id, member2.id], task, loss))
        else:
            # Individual assignment
            member_idx = col_idx - num_partnerships
            member = members[member_idx]
            loss = cost_matrix[task_idx, col_idx]
            assignments.append(([member.id], task, loss))

    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return assignments, total_cost


def assign_phase2(tasks, members):
    num_tasks = len(tasks)
    num_members = len(members)

    cost_matrix = np.zeros((num_tasks, num_members))

    for i, task in enumerate(tasks):
        for j, member in enumerate(members):
            cost_matrix[i][j] = loss_phase2(task, member.abilities, member.energy)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    assignments = {player: task for task, player in zip(row_indices, col_indices)}
    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return assignments, total_cost


def loss_phase1(task, player1, player2):
    cost = sum(
        max(task[k] - max(player1.abilities[k], player2.abilities[k]), 0)
        for k in range(len(task))
    )
    cost += (max(0, cost - player1.energy) + max(0, cost - player2.energy)) / 2
    # cost += sum(
    #     max(max(player1.abilities[k], player2.abilities[k]) - task[k], 0)
    #     for k in range(len(task))
    # )
    # cost += sum(
    #     abs(player1.abilities[k] - player2.abilities[k]) for k in range(len(task))
    # )
    return cost

def loss_phase2(task, abilities, current_energy):
    cost = sum([max(task[k] - abilities[k], 0) for k in range(len(abilities))])
    cost += max(0, cost - current_energy)
    # cost += sum([abs(abilities[k] - task[k]) for k in range(len(abilities))])
    return cost
