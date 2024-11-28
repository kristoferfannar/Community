import pprint
from typing import List
import numpy as np
from community import Community, Member
import pandas as pd
import os
from teams.team_10.constants import *

# Task 1: Figure out global variables...
# Nicky 

# Task 2: NEW SACRIFICING STRATEGY -- IMPLEMENTED 
# Julianna 

# Task 3: Check if any player is capable of doing certains tasks... (Open ended) -- THINKING ON THIS
# Julianna

# Task 4:
# We want to make sure we don't voluenteer for tasks that an individual can complete (IN PROGRESS)
# Julianna 

# Task 5: 
# Only volenteering the best 5 or 10 tasks (Max)
# 1. We'll probably get a speed up from early returning
# 2. Don't really want to step on other peoples
# Akhil

# Task 6: Variable lower bounds.. Would be cool to adjust lower bounds gradually? 
# Akhil

def phaseIpreferences(player, community: Community, global_random):
    """Return a list of task index and the partner id for the particular player.
    The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list]
    and the second index as the partner id"""
    # global tasks_at_turn
    # global acceptable_energy_level_at_turn
    
    csv_file = "teams/team_10/player_data.csv"
    columns = ["turn", "player_id", "tasks_at_turn", "acceptable_energy_level_at_turn"]
    player_data = pd.read_csv(csv_file)
    
    most_recent_turn = player_data["turn"].max()
    
    # THIS IS A HACK TO CLEAR THE DATA IF WE START A NEW ROUND... THIS IS A TERRIBLE IDEA>>>
    if community.completed_tasks == 0 and most_recent_turn > 20:
        player_data = pd.DataFrame(columns=columns)

    
    # ONLY KEEP THE LAST 20 TURNS!
    player_data = player_data[player_data["turn"] > most_recent_turn - 20]
    if len(player_data) == 0:
        player_data = pd.DataFrame(columns=columns)
        
    tasks_at_turn = player_data[player_data["player_id"] == player.id]['tasks_at_turn'].tolist()

    # player.tasks_at_turn.append(len(community.tasks))
    
    acceptable_energy_level = get_acceptable_energy_level(tasks_at_turn)
    # player.acceptable_energy_level_at_turn.append(acceptable_energy_level)
    new_data = pd.DataFrame([{
        "turn": len(tasks_at_turn),
        "player_id": player.id, 
        "tasks_at_turn": len(community.tasks), 
        "acceptable_energy_level_at_turn": acceptable_energy_level
    }])
    
    player_data = pd.concat([player_data, new_data], ignore_index=True)
    
    player_data.to_csv(csv_file, index=False)

    task_pairs = find_pairs(
        player, community.tasks, community.members, acceptable_energy_level
    )
    return task_pairs


def phaseIIpreferences(player: Member, community, global_random):
    """Return a list of tasks for the particular player to do individually"""
    
    csv_file = "teams/team_10/player_data.csv"
    columns = ["turn", "player_id", "tasks_at_turn", "acceptable_energy_level_at_turn"]
    all_player_data = pd.read_csv(csv_file, usecols=columns) # Data for all players
    
    player_data = all_player_data[all_player_data["player_id"] == player.id] # Data for this specific player
    
    tasks_at_turn = player_data["tasks_at_turn"].tolist() # Tasks at turn for this player
    acceptable_energy_level_at_turn = player_data["acceptable_energy_level_at_turn"].tolist()
    
    NUM_TURNS_TO_WAIT_BEFORE_SACRIFICING = 10
    
    if player.incapacitated:
        return []

    sacrifices = sacrifice(community.members, community.tasks)
    SACRIFICE_TIME = len(community.members) // 2

    if len(community.tasks) < SACRIFICE_TIME and sacrifices:
        to_be_sacrificed = find_weakest_agents(community.members, len(sacrifices))
        if player.id in to_be_sacrificed: 
            return sacrifices 

    return tasks_we_can_complete_alone(player, player.abilities, community.tasks)
    

def get_acceptable_energy_level(tasks_at_turn: list[int]) -> int:
    
    if len(tasks_at_turn) >= 10 and all(
        x == tasks_at_turn[-1] for x in tasks_at_turn[-10:]
    ):
        acceptable_energy_level = LOW_ENERGY_LEVEL
    else:
        acceptable_energy_level = NORMAL_ENERGY_LEVEL
    
    return acceptable_energy_level

    
def tasks_we_can_complete_alone(player, our_abilities, tasks) -> list[int]:
    alone_tasks = []
    our_abilities_np = np.array(our_abilities)

    for task_id, task in enumerate(tasks):
        task_np = np.array(task)
        diff = our_abilities_np - task_np
        negative_sum = np.sum(diff[diff < 0])

        if player.energy + negative_sum > 0:
            alone_tasks.append(task_id)
    return alone_tasks


def find_pairs(player: Member, tasks, members, acceptable_energy_level) -> list[int]:
    # [Task_ID, other_player_id ]
    task_player_pairs = []
    our_abilities_np = np.array(player.abilities)
    our_id = player.id
    for task_id, task in enumerate(tasks):
        task_np = np.array(task)

        for other_person in members:
            if other_person.id == our_id:
                continue

            others_abilities_np = np.array(other_person.abilities)
            combined_abilties = np.maximum(our_abilities_np, others_abilities_np)

            diff = combined_abilties - task_np
            negative_sum = np.sum(diff[diff < 0])
            player_above_acceptable_energy = (
                player.energy + negative_sum > acceptable_energy_level
            )
            other_person_above_acceptable_energy = (
                other_person.energy + negative_sum > acceptable_energy_level
            )
            # print(negative_sum)
            if player_above_acceptable_energy and other_person_above_acceptable_energy:
                task_player_pairs.append([task_id, other_person.id])
            if len(task_player_pairs) >= 8:
                return task_player_pairs

    return task_player_pairs


# For example, suppose n=4 and that an individual has skill levels (8,6,4,2), and that the task has difficulty vector (5,5,5,5). Then the individual would use up 1+3=4 units of energy to perform the task.


def find_weakest_agents(members: List[Member], n: int):
    agents = [(id, sum(member.abilties)) for member in members]
    n_weakest_agents = sorted(agents, key=lambda x: x[1])[:n]
    return n_weakest_agents

"""
sacrifice(members, tasks)
- identifies if there are tasks that require sacrificing members
- returns a list of tasks that require sacrifices
"""
def sacrifice(members: List[Member], tasks: List) -> List[int]:
    exhausting_tasks = []
    for i in range(len(tasks)):
        task = tasks[i]
        task_np = np.array(task)
        not_exhausting = False
        for member in members:
            abilities_np = np.array(member.abilities)
            diff = abilities_np - task_np
            # if one member can complete it themselves without being sacrificed
            if np.sum(diff[diff < 0]) < (MAX_ENERGY_LEVEL - EXHAUSTED_ENERGY_LEVEL):
                not_exhausting = True
            # else, check pairs 
            else:
                for other in members:
                    other_np = np.array(other.abilities)
                    combined_np = np.maximum(abilities_np, other_np)
                    diff = combined_np - task_np
                    if np.sum(diff[diff < 0]) < (MAX_ENERGY_LEVEL - EXHAUSTED_ENERGY_LEVEL) * 2:
                        not_exhausting = True
                        break
            if not_exhausting:
                break
        if not not_exhausting:
            exhausting_tasks.append(i)
    return exhausting_tasks

"""
non_solo_tasks(community: Community)
- identifies which tasks can be completed with no/little energy depletion by one member of society
- returns a list of all indices of tasks that should (and can) be completed by pairs
"""
# def non_solo_tasks(community: Community):
#     tasks = []

# def all_energized(members: List[Member]) -> bool:
#     for member in members:
#         if not member.incapacitated and member.energy < MAX_ENERGY_LEVEL:
#             return False
#     return True

    
"""
original sacrificing strategy in phase II:
if len(tasks_at_turn) >= START_SACRIFICING_YEAR and all(
        x == LOW_ENERGY_LEVEL for x in acceptable_energy_level_at_turn[-NUM_TURNS_TO_WAIT_BEFORE_SACRIFICING:]
    ):
        # SACRIFICE THE ELDERS / CHILDREN
        elders_or_children = find_weakest_agents(community.members, 3)
        print(f"elders_or_children: {elders_or_children}")
        if player.id in elders_or_children:
            return [task_id for (task_id, _) in community.tasks]
        else:
            return []
"""

# Task 7: (Do later...)
# We could just play submissive.... 
# We see if we are playing with a bunch of other teams.
# Just volenteer for every task with every player from every other team
# - We don't something weird / worse than them. 
# - Reduces the complexity of the situation.