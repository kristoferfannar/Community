import numpy as np
import scipy.optimize as opt
import random
import math
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from teams.team_5.Hivemind import Hivemind
import sys

# List of things to implement (Feel free to add to it)
#TODO: Finding the weakest players (Dummy players only? Or we include weak players too)
#TODO: Determine when a player should rest
#TODO: Minimize the number of pairs a player volunteers for (Currently we are volunteering for a lot)

def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''
    # print("THIS IS THE LIST OF TASKS: ", community.tasks)
    exhausted = True if player.energy < 0 else False # Am I exhuasted?
    spent_energy = 0
    list_choices = []
    # # if player.energy < -10:
    # #     return list_choices 
    
    # cost0_tasks = set()

    # #If any member can complete the task using 0 energy, then exclude it from our list
    # for i, task in enumerate(community.tasks):
    #     for member in community.members:
    #         energy_cost = sum([max(task[j] - member.abilities[j], 0) for j in range(len(member.abilities))])

    #         if energy_cost == 0:
    #             cost0_tasks.add(i)
    #             break
    # print("THESE ARE THE FREE TASKS: ", cost0_tasks)

    # # We're prioritizing tasks based on energy cost
    # task_priorities = sorted(
    #     [(i, sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))]))
    #      for i, task in enumerate(community.tasks)],
    #     key=lambda x: x[1]
    # )

    # exhausted_penalty = 0.5
    # # Finding compatible partners for top priority tasks
    # for task_id, _ in task_priorities:
    #     if task_id in cost0_tasks:
    #         continue

    #     best_partner = None
    #     min_cost = float('inf')
    #     for partner in community.members:
    #         if partner.id == player.id:
    #             continue
    #         combined_abilities = [max(player.abilities[j],  partner.abilities[j]) for j in range(len(player.abilities))]
    #         energy_cost = round((sum([max(community.tasks[task_id][j] - combined_abilities[j], 0) for j in range(len(player.abilities))]))/2, 2)
    #         added_cost = exhausted_penalty if partner.energy < 0 else 0 #We prioritize active workers instead of exhasuted workers

    #         comparison_metric = energy_cost + added_cost

    #         if comparison_metric < min_cost and (partner.energy - energy_cost) > -10 and (player.energy - energy_cost) > -10: # Make sure ourself and our partner will not become incapacitated after performing the task
    #             min_cost = comparison_metric
    #             best_partner = partner.id
    #             task_energy = energy_cost

    #     spent_energy += task_energy
        
    #     if (player.energy - spent_energy) <= -10:
    #         print(f"Member {player.id} has too much work load, ending partner search now.")
    #         print(f"Member {player.id}'s partner choices, ", list_choices)
    #         return list_choices

    #     if best_partner is not None:
    #         list_choices.append([task_id, best_partner])
        
    # print(f"Member {player.id}'s partner choices, ", list_choices)
        
    return list_choices
          
       
def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    # global hivemind

    # if hivemind and hivemind.prev_task == len(community.tasks):
    #     bids = []
    #     bids.append(hivemind.phase2_optimal_pair[player.id])
    #     return bids

    bids = []
    
    surviving_workers = []

    num_of_workers = len(community.members)
    num_of_tasks = len(community.tasks)
    num_of_abilities = len(player.abilities)

    # Find out the remaining workers
    for i in range(num_of_workers):
        if community.members[i].energy > -10:
            print(f"I CAN WORK, Says: {community.members[i].id}")
            surviving_workers.append(community.members[i])

    num_of_workers = len(surviving_workers) 

    cost_matrix  = [[np.inf for m in range(num_of_tasks)] for n in range(num_of_workers)] # n x m
    assert(len(cost_matrix) == num_of_workers and len(cost_matrix[0]) == num_of_tasks) # matrix check

    too_good = 1/((num_of_abilities * 10) + 1)
    tired_param = 2
    exhausted_param = 100000

    # Fill in matrix with value
    #TODO: Let the worst worker to be sacrificed first.
    for i in range(num_of_workers):
        for j in range(num_of_tasks):
            cost_matrix[i][j] = sum([max(community.tasks[j][k] - surviving_workers[i].abilities[k], 0) for k in range(num_of_abilities)])
            if (surviving_workers[i].energy - cost_matrix[i][j]) <= -10: # Worker will be sacrificed, if need be.
                cost_matrix[i][j] = exhausted_param
            elif (surviving_workers[i].energy - cost_matrix[i][j]) < 0:
                cost_matrix[i][j] *= tired_param
            elif cost_matrix[i][j] == 0:
                cost_matrix[i][j] += too_good * (sum([surviving_workers[i].abilities[k] - community.tasks[j][k] for k in range(num_of_abilities)]))

    # print("COST MATRIX:")
    # print(cost_matrix)

    # Solver
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # Variables
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if worker i is assigned to task j.
    x = {}
    for i in range(num_of_workers):
        for j in range(num_of_tasks):
            x[i, j] = solver.IntVar(0, 1, "")

    # Constraints
    if num_of_tasks > num_of_workers:
        # Each worker is assigned to at EXACTLY 1 task.
        for i in range(num_of_workers):
            solver.Add(solver.Sum([x[i, j] for j in range(num_of_tasks)]) == 1)

        # Each task is assigned to ONE OR LESS worker.
        for j in range(num_of_tasks):
            solver.Add(solver.Sum([x[i, j] for i in range(num_of_workers)]) <= 1)
    else:
        # Each worker is assigned to ONE OR LESS task.
        for i in range(num_of_workers):
            solver.Add(solver.Sum([x[i, j] for j in range(num_of_tasks)]) <= 1)

        # Each task is assigned to exactly worker.
        for j in range(num_of_tasks):
            solver.Add(solver.Sum([x[i, j] for i in range(num_of_workers)]) == 1)

    objective_terms = []
    for i in range(num_of_workers):
        for j in range(num_of_tasks):
            objective_terms.append(cost_matrix[i][j] * x[i, j])

    solver.Minimize(solver.Sum(objective_terms))

    # Solve
    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    phase2_optimal_pair = dict()

    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Total cost = {solver.Objective().Value()}\n")
        for i in range(num_of_workers):
            for j in range(num_of_tasks):
                # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                if x[i, j].solution_value() > 0.5:
                    print(f"Worker {i} assigned to task {j}." + f" Cost: {cost_matrix[i][j]}")
                    phase2_optimal_pair[surviving_workers[i].id] = j
    else:
        print("No solution found.")
 
    hivemind = Hivemind(phase2_optimal_pair=phase2_optimal_pair, prev_task=len(community.tasks))
    if player.id in hivemind.phase2_optimal_pair:
        bids.append(hivemind.phase2_optimal_pair[player.id]) 

    return bids
