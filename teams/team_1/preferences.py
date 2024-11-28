def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''

    all_players = set(range(len(community.members)))
    num_abilities = len(player.abilities)

    # Sort tasks in order of descending difficulty
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    if not isValidForPartnership(player, community, None):
        return []
    
    # Form partnerships based on the tasks
    partner_choices = []
    
    for task_id, task in sorted_tasks:
        best_partner_id = None
        best_final_energy = -float("inf")
        for partner_id in all_players:
            if partner_id == player.id:
                continue
        
            partner = community.members[partner_id]
            if isValidForPartnership(partner, community, task):
                joint_abilities = [max(a1, a2) for a1, a2 in zip(player.abilities, partner.abilities)]
                energy_cost = sum([max(task[j] - joint_abilities[j], 0) for j in range(num_abilities)]) / 2
                
                player_final_energy = player.energy - energy_cost
                partner_final_energy = partner.energy - energy_cost
                lowest_final_energy = min(player_final_energy, partner_final_energy)

                # Skip pairing if it will tire out either of the two players too much
                if lowest_final_energy < -5:
                    continue

                # Keep track of best partnership to preserve energy
                if lowest_final_energy > best_final_energy:
                    best_final_energy = lowest_final_energy
                    best_partner_id = partner.id
        
        if best_partner_id:
            partner_choices.append([task_id, best_partner_id])

    return partner_choices

def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    bids = []

    for task_id, task in enumerate(community.tasks):
        energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))])
        # Volunteering logic
        if player.energy - energy_cost >= 0:
            bids.append(task_id)

    # Handle extremely difficult tasks
    diff_tasks = findDifficultTasks(community)
    if len(diff_tasks) > 0:
        for task_id, _ in diff_tasks:
            # attempt to execute a sacrifice for these tasks
            sacrificer_id = executeSacrifice(community, community.tasks[task_id], len(player.abilities))
            if sacrificer_id is not None:
                print(f"Sacrificed player {sacrificer_id} to complete task {task_id}.")
                bids.append(task_id)

    return bids

def findDifficultTasks(community):
    """
    Identifies difficult tasks that cannot be completed by any individual player
    or through partnerships without causing energy exhaustion.
    """
    difficult_tasks = []
    num_abilities = len(community.members[0].abilities)

    for task_id, task in enumerate(community.tasks):
        # Check if any single player can complete the task
        individual_fail = all(
            sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)]) > player.energy
            for player in community.members
        )

        # Check if any partnership can complete the task
        partnership_fail = True
        for i, player1 in enumerate(community.members):
            for j, player2 in enumerate(community.members):
                if i >= j:
                    continue
                
                joint_abilities = [max(player1.abilities[k], player2.abilities[k]) for k in range(num_abilities)]
                energy_cost = sum([max(task[l] - joint_abilities[l], 0) for l in range(num_abilities)]) / 2

                # Check if both players can handle the energy cost
                if player1.energy - energy_cost >= -10 and player2.energy - energy_cost >= -10:
                    partnership_fail = False
                    break
            if not partnership_fail:
                break

        # ff neither individuals nor partnerships can complete the task, it's "difficult"
        if individual_fail and partnership_fail:
            max_energy_loss = max([
                sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])
                for player in community.members
            ])
            difficult_tasks.append((task_id, max_energy_loss))

    return difficult_tasks

def isValidForPartnership(player, community, task):
    if player.energy <= 0:
        return False
    
    ### THIS CODE BELOW IS LIKE THE STRONG PLAYER LOGIC, DIDN'T HELP....

    # if task:
    #     energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))])
    #     if energy_cost == 0:
    #         return False
    # else:
    #     for task in community.tasks:
    #         energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))])
    #         if energy_cost == 0:
    #             return False

    # sacrificing strategy
    
    return True


def executeSacrifice(community, task, num_abilities):
    """
    Execute a sacrifice to complete a difficult task.
    - identifies the player with the lowest energy who can complete the task
    - ensures no other valid partnerships or individual efforts can resolve the task
    - deducts the energy cost and marks the sacrificer as incapacitated
    """
    # check if the task requires a sacrifice (this might be redundant since this method is only called on difficult tasks anyway)
    individual_fail = all(
        sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)]) > player.energy
        for player in community.members
    )

    partnership_fail = True
    for i, player1 in enumerate(community.members):
        for j, player2 in enumerate(community.members):
            if i >= j:
                continue
            
            joint_abilities = [max(player1.abilities[k], player2.abilities[k]) for k in range(num_abilities)]
            energy_cost = sum([max(task[l] - joint_abilities[l], 0) for l in range(num_abilities)]) / 2

            if player1.energy - energy_cost >= -10 and player2.energy - energy_cost >= -10:
                partnership_fail = False
                break
        if not partnership_fail:
            break

    # if no valid partnerships or individual efforts, proceed with sacrifice
    if individual_fail and partnership_fail:
        # identify sacrificer (player with the lowest energy)
        sacrificer = None
        lowest_energy = float('inf')

        for player in community.members:
            energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])
            if energy_cost <= player.energy and player.energy < lowest_energy:
                lowest_energy = player.energy
                sacrificer = player

        # execute the sacrifice
        if sacrificer:
            energy_cost = sum([max(task[j] - sacrificer.abilities[j], 0) for j in range(num_abilities)])
            sacrificer.energy -= energy_cost
            print(f"Player {sacrificer.id} sacrificed for task, remaining energy: {sacrificer.energy}")
            
            if sacrificer.energy <= -10:
                sacrificer.incapacitated = True
                print(f"Player {sacrificer.id} is now incapacitated.")
            
            return sacrificer.id
    # no sacrifice made
    return None  
