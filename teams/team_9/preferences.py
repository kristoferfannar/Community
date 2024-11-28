def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player.'''
    preferences = []

    task_avg_difficulty = sum(sum(task) for task in community.tasks) / len(community.tasks)
    player_avg_skill = sum(sum(member.abilities) for member in community.members) / len(community.members)

    primary_energy_limit = 0  # Allow energy to drop but not too far into the negatives

    # Calculate the ratio of difficulty to skill
    difficulty_ratio = task_avg_difficulty / max(player_avg_skill, 1e-6)  # Avoid division by zero

    # Map difficulty ratio to the range [-10, primary_energy_limit]
    if difficulty_ratio <= 1:
        secondary_energy_limit = primary_energy_limit  # Tasks are manageable
    else:
        secondary_energy_limit = primary_energy_limit - (difficulty_ratio - 1) * 10
        secondary_energy_limit = max(secondary_energy_limit, -10)  # Ensure it doesn't drop below -10

    # Sort tasks by total difficulty (descending)
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    for task_id, task in sorted_tasks:
        best_partner = None
        best_remaining_energy = -float('inf')  # Track the best post-task energy state

        # Check if the player can complete the task alone
        if all(task[i] <= player.abilities[i] for i in range(len(task))):
            continue  # Skip partnering for tasks the player can complete alone

        for partner in community.members:
            if partner.id == player.id or partner.incapacitated:
                continue  # Skip self or incapacitated players

            # Check if the partner can complete the task alone
            if all(task[i] <= partner.abilities[i] for i in range(len(task))):
                continue  # Skip partnering for tasks the partner can complete alone

            # Calculate energy cost for both players
            energy_cost = sum(max(task[i] - max(player.abilities[i], partner.abilities[i]), 0) for i in range(len(task))) / 2
            player_remaining_energy = player.energy - energy_cost
            partner_remaining_energy = partner.energy - energy_cost

            # Allow partnering even if energy dips below the primary limit
            if (
                player_remaining_energy > secondary_energy_limit
                and partner_remaining_energy > secondary_energy_limit
            ):
                # Choose the partner that maximizes the minimum remaining energy
                if min(player_remaining_energy, partner_remaining_energy) > best_remaining_energy:
                    best_partner = partner.id
                    best_remaining_energy = min(player_remaining_energy, partner_remaining_energy)

        # Add the task and partner to preferences if a valid partner is found
        if best_partner is not None:
            preferences.append([task_id, best_partner])

    return preferences


def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually.'''
    preferences = []

    task_avg_difficulty = sum(sum(task) for task in community.tasks) / max(len(community.tasks), 1e-6) 
    player_avg_skill = sum(sum(member.abilities) for member in community.members) / max(len(community.members), 1e-6)

    primary_energy_limit = 0  # Allow energy to drop but not too far into the negatives

    # Calculate the ratio of difficulty to skill
    difficulty_ratio = task_avg_difficulty / max(player_avg_skill, 1e-6)  # Avoid division by zero

    # Map difficulty ratio to the range [-10, primary_energy_limit]
    if difficulty_ratio <= 1:
        secondary_energy_limit = primary_energy_limit  # Tasks are manageable
    else:
        secondary_energy_limit = primary_energy_limit - (difficulty_ratio - 1) * 10
        secondary_energy_limit = max(secondary_energy_limit, -10)  # Ensure it doesn't drop below -10

    # Evaluate tasks for individual completion
    for task_id, task in enumerate(community.tasks):
        energy_cost = sum(max(task[i] - player.abilities[i], 0) for i in range(len(task)))
        remaining_energy = player.energy - energy_cost

        # Consider tasks that leave the player with energy above the secondary limit
        if remaining_energy > secondary_energy_limit:
            preferences.append((task_id, energy_cost, remaining_energy))

    # Sort tasks by a combination of low energy cost and high remaining energy
    preferences.sort(key=lambda x: (x[1], -x[2]))  # Sort by energy cost, then remaining energy

    # Return task IDs in preferred order
    return [task_id for task_id, _, _ in preferences]

'''
Prevent overlap in volunteering
Avoid volunteering to partner on easy tasks
Player energy can fall below -10 but they become incapacitated
'''