from community import Member, Community


def player_score(community: Community) -> list[int]:
    members = {member.id: 0 for member in community.members}

    for member in community.members:
        for task in community.tasks:
            cost = sum(
                max(task[i] - member.abilities[i], 0)
                for i in range(len(member.abilities))
            )
            members[member.id] += cost

    return sorted(members, key=members.get)


def calculate_minimum_delta_individual(player: Member, community: Community):
    num_abilities = len(player.abilities)

    minimum_delta = float("inf")
    best_task = -1

    for task_index, task in enumerate(community.tasks):
        cost = sum(max(task[i] - player.abilities[i], 0) for i in range(num_abilities))
        waste = (
            sum(max(player.abilities[i] - task[i], 0) for i in range(num_abilities))
            * 0.5
        )

        if minimum_delta >= cost + waste:
            minimum_delta = cost + waste
            best_task = task_index

    return (best_task, minimum_delta)


# Delta = min(energy + waste)
def calculate_minimum_delta_pair(
    player: Member, best_partner: int, community: Community
):
    members = {member.id: member for member in community.members}
    partner = members[best_partner]
    num_abilities = len(player.abilities)

    minimum_delta = float("inf")
    best_task = 0

    for task_index, task in enumerate(community.tasks):
        pair_cost = (
            sum(
                max(task[i] - max(player.abilities[i], partner.abilities[i]), 0)
                for i in range(num_abilities)
            )
            / 2
        )
        pair_waste = (
            sum(
                max(max(player.abilities[i], partner.abilities[i]) - task[i], 0)
                for i in range(num_abilities)
            )
            * 0.5
        )

        if minimum_delta >= pair_cost + pair_waste:
            minimum_delta = pair_cost + pair_waste
            best_task = task_index

    return (best_task, minimum_delta)


def phaseIpreferences(player: Member, community: Community, global_random):
    """Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id
    """
    list_choices = []

    if player.energy < 2:
        return list_choices

    members = player_score(community)
    index = members.index(player.id)

    best_partner = members[len(members) - 1 - index]
    best_task, pair_delta = calculate_minimum_delta_pair(
        player, best_partner, community
    )

    player_cost = sum(
        max(community.tasks[best_task][i] - player.abilities[i], 0)
        for i in range(len(player.abilities))
    )
    player_waste = sum(
        max(player.abilities[i] - community.tasks[best_task][i], 0)
        for i in range(len(player.abilities))
    )

    player_delta = player_cost + player_waste

    if pair_delta < player_delta * 1.5:
        list_choices.append([best_task, best_partner])

    return list_choices


def phaseIIpreferences(player, community, global_random):
    """Return a list of tasks for the particular player to do individually"""
    bids = []
    num_abilities = len(player.abilities)
    best_task, _ = calculate_minimum_delta_individual(player, community)

    if best_task == -1:
        return []

    energy_cost = sum(
        max(community.tasks[best_task][j] - player.abilities[j], 0)
        for j in range(num_abilities)
    )

    if energy_cost <= player.energy:
        bids.append(best_task)

    return bids
