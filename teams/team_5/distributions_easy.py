import random
import math
from scipy.stats import skewnorm
print("easy")
print("THIS IS THE RVS: ", skewnorm.rvs(a=5,loc=5, scale=5, size=1000))

#Negatively skewed normal distribution
def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    ability_dist=[]
    for i in range(num_abilities):
        rv =  skewnorm.rvs(a=5,loc=5, scale=5)
        rv_int = math.floor(rv)
        prowess = rv_int
        if rv_int > 10:
            prowess = 10
        elif rv_int < 5:
            prowess = 5
        ability_dist.append(prowess)

    return ability_dist

#Positively skewed normal distribution
def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    difficulties=[]
    for i in range(num_abilities):
        rv =  skewnorm.rvs(a=-4,loc=5, scale=5)
        rv_int = math.floor(rv)
        difficulty = rv_int
        if rv_int > 7:
            difficulty = 7
        elif rv_int < 0:
            difficulty = 0

        difficulties.append(difficulty)

    return difficulties