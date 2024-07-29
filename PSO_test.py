import numpy as np
from PSO import PSO
def f(x,y):
    "Objective function"
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)

pso = PSO(2,1,10,1000)
iterations = 1000
solutions = pso.generate_start_solution()
for i in range(iterations):
    scores = []
    for solution in solutions:
        score = f(solution[0],solution[1])
        scores.append(score)
    solutions = pso.move_one_step(scores)
print(pso.best_particle_of_all)



