import sys
sys.path.append('../')
from platypus import NSGAII, DTLZ2, ProcessPoolEvaluator
import matplotlib.pyplot as plt

# simulate an computationally expensive problem
class DTLZ2_Slow(DTLZ2):

    def evaluate(self, solution):
        sum(range(100000))
        super().evaluate(solution)

if __name__ == "__main__":
    problem = DTLZ2_Slow()

    # supply an evaluator to run in parallel
    with ProcessPoolEvaluator(4) as evaluator:
        algorithm = NSGAII(problem, evaluator=evaluator)
        algorithm.run(10000)

    # display the results
    for solution in algorithm.result:
        print(solution.objectives)

    fig = plt.figure()
    plt.scatter([s.objectives[0] for s in algorithm.result], 
        [s.objectives[1] for s in algorithm.result])
    plt.show()
