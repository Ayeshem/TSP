from abc import ABC, abstractmethod
from typing import List
from scipy.spatial import distance
import numpy as np
import random
import math
import time
import asyncio
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO)

DEFAULT_SOLVER_TIMEOUT = 20

def valid_problem(problem):
    """Check if the problem is valid (you can modify this logic according to your needs)."""
    return problem.n_nodes > 0

def normalize_coordinates(nodes):
    """Normalize coordinates as needed for the TSP problem."""
    nodes = np.array(nodes)
    min_vals = np.min(nodes, axis=0)
    max_vals = np.max(nodes, axis=0)
    return (nodes - min_vals) / (max_vals - min_vals)

class GraphProblem:
    """Mockup class to simulate the TSP problem input."""
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.nodes = np.random.rand(n_nodes, 2)
        self.problem_type = "TSP"

class BaseSolver(ABC):
    def __init__(self, problem_types: List[GraphProblem]):
        self.problem_types = [problem.problem_type for problem in problem_types]
        self.future_tracker = {}

    @abstractmethod
    async def solve(self, formatted_problem, future_id, *args, **kwargs) -> List[int]:
        """Abstract method for solving the TSP problem."""
        pass

    @abstractmethod
    def problem_transformations(self, problem: GraphProblem):
        """Transform the problem as required by the solver."""
        pass

    def is_valid_problem(self, problem):
        """Check if the solver can solve the given problem."""
        return valid_problem(problem) and problem.problem_type in self.problem_types

    async def solve_problem(self, problem: GraphProblem, timeout: int = DEFAULT_SOLVER_TIMEOUT):
        """Main method to solve the problem, including transformations and timeout."""
        if self.is_valid_problem(problem):
            future_id = id(problem)
            self.future_tracker[future_id] = False

            transformed_problem = self.problem_transformations(problem)

            loop = asyncio.get_running_loop()
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = loop.run_in_executor(executor, lambda: asyncio.run(self.solve(transformed_problem, future_id)))
                try:
                    result = await asyncio.wait_for(future, timeout)
                    return result
                except asyncio.TimeoutError:
                    logging.error(f"Task {future_id} timed out after: {time.time() - start_time}, with timeout set to {timeout}")
                    self.future_tracker[future_id] = True
                    return False
                except Exception as exc:
                    logging.error(f"Task generated an exception: {exc}")
                    return False
        else:
            logging.error(f"current solver: {self.__class__.__name__} cannot handle received problem: {problem.problem_type}")
            return False

class SimulatedAnnealingSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2)], initial_temp=1000, cooling_rate=0.995, max_iterations=100000):
        super().__init__(problem_types=problem_types)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations

    def calculate_distance(self, tour, distance_matrix):
        return sum(distance_matrix[tour[i - 1]][tour[i]] for i in range(len(tour)))

    def simulated_annealing(self, distance_matrix):
        n = len(distance_matrix)

        # Start with a random tour
        current_tour = list(range(n))
        random.shuffle(current_tour)
        current_distance = self.calculate_distance(current_tour, distance_matrix)

        best_tour = list(current_tour)
        best_distance = current_distance

        temperature = self.initial_temp

        for iteration in range(self.max_iterations):
            # Pick two cities to swap
            i, j = random.sample(range(n), 2)
            new_tour = list(current_tour)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            new_distance = self.calculate_distance(new_tour, distance_matrix)

            # Calculate the acceptance probability
            if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temperature):
                current_tour = new_tour
                current_distance = new_distance

                # Update the best found solution
                if current_distance < best_distance:
                    best_tour = current_tour
                    best_distance = current_distance

            # Cool down
            temperature *= self.cooling_rate

            # Early stopping if temperature is too low
            if temperature < 1e-10:
                break

        return best_tour, best_distance

    # Adding the method to calculate path cost
    def calculate_path_cost(self, tour, distance_matrix):
        """Calculate the total path cost based on the given tour and distance matrix."""
        path_cost = 0.0
        for i in range(len(tour)):
            from_node = tour[i - 1]
            to_node = tour[i]
            path_cost += distance_matrix[from_node][to_node]
        return path_cost

    async def solve(self, formatted_problem, future_id: int, post_process: bool = False) -> List[int]:
        coordinates = formatted_problem
        size = len(coordinates)

        # Create a distance matrix based on the coordinates
        distance_matrix = distance.cdist(coordinates, coordinates, 'euclidean')

        best_tour, best_distance = self.simulated_annealing(distance_matrix)

        # Calculate path cost
        path_cost = self.calculate_path_cost(best_tour, distance_matrix)
        logging.info(f"Best tour found with distance: {best_distance}, Path Cost: {path_cost}")

        return best_tour

    def problem_transformations(self, problem: GraphProblem):
        # Normalize coordinates
        formatted_problem = normalize_coordinates(problem.nodes)
        return formatted_problem

if __name__ == '__main__':
    n_nodes = 250  # You can modify the number of nodes for testing
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = SimulatedAnnealingSolver(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"Solution: {route}")
    print(f"Time Taken for {n_nodes} Nodes: {time.time() - start_time}")
