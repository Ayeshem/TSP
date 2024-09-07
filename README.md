# Travelling Salesman Problem (TSP) Solvers

This project implements various algorithms to solve the Travelling Salesman Problem (TSP) and compares their performance. The focus is on improving the Simulated Annealing (SA) algorithm to achieve optimal results within a specified time.

# Algorithms Implemented

- **Simulated Annealing Solver:** Achieved a mean cost of 1.4 with an average processing time of approximately 3.92 seconds.
- **Christofides Solver:** Provides a baseline for comparison.
- **Baseline Solver:** Simple heuristic approach for comparison.
- **MIN Solver:** Another heuristic for performance comparison.

# Installation and Setup

1. **Clone the Repository:**
   git clone https://github.com/nguyendunghy/test-travelling-salesman-problem.git
   

2. **Navigate to the Project Directory:**
   cd test-travelling-salesman-problem
   

3. **Set Up a Python Virtual Environment:**
   python3 -m venv venv
   source venv/bin/activate

4. **Install Required Packages:**
   pip install -r requirements.txt

# Usage

To run the solvers and evaluate their performance, use the following command:
python test.py
This will execute the TSP solvers and provide performance comparisons.
