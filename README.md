# Particle Swarm Optimization

This project implements a 2d [particle swarm optmization algorithm (PSO)](https://en.wikipedia.org/wiki/Particle_swarm_optimization#Algorithm).

Result with grid-like initial positions:
![](img/pso-grid.gif)

Result with uniform random distribution in initial positions:
![](img/pso-uniform.gif)

## Installation

```
git clone https://github.com/brunomariz/pso.git
cd pso

python3 -m venv .venv
source .venv/bin/activate

pip install poetry
poetry lock
poetry install

pytest
```
