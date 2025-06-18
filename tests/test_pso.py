from pso import (
    create_particles_uniform_distribution,
    create_particles_grid,
    initialize_velocities_uniform,
    initialize_velocities_null,
    Domain,
    Point,
    Particle,
    pso,
)

import numpy as np
import pytest
from typing import List, Callable


@pytest.fixture
def domain():
    return Domain(Point([0, 0]), Point([1, 1]))


@pytest.fixture
def num_particles():
    return 36


@pytest.fixture
def uniform_particles(domain, num_particles):
    return create_particles_uniform_distribution(domain, num_particles)


@pytest.fixture
def grid_particles(domain, num_particles):
    return create_particles_grid(domain, num_particles)


@pytest.fixture
def f(domain, num_particles):
    def waves_and_parabola(x: Point):
        if (
            x.x1 < domain.upper_corner.x1
            and x.x2 < domain.upper_corner.x2
            and x.x1 > domain.lower_corner.x1
            and x.x2 > domain.lower_corner.x2
        ):
            waves = np.cos(x.x1 * 2 * np.pi * 3) + np.cos(x.x2 * 2 * np.pi * 3)
            parabola = (
                x.x1 - (domain.upper_corner.x1 - domain.lower_corner.x1) / 2
            ) ** 2 + (x.x2 - (domain.upper_corner.x2 - domain.lower_corner.x2) / 2) ** 2
            combined = waves + parabola * 10
            return combined
        else:
            return np.inf

    return waves_and_parabola


def test_point():
    p = Point([10, 15])
    assert p.x1 == 10
    assert p.x2 == 15
    assert (p.data == np.array([10, 15])).all()


def test_create_particles_uniform_distribution(
    domain: Domain, uniform_particles: List[Particle]
):

    # import matplotlib.pyplot as plt

    # plt.scatter([p.x.x1 for p in particles], [p.x.x2 for p in particles])
    # plt.show()

    assert type(uniform_particles[0]) == Particle

    for particle in uniform_particles:
        assert particle.x.x1 <= domain.upper_corner.x1
        assert particle.x.x2 <= domain.upper_corner.x2
        assert particle.x.x1 >= domain.lower_corner.x1
        assert particle.x.x2 >= domain.lower_corner.x2


def test_create_particles_grid(domain: Domain, grid_particles: List[Particle]):
    # import matplotlib.pyplot as plt

    # plt.scatter([p.x.x1 for p in particles], [p.x.x2 for p in particles])
    # plt.show()
    assert type(grid_particles[0]) == Particle

    for particle in grid_particles:
        assert particle.x.x2 <= domain.upper_corner.x1
        assert particle.x.x2 <= domain.upper_corner.x2
        assert particle.x.x1 >= domain.lower_corner.x1
        assert particle.x.x2 >= domain.lower_corner.x2


def test_pso_uniform(f: Callable, domain: Domain, grid_particles: List[Particle]):

    initialize_velocities_uniform(domain, grid_particles)

    result = pso(
        f,
        grid_particles,
        domain,
        0.5,
        0.5,
        0.2,
        100,
        make_gif=True,
        gif_save_path="pso-uniform.gif",
        gif_frame_duration_ms=100,
    )
    assert result.x1 == pytest.approx(0.5, 0.001)
    assert result.x2 == pytest.approx(0.5, 0.001)


def test_pso_grid(f: Callable, domain: Domain, grid_particles: List[Particle]):

    initialize_velocities_null(domain, grid_particles)

    result = pso(
        f,
        grid_particles,
        domain,
        0.5,
        0.5,
        0.2,
        100,
        make_gif=True,
        gif_save_path="pso-grid.gif",
        gif_frame_duration_ms=100,
    )
    assert result.x1 == pytest.approx(0.5, 0.001)
    assert result.x2 == pytest.approx(0.5, 0.001)
