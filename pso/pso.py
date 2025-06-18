import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil


@dataclass
class Point:
    data: np.ndarray

    def __post_init__(self):
        self.data = np.array(self.data)

    @property
    def x1(self):
        return self.data[0]

    @property
    def x2(self):
        return self.data[1]


@dataclass
class Particle:
    x: Point
    v: Point = field(init=False, default_factory=lambda: Point([0, 0]))
    p: Point = field(init=False, default_factory=lambda: Point([0, 0]))

    def __post_init__(self):
        self.p.data = self.x.data.copy()


@dataclass
class Domain:
    lower_corner: Point
    upper_corner: Point


def create_particles_uniform_distribution(domain: Domain, num_particles: int):
    random_x1s = np.random.uniform(
        domain.lower_corner.x1, domain.upper_corner.x1, num_particles
    )
    random_x2s = np.random.uniform(
        domain.lower_corner.x2, domain.upper_corner.x2, num_particles
    )
    particles = [Particle(Point([x1, x2])) for x1, x2 in zip(random_x1s, random_x2s)]

    return particles


def create_particles_grid(domain: Domain, num_particles: int):
    particles = [
        Particle(Point([x1, x2]))
        for x1 in np.linspace(
            domain.lower_corner.x1, domain.upper_corner.x1, int(np.sqrt(num_particles))
        )
        for x2 in np.linspace(
            domain.lower_corner.x2, domain.upper_corner.x2, int(np.sqrt(num_particles))
        )
    ]

    return particles


def initialize_velocities_uniform(domain: Domain, particles: List[Particle]):
    num_particles = len(particles)

    random_v1s = np.random.uniform(
        -np.abs(domain.upper_corner.x1 - domain.lower_corner.x1),
        np.abs(domain.upper_corner.x1 - domain.lower_corner.x1),
        num_particles,
    )
    random_v2s = np.random.uniform(
        -np.abs(domain.upper_corner.x2 - domain.lower_corner.x2),
        np.abs(domain.upper_corner.x2 - domain.lower_corner.x2),
        num_particles,
    )
    for i, particle in enumerate(particles):
        particle.v.data = np.array([random_v1s[i], random_v2s[i]])


def initialize_velocities_null(domain: Domain, particles: List[Particle]):
    for particle in particles:
        particle.v.data = np.array([0, 0])


def create_gif_from_figures(
    output_path="pso.gif",
    num_frames=10,
    duration_ms=1000,
    tmp_frames_dir: str = "./tmp-pso-frames",
):
    frames = []

    # Use the first frame to generate a fixed palette
    palette_source = Image.open(os.path.join(tmp_frames_dir, "fig0.png")).convert(
        "P", palette=Image.ADAPTIVE
    )

    for i in range(num_frames):
        frame_path = os.path.join(tmp_frames_dir, f"fig{i}.png")
        # Load RGB image
        frame = Image.open(frame_path).convert("RGB")
        # Quantize to the same palette as fig0
        frame = frame.quantize(palette=palette_source)
        frames.append(frame)

    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,  # milliseconds per frame
        loop=0,
        disposal=2,
    )

    # Remove temporary frames dir
    shutil.rmtree(tmp_frames_dir)


def plot_particles_with_velocities(
    domain: Domain,
    particles: List[Particle],
    iteration: int,
    swarms_best_position: Point,
    f: Callable[[Point], float],  # pass the function f here
    tmp_frames_dir: str = "./tmp-pso-frames",
):

    os.makedirs(tmp_frames_dir, exist_ok=True)

    # Generate overlay data from f(...) directly here:
    x = np.linspace(domain.lower_corner.x1, domain.upper_corner.x1, 100)
    y = np.linspace(domain.lower_corner.x2, domain.upper_corner.x2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Vectorize f if needed to speed up, but keep simple here:
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(Point([X[i, j], Y[i, j]]))

    fig, ax = plt.subplots(figsize=(6, 6))

    # Show function values as background heatmap with extent to match domain
    vmin = np.percentile(Z, 5)
    vmax = np.percentile(Z, 95)
    im = ax.imshow(
        Z,  # multiply by 100 to increase contrast
        origin="lower",
        extent=(
            domain.lower_corner.x1,
            domain.upper_corner.x1,
            domain.lower_corner.x2,
            domain.upper_corner.x2,
        ),
        cmap="gray",
        alpha=0.8,
        aspect="auto",
        vmin=vmin,  # Lower bound of colormap
        vmax=vmax,  # Upper bound of colormap
    )

    # Plot particle positions and velocities
    x_positions = [p.x.x1 for p in particles]
    y_positions = [p.x.x2 for p in particles]
    u_velocities = [p.v.x1 for p in particles]
    v_velocities = [p.v.x2 for p in particles]

    ax.quiver(
        x_positions,
        y_positions,
        u_velocities,
        v_velocities,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        alpha=0.8,
    )
    ax.scatter(
        [swarms_best_position.x1],
        [swarms_best_position.x2],
        color="red",
        label="Swarm's centroid",
        s=150,
        alpha=0.7,
        zorder=10,
        edgecolor="black",
        linewidth=1,
    )

    ax.scatter(x_positions, y_positions, color="blue", alpha=0.9, label="Particles")

    ax.set_xlim(domain.lower_corner.x1, domain.upper_corner.x1)
    ax.set_ylim(domain.lower_corner.x2, domain.upper_corner.x2)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"Particles and Velocities (Iteration {iteration})")
    ax.grid(True)
    ax.legend(loc="upper right")

    # plt.colorbar(im, ax=ax, label="f(x) value")

    fig.savefig(os.path.join(tmp_frames_dir, f"fig{iteration}.png"))
    plt.close(fig)
    plt.close()


def pso(
    f: Callable,
    particles: List[Particle],
    domain: Domain,
    w: float,
    phi_p: float,
    phi_g: float,
    num_iter: int,
    make_gif=False,
    gif_save_path="./pso.gif",
    gif_frame_duration_ms: int = 100,
):

    swarms_best_position = Point([0, 0])
    swarms_best_position.data = domain.lower_corner.data.copy()

    for i in range(num_iter):

        if make_gif:
            plot_particles_with_velocities(
                domain, particles, i, swarms_best_position, f
            )

        for particle in particles:
            rp = np.random.uniform(0, 1)
            rg = np.random.uniform(0, 1)

            particle.v.data = (
                w * particle.v.data
                + phi_p * rp * (particle.p.data - particle.x.data)
                + phi_g * rg * (swarms_best_position.data - particle.x.data)
            )
            particle.x.data += particle.v.data

            if f(particle.x) < f(particle.p):
                particle.p.data = particle.x.data.copy()
                if f(particle.p) < f(swarms_best_position):
                    swarms_best_position.data = particle.p.data.copy()

    create_gif_from_figures(
        gif_save_path, num_frames=num_iter, duration_ms=gif_frame_duration_ms
    )

    return swarms_best_position
