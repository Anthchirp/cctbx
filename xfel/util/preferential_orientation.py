from __future__ import division

import typing
from dataclasses import dataclass
import glob
from typing import List
import sys

from dxtbx.model import ExperimentList
from xfel.util.drift import params_from_phil, read_experiments

from mpl_toolkits.mplot3d import Axes3D  # noqa: required to use 3D axes
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy as sp


message = """
This utility tool aims to determinate, characterise, and quantify the degree
of preferential orientation in crystals. To this aim, it investigates
the distribution of vectors a, b, and c on a directions sphere in 3D.
The code assumes each set of vectors follows Wilson Distribution, and attempts
to model said distribution by fitting its parameter `mu` and `kappa`.

Wilson distribution describes a bimodal arrangement of points / unit vectors
on a sphere around a central axis called `mu`. The distribution is invariant
to any rotation around `mu` and inversion, and its exact type depends on the
concentration parameter `kappa`. For `kappa` > 0, the points are focused in
the polar region around +/- `mu`. In case of  `kappa` < 0, the points
concentrate mostly in a equatorial region far from `mu`. `kappa` close to 0
describes a distribution uniform on sphere: no preferential orientation. 

This code has been prepared using the following books & papers as references:
- http://palaeo.spb.ru/pmlibrary/pmbooks/mardia&jupp_2000.pdf, section 10.3.2
- https://www.sciencedirect.com/science/article/pii/S0047259X12002084, sect. 2
- https://www.tandfonline.com/doi/abs/10.1080/03610919308813139

This is a work in progress.
""".strip()


phil_scope_str = """
  input {
    glob = None
      .type = str
      .multiple = True
      .help = glob which matches all expt files to be investigated.
    exclude = None
      .type = str
      .multiple = True
      .help = glob which matches all expt files to be excluded from input.
  }
"""

cctbx_point_group_type = typing.Any


############################ ORIENTATION SCRAPPING ############################


class DirectSpaceVectors(np.ndarray):
  """Class responsible for scraping and storing vectors a, b, c from expts"""
  def __init__(self, shape, *args, **kwargs):
    super().__init__(shape, *args, **kwargs)
    if len(shape) < 3 or shape[0] != 3 or shape[2] != 3:
      msg = 'DirectSpaceVectors must be init with a 3xNx3 array of abc vectors'
      raise ValueError(msg)

  @classmethod
  def from_expts(cls, expts: ExperimentList) -> 'DirectSpaceVectors':
    """Extract N vectors a, b, c from N expts into a 3xNx3 ndarray, return"""
    abc = [e.crystal.get_real_space_vectors().as_numpy_array() for e in expts]
    return cls(np.stack(abc, axis=1).T)

  @classmethod
  def from_glob(cls, parameters) -> 'DirectSpaceVectors':
    """Read and return a Nx3x3 orientation matrix based on input parameters"""
    expt_paths = cls.locate_input_paths(parameters=parameters)
    expts = read_experiments(*expt_paths)
    return cls.from_expts(expts)

  @staticmethod
  def locate_input_paths(parameters) -> List[str]:
    """Return a list of expt paths in input.glob, but not in exclude"""
    input_paths, exclude_paths = [], []
    for ig in parameters.input.glob:
      input_paths.extend(glob.glob(ig))
    for ie in parameters.input.exclude:
      exclude_paths.extend(glob.glob(ie))
    return [it for it in input_paths if it not in exclude_paths]

  @property
  def a(self) -> np.ndarray:
    return self[0]

  @property
  def b(self) -> np.ndarray:
    return self[1]

  @property
  def c(self) -> np.ndarray:
    return self[2]


##################### PREFERENTIAL ORIENTATION CALCULATOR #####################


class SphericalDistribution:
  """General class for handling distribution of unit vectors in 3D"""
  E1 = np.array([1, 0, 0])
  E2 = np.array([0, 1, 0])
  E3 = np.array([0, 0, 1])

  def __init__(self):
    self.vectors: np.ndarray = None
    self.mu: np.ndarray = np.array([1, 0, 0])

  @staticmethod
  def normalized(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return `vectors` normalized using standard l2 norm along `axis` """
    l2 = np.atleast_1d(np.linalg.norm(vectors, 2, axis))
    l2[l2 == 0] = 1
    return vectors / np.expand_dims(l2, axis)

  @staticmethod
  def are_parallel(v: np.ndarray, w: np.ndarray, eps: float = 1e-8) -> bool:
    return abs(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))) < 1 - eps

  @property
  def mu_basis_vectors(self):
    """Basis vector of cartesian system in which e1 = mu; e2 & e3 arbitrary"""
    e1 = self.mu / np.linalg.norm(self.mu)
    e0 = self.E1 if not self.are_parallel(e1, self.E1) else self.E2
    e2 = np.cross(e1, e0)
    e3 = np.cross(e1, e2)
    return e1, e2, e3

  def mu_sph2cart(self, vectors: np.ndarray):
    """Convert spherical coordinates r, polar, azim to cartesian in mu basis"""
    r, polar, azim = np.hsplit(vectors, 3)
    e1, e2, e3 = self.mu_basis_vectors
    e1_component = e1 * np.cos(polar)
    e2_component = e2 * np.sin(polar) * np.cos(azim)
    e3_component = e3 * np.sin(polar) * np.sin(azim)
    return r * (e1_component + e2_component + e3_component)

  def apply_symmetry(self, symmetry: cctbx_point_group_type):
    """Apply all symmetry elements of a given point group to `self.vectors` in
    order to eliminate any bias coming from non-uniform orientation choice"""
    pass  # TODO


class WatsonDistribution(SphericalDistribution):
  """Class for holding, fitting, and generating Watson distribution.
  Description names reflect those used in respective references:
  - https://arxiv.org/pdf/1104.4422.pdf, page 3
  - http://palaeo.spb.ru/pmlibrary/pmbooks/mardia&jupp_2000.pdf, section 10.3.2
  - https://www.tandfonline.com/doi/abs/10.1080/03610919308813139"""
  def __init__(self, mu: np.ndarray = None, kappa: float = None) -> None:
    super().__init__()
    self.kappa: float = kappa
    self.mu: np.ndarray = mu
    self.nll: float = np.Infinity

  def __str__(self) -> str:
    return f'Watson Distribution around mu={self.mu} with kappa={self.kappa}'

  @classmethod
  def from_vectors(cls, vectors: np.ndarray) -> 'WatsonDistribution':
    """Define the distribution by fitting it to a list of vectors"""
    new = WatsonDistribution()
    new.fit(vectors=cls.normalized(vectors))
    return new

  @property
  def r_avg(self) -> float:
    """Length of the not-normalized mean direction of vectors, bar{R}"""
    return np.linalg.norm(self.x_avg)

  @property
  def x_avg(self) -> np.ndarray:
    """Sum of vectors divided by their count, bar{x}"""
    return np.sum(self.vectors, axis=0) / self.vectors.shape[0]

  @staticmethod
  def kummer_function(a: float, b: float, kappa: float) -> float:
    """Confluent hypergeometric function 1F1, a.k.a. Kummer function"""
    return sp.special.hyp1f1(a, b, kappa)

  @property
  def scatter_matrix(self) -> np.ndarray:
    """Scatter matrix of `vectors` distribution (9.2.10)"""
    return np.matmul(self.vectors.T, self.vectors) / len(self.vectors)

  def log_likelihood(self, mu: np.ndarray, kappa: float) -> float:
    """Log likelihood of given mu, kappa given current vectors. (10.3.30)"""
    t = self.scatter_matrix
    m = self.kummer_function(1/2, 3/2, kappa)
    return len(self.vectors) * (kappa * mu.T @ t @ mu - np.log(m))

  def nll_of_kappa(self, kappa: float, mu: np.ndarray) -> float:
    """Negative log likelihood of this Watson Distribution as a function
    of kappa, with `mu` and `vectors` fixed and given in `params`"""
    return -self.log_likelihood(mu=mu, kappa=kappa)

  def fit(self, vectors: np.ndarray) -> None:
    """Fit distribution to `vectors`, update `self.mu` and `self.kappa`"""
    self.vectors = vectors
    eig_val, eig_vec = np.linalg.eig(self.scatter_matrix)
    fitted = {'mu': np.array([1., 0., 0.]), 'kappa': 0., 'nll': np.inf}
    for eig_val, eig_vec in zip(eig_val, eig_vec.T):
        result = sp.optimize.minimize(self.nll_of_kappa, x0=0., args=eig_vec)
        if (nll := result['fun']) < fitted['nll']:
            fitted = {'mu': eig_vec, 'kappa': result['x'][0], 'nll': nll}
    self.kappa = fitted['kappa']
    self.mu = fitted['mu']
    self.nll = fitted['nll']

  def sample(self, n: int, seed: int = 42) -> np.ndarray:
    """Sample `n` vectors from self, based on doi 10.1080/03610919308813139"""
    if n < 0:
        return
    k = self.kappa
    rho = (4 * k) / (2 * k + 3 + ((2 * k + 3) ** 2 - 16 * k) ** 0.5)
    r = ((3 * rho) / (2 * k)) ** 3 * np.exp(-3 + 2 * k / rho)
    rng = np.random.default_rng(seed=seed)

    def cos2_of_polar_angle(_n: int) -> np.ndarray:
      u0 = rng.uniform(size=2*_n)
      u1 = rng.uniform(size=2*_n)
      s = u0 ** 2 / (1 - rho * (1 - u0 ** 2))
      v = (r * u1 ** 2) / (1 - rho * s) ** 3
      good_s = s[v <= np.exp(2 * k * s)]
      return good_s[:_n] if (lgs := len(good_s)) >= _n else \
          np.concatenate([good_s, cos2_of_polar_angle(_n-lgs)], axis=None)
    u2 = rng.uniform(size=n)
    theta = np.arccos(cos2_of_polar_angle(n) ** 0.5)
    phi = 4 * np.pi * u2
    theta[u2 < 0.5] = np.pi - theta[u2 < 0.5]
    phi[u2 >= 0.5] = 2 * np.pi * (2 * u2[u2 >= 0.5] - 1)
    self.vectors = self.mu_sph2cart(np.vstack([np.ones_like(theta), theta, phi]).T)


class PQRArray:
  """A collection of pseudo-vectors representing directions in direct space"""
  RADIUS = 5

  def __init__(self):
    self.pqr: np.ndarray = np.array([0, 0, 0])
    self.expand(around=np.array([0, 0, 0]))

  def expand(self, around: np.ndarray) -> None:
    """Generate new direction pseudo-vectors in a `RADIUS` around `around`."""
    p_range = np.arange(around[0] - self.RADIUS, around[0] + self.RADIUS + 1)
    q_range = np.arange(around[1] - self.RADIUS, around[1] + self.RADIUS + 1)
    r_range = np.arange(around[2] - self.RADIUS, around[2] + self.RADIUS + 1)
    pqr_mesh = np.meshgrid(p_range, q_range, r_range)
    pqr = np.column_stack([mesh_comp.ravel() for mesh_comp in pqr_mesh])
    pqr = pqr[np.linalg.norm(pqr, axis=1) <= self.RADIUS]
    p, q, r = pqr.T
    pqr = pqr[(p > 0) | ((p == 0) & (q > 0)) | ((p == 0) & (q == 0) & (r == 1))]
    pqr = pqr // np.gcd(np.gcd(pqr[:, 0], pqr[:, 1]), pqr[:, 2])[:, np.newaxis]
    pqr = np.vstack(self.pqr, pqr)
    self.pqr = np.unique(pqr, axis=0)


def find_preferential_orientation(dsv: DirectSpaceVectors, params_) -> dict:
  """Look for a preferential orientation in any direct space direction pqr"""
  pqr_array = PQRArray()
  wds: List[WatsonDistribution] = []
  for pqr in pqr_array.pqr:
    vectors = dsv.a * pqr[0] + dsv.b * pqr[1] + dsv.c * pqr[2]
    wds.append(WatsonDistribution.from_vectors(vectors))
  i = np.argmin([wd.nll for wd in wds]) # index of distribution with best fit
  print(f'Best fit found for direction {pqr_array.pqr[i]}: {str(wds[i])}')


########################### ORIENTATION VISUALIZING ###########################

@dataclass
class Hedgehog:
  """Class for holding any `SphericalDistribution` with its metadata"""
  distribution: SphericalDistribution
  color: str
  name: str


class HedgehogArtist:
  """Class responsible for drawing distribution of vectors as "hedgehogs"."""
  def __init__(self, parameters) -> None:
    self.parameters = parameters
    self.hedgehogs = []
    self._init_figure()

  def __len__(self) -> int:
    return len(self.hedgehogs)

  def _init_figure(self) -> None:
    self.fig = plt.figure()
    self.axes = []

  def _generate_axes(self) -> None:
    gs_width = np.ceil(np.sqrt(len(self))).astype(int)
    gs_height = np.ceil(len(self) / gs_width).astype(int)
    gs = GridSpec(gs_height, gs_width, hspace=0, wspace=0)
    for h in range(gs_height):
      for w in range(gs_width):
        self.axes.append(self.fig.add_subplot(gs[h, w], projection='3d'))

  def _plot_hedgehog(self, axes: plt.Axes, hedgehog: Hedgehog) -> None:
    origin = [0., 0., 0.]
    name = hedgehog.name
    v = hedgehog.distribution.vectors
    axes.quiver(*origin, v[:, 0], v[:, 1], v[:, 2], colors=hedgehog.color,
                alpha=0.1, arrow_length_ratio=0.0)
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    axes.set_zlim([-1, 1])
    axes.set_label(axes.get_label() + ' ' + name if axes.get_label() else name)

  def register_hedgehog(self, hedgehog: Hedgehog) -> None:
    self.hedgehogs.append(hedgehog)

  def plot(self):
    self._generate_axes()
    for axes, hedgehog in zip(self.axes, self.hedgehogs):
      self._plot_hedgehog(axes=axes, hedgehog=hedgehog)
    plt.show()


################################ ENTRY POINTS #################################


def run(params_):
  abc_stack = DirectSpaceVectors.from_glob(params_).abc
  hha = HedgehogArtist(parameters=params_)
  for vectors, color, name in zip(abc_stack, 'rgb', 'abc'):
    wd = WatsonDistribution.from_vectors(vectors)
    print(name + ': ' + str(wd))
    hh = Hedgehog(distribution=wd, color=color, name=name)
    hha.register_hedgehog(hh)
  hha.plot()


def exercise_watson_distribution():
  hha = HedgehogArtist(parameters=None)
  for kappa in [-1000, -100, -10, 0.000001, 10, 100]:
    wd = WatsonDistribution(mu=np.array([0, 0, 1]), kappa=kappa)
    wd.sample(1000)
    wd.fit(wd.vectors)
    print(wd)
    hh = Hedgehog(distribution=wd, color='r', name='kappa=5.0')
    hha.register_hedgehog(hh)
  hha.plot()


params = []
if __name__ == '__main__':
  if '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
    print(message)
    exit()
  params = params_from_phil(sys.argv[1:])
  run(params)