import random

from skopt.space import Categorical, Integer, Real
from skopt.sampler import InitialPointGenerator, Sobol, Halton, Lhs, Hammersly, Grid
from skopt.space import Space
from scipy.special import binom
from scipy.optimize import minimize
import numpy as np
from sklearn.utils import check_random_state
from scipy.interpolate import interp1d

def fix_sampler_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

def cook_initial_point_generator(generator, **kwargs):
    """Cook a default initial point generator.
    For the special generator called "random" the return value is None.
    Parameters
    ----------
    generator : "lhs", "sobol", "halton", "hammersly", "grid", "random" \
            or InitialPointGenerator instance"
        Should inherit from `skopt.sampler.InitialPointGenerator`.
    kwargs : dict
        Extra parameters provided to the generator at init time.
    """
    if generator is None:
        generator = "random"
    elif isinstance(generator, str):
        generator = generator.lower()
        if generator not in ["sobol", "halton", "hammersly", "lhs", "random",
                             "grid", "maxpro", "maxpro-gd"]:
            raise ValueError("Valid strings for the generator parameter "
                             " are: 'sobol', 'lhs', 'halton', 'hammersly',"
                             "'random', 'maxpro','maxpro-gd', or 'grid' not "
                             "%s." % generator)
    elif not isinstance(generator, InitialPointGenerator):
        raise ValueError("generator has to be an InitialPointGenerator."
                         "Got %s" % (str(type(generator))))

    if isinstance(generator, str):
        if generator == "sobol":
            generator = Sobol()
        elif generator == "halton":
            generator = Halton()
        elif generator == "hammersly":
            generator = Hammersly()
        elif generator == "lhs":
            generator = Lhs()
        elif generator == "grid":
            generator = Grid()
        elif generator == "random":
            return None
        elif generator == "maxpro":
            generator = MaxPro(use_gradient=False)
        elif generator == "maxpro-gd":
            generator = MaxPro(use_gradient=True)
    generator.set_params(**kwargs)
    return generator

def _random_permute_matrix(h, random_state=None):
    rng = check_random_state(random_state)
    h_rand_perm = np.zeros_like(h)
    samples, n = h.shape
    for j in range(n):
        order = rng.permutation(range(samples))
        h_rand_perm[:, j] = h[order, j]
    return h_rand_perm


class MaxPro(InitialPointGenerator):
    """Latin hypercube sampling
    Parameters
    ----------
    lhs_type : str, default='classic'
        - 'classic' - a small random number is added
        - 'centered' - points are set uniformly in each interval
    criterion : str or None, default='maximin'
        When set to None, the LHS is not optimized
        - 'correlation' : optimized LHS by minimizing the correlation
        - 'maximin' : optimized LHS by maximizing the minimal pdist
        - 'ratio' : optimized LHS by minimizing the ratio
          `max(pdist) / min(pdist)`
    iterations : int
        Defines the number of iterations for optimizing LHS
    """
    def __init__(self,
                 iterations=1000, use_gradient=True, lhs_type = "classic"):
        self.iterations = iterations
        self.use_gradient = use_gradient
        self.lhs_type = lhs_type

    def generate(self, dimensions, n_samples, random_state=None):
        """Creates latin hypercube samples with maxpro criterion.
        Parameters
        ----------
        dimensions : list, shape (n_dims,)
            List of search space dimensions.
            Each search dimension can be defined either as
            - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
              dimensions),
            - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
              dimensions),
            - as a list of categories (for `Categorical` dimensions), or
            - an instance of a `Dimension` object (`Real`, `Integer` or
              `Categorical`).
        n_samples : int
            The order of the LHS sequence. Defines the number of samples.
        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.
        Returns
        -------
        np.array, shape=(n_dim, n_samples)
            LHS set
        """
        rng = check_random_state(random_state)
        space = Space(dimensions)
        transformer = space.get_transformer()
        n_dim = space.n_dims
        space.set_transformer("normalize")
        h = self._lhs_normalized(n_dim, n_samples, rng)

        self.num_pts = n_samples
        self.dim = n_dim
        if self.use_gradient:
            print('Using gradient descent')
            bounds = [(0,1)] * len(dimensions) * self.num_pts
            h_opt = minimize(self.maxpro_criter, h, jac=self.maxpro_grad, bounds=bounds)
            h_opt = h_opt['x'].reshape(n_samples, n_dim)
        else:
            print('Using naive method')
            best = 1e+6
            for i in range(self.iterations):
                h = self._lhs_normalized(n_dim, n_samples, i*rng)
                criter = self.maxpro_criter(h)
                if best > criter:
                    best = criter
                    h_opt = h.copy()
        h_opt = space.inverse_transform(h_opt)
        space.set_transformer(transformer)
        return h_opt

    def maxpro_criter(self, X):
        """
        :param X: all x data
        :return: value of MaxPro criterion
        """
        x = X.copy()
        #print(x.shape)
        if x.ndim < 2:
            x = x.reshape(self.num_pts, self.dim)
            #print(x.shape, self.dim)
        res = 1 / binom(self.num_pts, 2)
        sum_part = 0
        for i in range(self.num_pts-1):
            for j in range(i+1, self.num_pts):
                #print(np.prod(((x[i] - x[j]) ** 2)))
                #print(np.prod(((x[i] - x[j]) ** 2)) ** (-1))
                sum_part += (np.prod(((x[i] - x[j]) ** 2)) + 1e-8)** (-1)
        res *= sum_part ** (1/self.dim)
        return res

    def maxpro_deriv(self, x, r, s):
        """Returns derivative of maxpro criterion for Z_rs

        :param X: all x points
        :param r: number of point to get derivative
        :param s: number of coordinate to get derivative
        :return: derivative value
        """
        res = 2 / binom(self.num_pts, 2)
        sum_part = 0
        for i in range(len(x)):
            if i != r:
                sum_part += (np.prod(((x[i] - x[r]) ** 2)) + 1e-8) ** (-1) * (x[i][s] - x[r][s] + 1e-8) ** (-1)
        res *= sum_part
        return res

    def maxpro_grad(self, X):
        x = X.copy()
        if x.ndim < 2:
            x = x.reshape(self.num_pts, self.dim)
        grad_val = np.zeros(x.shape)
        for r in range(self.num_pts):
            for i in range(self.dim):
                grad_val[r][i] = self.maxpro_deriv(x, r, i)
        return grad_val.flatten()

    def _lhs_normalized(self, n_dim, n_samples, random_state):
        rng = check_random_state(random_state)
        x = np.linspace(0, 1, n_samples + 1)
        u = rng.rand(n_samples, n_dim)
        h = np.zeros_like(u)
        if self.lhs_type == "centered":
            for j in range(n_dim):
                h[:, j] = np.diff(x) / 2.0 + x[:n_samples]
        elif self.lhs_type == "classic":
            for j in range(n_dim):
                h[:, j] = u[:, j] * np.diff(x) + x[:n_samples]
        else:
            raise ValueError("Wrong lhs_type. Got ".format(self.lhs_type))
        return _random_permute_matrix(h, random_state=rng)

class Sampler:
    def __init__(self, method, api_config, n_points=8, generator_kwargs=None):
        if generator_kwargs is None:
            generator_kwargs = {}
        self.method = cook_initial_point_generator(method, **generator_kwargs)

        self.dimensions, self.round_to_values = Sampler.get_sk_dimensions(api_config)
        self.dimensions_list = tuple(dd.name for dd in self.dimensions)

        self.n_points = n_points

    @staticmethod
    def get_sk_dimensions(api_config, transform="normalize"):
        """Help routine to setup skopt search space in constructor.

        Take api_config as argument so this can be static.
        """
        # The ordering of iteration prob makes no difference, but just to be
        # safe and consistnent with space.py, I will make sorted.
        param_list = sorted(api_config.keys())

        sk_dims = []
        round_to_values = {}
        for param_name in param_list:
            param_config = api_config[param_name]

            param_type = param_config["type"]

            param_space = param_config.get("space", None)
            param_range = param_config.get("range", None)
            param_values = param_config.get("values", None)

            # Some setup for case that whitelist of values is provided:
            values_only_type = param_type in ("cat", "ordinal")
            if (param_values is not None) and (not values_only_type):
                assert param_range is None
                param_values = np.unique(param_values)
                param_range = (param_values[0], param_values[-1])
                round_to_values[param_name] = interp1d(
                    param_values, param_values, kind="nearest", fill_value="extrapolate"
                )

            if param_type == "int":
                # Integer space in sklearn does not support any warping => Need
                # to leave the warping as linear in skopt.
                sk_dims.append(Integer(param_range[0], param_range[-1], transform=transform, name=param_name))
            elif param_type == "bool":
                assert param_range is None
                assert param_values is None
                sk_dims.append(Integer(0, 1, transform=transform, name=param_name))
            elif param_type in ("cat", "ordinal"):
                assert param_range is None
                # Leave x-form to one-hot as per skopt default
                sk_dims.append(Categorical(param_values, name=param_name))
            elif param_type == "real":
                # Skopt doesn't support all our warpings, so need to pick
                # closest substitute it does support.
                prior = "log-uniform" if param_space in ("log", "logit") else "uniform"
                sk_dims.append(Real(param_range[0], param_range[-1], prior=prior, transform=transform, name=param_name))
            else:
                assert False, "type %s not handled in API" % param_type
        return sk_dims, round_to_values

    def generate(self, random_state):
        # First get list of lists from the sampling method.
        next_guess = self.method.generate(dimensions=self.dimensions,
                                          n_samples=self.n_points,
                                          random_state=random_state)
        # Then convert to list of dicts
        next_guess = [dict(zip(self.dimensions_list, x)) for x in next_guess]

        # Now do the rounding, custom rounding is not supported in skopt. Note
        # that there is not nec a round function for each dimension here.
        for param_name, round_f in self.round_to_values.items():
            for xx in next_guess:
                xx[param_name] = round_f(xx[param_name])
        return next_guess


if __name__ == "__main__":
    api_config = {
        "max_depth": {
            "type": "int",
            "space": "linear",
            "range": (1, 15)
        },
        "min_samples_split": {
            "type": "real",
            "space": "logit",
            "range": (0.01, 0.99)
        },
        "min_samples_leaf": {
            "type": "real",
            "space": "logit",
            "range": (0.01, 0.49)
        },
        "min_weight_fraction_leaf": {
            "type": "real",
            "space": "logit",
            "range": (0.01, 0.49)
        },
        "max_features": {
            "type": "real",
            "space": "logit",
            "range": (0.01, 0.99)
        },
        "min_impurity_decrease": {
            "type": "real",
            "space": "linear",
            "range": (0.0, 0.5)
        },
    }
    n_points = 8

    sobol_points = Sampler(method='sobol', api_config=api_config, n_points=n_points).generate(random_state=42)
    halton_points = Sampler(method='halton', api_config=api_config, n_points=n_points).generate(random_state=42)
    hammersly_points = Sampler(method='hammersly', api_config=api_config, n_points=n_points).generate(random_state=42)
    lhs_classic_points = Sampler(method='lhs', api_config=api_config, n_points=n_points, generator_kwargs={'lhs_type': 'classic', 'criterion': 'maximin'}).generate(random_state=42)
    lhs_centered_points = Sampler(method='lhs', api_config=api_config, n_points=n_points, generator_kwargs={'lhs_type': 'centered'}).generate(random_state=42)
    grid_points = Sampler(method='grid', api_config=api_config, n_points=n_points).generate(random_state=42)

    t = 0
