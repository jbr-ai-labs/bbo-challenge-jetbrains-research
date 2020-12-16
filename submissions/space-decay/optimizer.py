import json
import random
from copy import deepcopy
from typing import Optional

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace

# It depends on scikit-optimize==0.8.dev0, which is not in the default environment.
import sampler
from turbo1 import Turbo1
from util import copula_standardize

try:
  import open3d
  DEBUG = True
except ImportError as _:
  DEBUG = False


def fix_optimizer_seed(seed):
  if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _add_pcd(pcds, points, color):
  if len(points) == 0:
    return
  if points.shape[1] == 2:
    extended_points = np.zeros((len(points), 3))
    extended_points[:, :2] = points[:, :]
    points = extended_points
  elif points.shape[1] != 3:
    raise ValueError('The points for the DEBUG should either be 2D or 3D.')
  pcd = open3d.geometry.PointCloud()
  pcd.points = open3d.utility.Vector3dVector(points)
  pcd.colors = open3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
  pcds.append(pcd)


class SpacePartitioningOptimizer(AbstractOptimizer):
  primary_import = 'scikit-learn'

  def __init__(self, api_config, **kwargs):
    AbstractOptimizer.__init__(self, api_config)

    print('api_config:', api_config)
    self.api_config = api_config

    self.space_x = JointSpace(api_config)
    self.bounds = self.space_x.get_bounds()
    self.lb, self.ub = self.bounds[:, 0], self.bounds[:, 1]
    self.dim = len(self.bounds)

    self.X = np.zeros((0, self.dim))
    self.y = np.zeros((0, 1))

    self.X_init = None
    self.batch_size = None
    self.turbo = None
    self.split_used = 0
    self.node = None
    self.best_values = []

    self.config = self._read_config()
    print('config:', self.config)
    optimizer_seed = self.config.get('optimizer_seed')
    fix_optimizer_seed(optimizer_seed)
    self.sampler_seed = self.config.get('sampler_seed')
    sampler.fix_sampler_seed(self.sampler_seed)

    self.is_init_batch = False
    self.init_batches = []

  def _read_config(self):
    return {'turbo_training_steps': 100, 'turbo_length_retries': 10, 'turbo_length_init_method': 'default', 'experimental_design': 'lhs_classic_ratio', 'n_init_points': 24, 'max_tree_depth': 5, 'kmeans_resplits': 10, 'split_model': {'type': 'SVC', 'args': {'kernel': 'poly', 'gamma': 'scale', 'C': 745.3227447730735}}, 'reset_no_improvement': 8, 'reset_split_after': 4, 'turbo': {'budget': 128, 'use_cylinder': 0, 'use_pull': 0, 'use_lcb': 0, 'kappa': 2.0, 'use_decay': 1, 'decay_alpha': 0.49937937259674076, 'decay_threshold': 0.5, 'length_min': 1e-06, 'length_max': 2.0, 'length_init': 0.8, 'length_multiplier': 2.0}, 'sampler_seed': 42, 'optimizer_seed': 578330}

  def _init(self, n_suggestions):
    self.batch_size = n_suggestions
    n_init_points = self.config['n_init_points']
    if n_init_points == -1:
      # Special value to use the default 2*D+1 number.
      n_init_points = 2 * self.dim + 1
    self.n_init = max(self.batch_size, n_init_points)
    exp_design = self.config['experimental_design']
    if exp_design == 'latin_hypercube':
      X_init = latin_hypercube(self.n_init, self.dim)
    elif exp_design == 'halton':
      halton_sampler = sampler.Sampler(method='halton', api_config=self.api_config, n_points=self.n_init)
      X_init = halton_sampler.generate(random_state=self.sampler_seed)
      X_init = self.space_x.warp(X_init)
      X_init = to_unit_cube(X_init, self.lb, self.ub)
    elif exp_design == 'lhs_classic_ratio':
      lhs_sampler = sampler.Sampler(
        method='lhs',
        api_config=self.api_config,
        n_points=self.n_init,
        generator_kwargs={'lhs_type': 'classic', 'criterion': 'ratio'})
      X_init = lhs_sampler.generate(random_state=self.sampler_seed)
      X_init = self.space_x.warp(X_init)
      X_init = to_unit_cube(X_init, self.lb, self.ub)
    else:
      raise ValueError(f'Unknown experimental design: {exp_design}.')
    self.X_init = X_init
    if DEBUG:
      print(f'Initialized the method with {self.n_init} points by {exp_design}:')
      print(X_init)

  def _get_split_model(self, X, kmeans_labels):
    split_model_config = self.config['split_model']
    model_type = split_model_config['type']
    args = split_model_config['args']
    if model_type == 'SVC':
      split_model = SVC(**args, max_iter=10**7)
    elif model_type == 'KNeighborsClassifier':
      split_model = KNeighborsClassifier(**args)
    else:
      raise ValueError(f'Unknown split model type in the config: {model_type}.')

    split_model.fit(X, kmeans_labels)
    split_model_predictions = split_model.predict(X)
    split_model_matches = np.sum(split_model_predictions == kmeans_labels)
    split_model_mismatches = np.sum(split_model_predictions != kmeans_labels)
    print('Labels for the split model:', kmeans_labels)
    print('Predictions of the split model:', split_model_predictions)
    print(f'Split model matches {split_model_matches} and mismatches {split_model_mismatches}')
    return split_model

  def _find_split(self, X, y) -> Optional:
    max_margin = None
    max_margin_labels = None
    for _ in range(self.config['kmeans_resplits']):
      kmeans = KMeans(n_clusters=2).fit(y)
      kmeans_labels = kmeans.labels_
      if np.count_nonzero(kmeans_labels == 1) > 0 and np.count_nonzero(kmeans_labels == 0) > 0:
        if np.mean(y[kmeans_labels == 1]) > np.mean(y[kmeans_labels == 0]):
          # Reverse labels if the entries with 1s have a higher mean error, since 1s go to the left branch.
          kmeans_labels = 1 - kmeans_labels
        margin = -(np.mean(y[kmeans_labels == 1]) - np.mean(y[kmeans_labels == 0]))
        if DEBUG:
          print('MARGIN is', margin, np.count_nonzero(kmeans_labels == 1), np.count_nonzero(kmeans_labels == 0))
        if max_margin is None or margin > max_margin:
          max_margin = margin
          max_margin_labels = kmeans_labels
    if DEBUG:
      print('MAX MARGIN is', max_margin)
    if max_margin_labels is None:
      return None
    else:
      return self._get_split_model(X, max_margin_labels)

  def _build_tree(self, X, y, depth=0):
    print('len(X) in _build_tree is', len(X))
    if depth == self.config['max_tree_depth']:
      return []
    split = self._find_split(X, y)
    if split is None:
      return []
    in_region_points = split.predict(X)
    left_subtree_size = np.count_nonzero(in_region_points == 1)
    right_subtree_size = np.count_nonzero(in_region_points == 0)
    print(f'{len(X)} points would be split {left_subtree_size}/{right_subtree_size}.')
    if left_subtree_size < self.n_init:
      return []
    idx = (in_region_points == 1)
    splits = self._build_tree(X[idx], y[idx], depth + 1)
    return [split] + splits

  def _get_in_node_region(self, points, splits):
    in_region = np.ones(len(points))
    for split in splits:
      split_in_region = split.predict(points)
      in_region *= split_in_region
    return in_region

  def _suggest(self, n_suggestions):
    X = to_unit_cube(deepcopy(self.X), self.lb, self.ub)
    y = deepcopy(self.y)
    if not self.node:
      self.split_used = 0
      self.node = self._build_tree(X, y)
      used_budget = len(y)
      idx = (self._get_in_node_region(X, self.node) == 1)
      X = X[idx]
      y = y[idx]
      print(f'Rebuilt the tree of depth {len(self.node)}')
      model_config = self.config['turbo']
      #print('CONFIG!!!!!', model_config)
      self.turbo = Turbo1(
        f=None,
        lb=self.bounds[:, 0],
        ub=self.bounds[:, 1],
        n_init=len(X),
        max_evals=np.iinfo(np.int32).max,
        batch_size=self.batch_size,
        verbose=False,
        use_cylinder=model_config['use_cylinder'],
        budget=model_config['budget'],
        use_decay=model_config['use_decay'],
        decay_threshold=model_config['decay_threshold'],
        decay_alpha=model_config['decay_alpha'],
        use_pull=model_config['use_pull'],
        use_lcb=model_config['use_lcb'],
        kappa=model_config['kappa'],
        length_min=model_config['length_min'],
        length_max=model_config['length_max'],
        length_init=model_config['length_init'],
        length_multiplier=model_config['length_multiplier'],
        used_budget=used_budget
      )
      self.turbo._X = np.array(X, copy=True)
      self.turbo._fX = np.array(y, copy=True)
      self.turbo.X = np.array(X, copy=True)
      self.turbo.fX = np.array(y, copy=True)
      print('Initialized TURBO')
    else:
      idx = (self._get_in_node_region(X, self.node) == 1)
      X = X[idx]
      y = y[idx]
    self.split_used += 1

    length_init_method = self.config['turbo_length_init_method']
    if length_init_method == 'default':
      length = self.turbo.length
    elif length_init_method == 'length_init':
      length = self.turbo.length_init
    elif length_init_method == 'length_max':
      length = self.turbo.length_max
    elif length_init_method == 'infinity':
      length = np.iinfo(np.int32).max
    else:
      raise ValueError(f'Unknown init method for turbo\'s length: {length_init_method}.')
    length_reties = self.config['turbo_length_retries']
    for retry in range(length_reties):
      XX = X
      yy = copula_standardize(y.ravel())
      X_cand, y_cand, _ = self.turbo._create_candidates(
        XX, yy, length=length, n_training_steps=self.config['turbo_training_steps'], hypers={})
      in_region_predictions = self._get_in_node_region(X_cand, self.node)
      in_region_idx = in_region_predictions == 1
      if DEBUG:
        print(f'In region: {np.sum(in_region_idx)} out of {len(X_cand)}')
      if np.sum(in_region_idx) >= n_suggestions:
        X_cand, y_cand = X_cand[in_region_idx], y_cand[in_region_idx]
        self.turbo.f_var = self.turbo.f_var[in_region_idx]
        if DEBUG:
          print('Found a suitable set of candidates.')
        break
      else:
        length /= 2
        if DEBUG:
          print(f'Retrying {retry + 1}/{length_reties} time')

    X_cand = self.turbo._select_candidates(X_cand, y_cand)[:n_suggestions, :]
    if DEBUG:
      if X.shape[1] == 3:
        tx = np.arange(0.0, 1.0 + 1e-6, 0.1)
        ty = np.arange(0.0, 1.0 + 1e-6, 0.1)
        tz = np.arange(0.0, 1.0 + 1e-6, 0.1)
        p = np.array([[x, y, z] for x in tx for y in ty for z in tz])
      elif X.shape[1] == 2:
        tx = np.arange(0.0, 1.0 + 1e-6, 0.1)
        ty = np.arange(0.0, 1.0 + 1e-6, 0.1)
        p = np.array([[x, y] for x in tx for y in ty])
      else:
        raise ValueError('The points for the DEBUG should either be 2D or 3D.')
      p_predictions = self._get_in_node_region(p, self.node)
      in_turbo_bounds = np.logical_and(
        np.all(self.turbo.cand_lb <= p, axis=1),
        np.all(p <= self.turbo.cand_ub, axis=1))
      pcds = []
      _add_pcd(pcds, p[p_predictions == 0], (1.0, 0.0, 0.0))
      _add_pcd(pcds, p[np.logical_and(p_predictions == 1, np.logical_not(in_turbo_bounds))], (0.0, 1.0, 0.0))
      _add_pcd(pcds, p[np.logical_and(p_predictions == 1, in_turbo_bounds)], (0.0, 0.5, 0.0))
      _add_pcd(pcds, X_cand, (0.0, 0.0, 0.0))
      open3d.visualization.draw_geometries(pcds)
    return X_cand

  def suggest(self, n_suggestions=1):
    X_suggestions = np.zeros((n_suggestions, self.dim))
    # Initialize the design if it is the first call
    if self.X_init is None:
      self._init(n_suggestions)
      if self.init_batches:
        print('REUSING INITIALIZATION:')
        for X, Y in self.init_batches:
          print('Re-observing a batch!')
          self.observe(X, Y)
        self.X_init = []

    # Pick from the experimental design
    n_init = min(len(self.X_init), n_suggestions)
    if n_init > 0:
      X_suggestions[:n_init] = self.X_init[:n_init]
      self.X_init = self.X_init[n_init:]
      self.is_init_batch = True
    else:
      self.is_init_batch = False

    # Pick from the model based on the already received observations
    n_suggest = n_suggestions - n_init
    if n_suggest > 0:
      X_cand = self._suggest(n_suggest)
      X_suggestions[-n_suggest:] = X_cand

    # Map into the continuous space with the api bounds and unwarp the suggestions
    X_min_bound = 0.0
    X_max_bound = 1.0
    X_suggestions_min = X_suggestions.min()
    X_suggestions_max = X_suggestions.max()
    if X_suggestions_min < X_min_bound or X_suggestions_max > X_max_bound:
      print(f'Some suggestions are out of the bounds in suggest(): {X_suggestions_min}, {X_suggestions_max}')
      print('Clipping everything...')
      X_suggestions = np.clip(X_suggestions, X_min_bound, X_max_bound)
    X_suggestions = from_unit_cube(X_suggestions, self.lb, self.ub)
    X_suggestions = self.space_x.unwarp(X_suggestions)
    return X_suggestions

  def observe(self, X_observed, Y_observed):
    if self.is_init_batch:
      self.init_batches.append([X_observed, Y_observed])
    X, Y = [], []
    for x, y in zip(X_observed, Y_observed):
      if np.isfinite(y):
        X.append(x)
        Y.append(y)
      else:
        # Ignore for now; could potentially substitute with an upper bound.
        continue
    if not X:
      return
    X, Y = self.space_x.warp(X), np.array(Y)[:, None]
    self.X = np.vstack((self.X, deepcopy(X)))
    self.y = np.vstack((self.y, deepcopy(Y)))
    self.best_values.append(Y.min())

    if self.turbo:
      if len(self.turbo._X) >= self.turbo.n_init:
        self.turbo._adjust_length(Y)
      print('TURBO length:', self.turbo.length)
      self.turbo._X = np.vstack((self.turbo._X, deepcopy(X)))
      self.turbo._fX = np.vstack((self.turbo._fX, deepcopy(Y)))
      self.turbo.X = np.vstack((self.turbo.X, deepcopy(X)))
      self.turbo.fX = np.vstack((self.turbo.fX, deepcopy(Y)))

    N = self.config['reset_no_improvement']
    if len(self.best_values) > N and np.min(self.best_values[:-N]) <= np.min(self.best_values[-N:]):
      print('########## RESETTING COMPLETELY! ##########')
      self.X = np.zeros((0, self.dim))
      self.y = np.zeros((0, 1))
      self.best_values = []
      self.X_init = None
      self.node = None
      self.turbo = None
      self.split_used = 0

    if self.split_used >= self.config['reset_split_after']:
      print('########## REBUILDING THE SPLIT! ##########')
      self.node = None
      self.turbo = None
      self.split_used = 0


if __name__ == '__main__':
  experiment_main(SpacePartitioningOptimizer)
