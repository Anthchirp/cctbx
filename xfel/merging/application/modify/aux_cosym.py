from __future__ import absolute_import, division, print_function
import os
from scitbx.array_family import flex
from cctbx.array_family import flex as cctbx_flex
from cctbx import sgtbx, miller
from libtbx import easy_mp, Auto
from scipy import sparse
import numpy as np
from ordered_set import OrderedSet
import copy

# Specialization, run only a subset of cosym steps and include plot
from dials.algorithms.symmetry.cosym import CosymAnalysis as BaseClass
from dials.util.multi_dataset_handling import select_datasets_on_identifiers

from dials.algorithms.symmetry.cosym.target import Target

from cctbx.sgtbx.literal_description import literal_description

class CosymAnalysis(BaseClass):

  def __init__(self, *args, **kwargs):
    self.do_plot = kwargs.pop('do_plot', False)
    self.i_plot = kwargs.pop('i_plot', None)
    self.plot_fname = kwargs.pop('plot_fname', None)
    self.plot_format = kwargs.pop('plot_format', None)
    self.output_dir = kwargs.pop('output_dir', None)
    self.cb_op = kwargs.pop('cb_op', None)
    super(CosymAnalysis, self).__init__(*args, **kwargs)

  def run(self):
    super(CosymAnalysis, self).run()
    if self.do_plot: self.plot_after_cluster_analysis()
    # we have the xy embedded coords at this point.

  def plot_after_optimize(self):
          print ("optimized coordinates", self.coords.shape)
          xx = []
          yy = []
          for item in range(self.coords.shape[0]):
            xx.append(self.coords[(item,0)])
            yy.append(self.coords[(item,1)])
          from matplotlib import pyplot as plt
          plt.plot(xx,yy,"r.")
          # denominator of 12 is specific to the use case of P6 (# symops in the metric superlattice)
          plt.plot(xx[::len(xx)//12],yy[::len(yy)//12],"b.")
          plt.plot(xx[:1],yy[:1],"g.")
          plt.axes().set_aspect("equal")
          circle = plt.Circle((0,0),1,fill=False,edgecolor="b")
          ax = plt.gca()
          ax.add_artist(circle)
          plt.show()

  def plot_after_cluster_analysis(self):
      if self.coords.shape[1] == 2: # one twining operator, most merohedry problems
          xx = flex.double()
          yy = flex.double()
          for item in range(self.coords.shape[0]):
            xx.append(self.coords[(item,0)])
            yy.append(self.coords[(item,1)])
          from matplotlib import pyplot as plt
          plt.plot(xx, yy, 'r.')
          plt.plot(xx[0:1], yy[0:1], 'k.')
          plt.plot([0,0],[-0.01,0.01],"k-")
          plt.plot([-0.01,0.01],[0,0],"k-")
          plt.xlim(-0.2,1.0)
          plt.ylim(-0.2,1.0)
          plt.title("$<w_{ij}>$=%.1f"%(np.mean(self.target.wij_matrix)))
          ax = plt.gca()
          ax.set_aspect("equal")
          circle = plt.Circle((0,0),1,fill=False,edgecolor="b")
          ax.add_artist(circle)
      elif self.coords.shape[1] == 4: # special case of P3 having 4 cosets
          # cast the data into two different shaped arrays.
          # one for me
          xyzt = [flex.double(),flex.double(),flex.double(),flex.double()]
          datasize = self.coords.shape[0]
          for item in range(datasize):
            for icoset in range(4):
              xyzt[icoset].append(self.coords[(item,icoset)])
          # and one for sklearn
          xsrc = []
          for i in range(datasize):
            pt = []
            for j in range(4):
              pt.append(xyzt[j][i])
            xsrc.append(pt)
          X = np.array(xsrc)

          from sklearn.cluster import DBSCAN
          db = DBSCAN(eps=0.1, min_samples=10).fit(X)
          labels = db.labels_

          # Number of clusters in labels, ignoring noise if present.
          n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
          n_noise_ = list(labels).count(-1)

          print("Estimated number of clusters: %d" % n_clusters_)
          print("Estimated number of noise points: %d" % n_noise_)

          coset_keys = list(set(self.reindexing_ops))
          assert len(coset_keys)==4 # result from dials cosym nearest neighbor clustering
          assert n_clusters_ == 4 # result from dbscan

          assert len(labels)==datasize
          print("unique labels",set(labels))

          from matplotlib import pyplot as plt
          colordict = {-1:"black",0:"green",1:"red",2:"magenta",3:"blue"}
          colors = [colordict[item] for item in labels]

          import matplotlib.gridspec as gridspec
          fig = plt.figure(figsize=(8,7))
          gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1])
          label={0:"$x$",1:"$y$",2:"$z$",3:"$t$"}
          for axp, axf in zip([(0,0),(0,1),(1,0),(1,1)],[(0,1),(3,1),(0,2),(3,2)]):
            axspec = fig.add_subplot(gs[axp[0],axp[1]])
            axspec.scatter(xyzt[axf[0]], xyzt[axf[1]], c=colors, marker='.')

            axspec.plot([0,0],[-0.01,0.01],"k-")
            axspec.plot([-0.01,0.01],[0,0],"k-")
            axspec.set_xlim(-0.2,1.0)
            axspec.set_ylim(-0.2,1.0)
            axspec.set_xlabel(label[axf[0]])
            axspec.set_ylabel(label[axf[1]])
            axspec.set_aspect("equal")
            plt.title("$<w_{ij}>$=%.1f"%(np.mean(self.target.wij_matrix)))
            if axp[0]==1: plt.title("")
            ax = plt.gca()
            circle = plt.Circle((0,0),1,fill=False,edgecolor="b")
            ax.add_artist(circle)

      if self.plot_fname is None:
            plt.show()
      else:
            plot_path = os.path.join(self.output_dir, self.plot_fname)
            plot_fname = "{}_{}.{}".format(
                plot_path, self.i_plot, self.plot_format)
            plt.savefig(plot_fname)

  def _intialise_target(self):
      if self.params.dimensions is Auto:
          dimensions = None
      else:
          dimensions = self.params.dimensions
      if self.params.lattice_group is not None:
          self.lattice_group = (
              self.params.lattice_group.group()
              .build_derived_patterson_group()
              .info()
              .primitive_setting()
              .group()
          )
      if self.params.twin_axis:
        twin_axis = [tuple(x) for x in self.params.twin_axis]
        twin_rotation = self.params.twin_rotation
      else:
        twin_axis = None
        twin_rotation = None
      self.target = TargetWithCustomSymops(
          self.intensities,
          self.dataset_ids.as_numpy_array(),
          min_pairs=self.params.min_pairs,
          lattice_group=self.lattice_group,
          dimensions=dimensions,
          weights=self.params.weights,
          twin_axes=twin_axis,
          twin_angles=twin_rotation,
          cb_op=self.cb_op
      )



from dials.command_line.cosym import logger
from dials.command_line.cosym import cosym as dials_cl_cosym_wrapper
from dials.util.exclude_images import get_selection_for_valid_image_ranges
from dials.command_line.symmetry import (
    apply_change_of_basis_ops,
    change_of_basis_ops_to_minimum_cell,
    eliminate_sys_absent,
)
from dials.util.filter_reflections import filtered_arrays_from_experiments_reflections

class dials_cl_cosym_subclass (dials_cl_cosym_wrapper):
    def __init__(self, experiments, reflections, uuid_cache_in, params=None,
            do_plot=False, i_plot=None, output_dir=None):
        super(dials_cl_cosym_wrapper, self).__init__(
            events=["run_cosym", "performed_unit_cell_clustering"]
        )
        if params is None:
            params = phil_scope.extract()
        self.params = params

        self._reflections = []
        for refl, expt in zip(reflections, experiments):
            sel = get_selection_for_valid_image_ranges(refl, expt)
            self._reflections.append(refl.select(sel))

        self._experiments, self._reflections = self._filter_min_reflections(
            experiments, self._reflections, uuid_cache_in
        )
        self.ids_to_identifiers_map = {}
        for table in self._reflections:
            self.ids_to_identifiers_map.update(table.experiment_identifiers())
        self.identifiers_to_ids_map = {
            value: key for key, value in self.ids_to_identifiers_map.items()
        }

        if len(self._experiments) > 1:
            # perform unit cell clustering
            identifiers = self._unit_cell_clustering(self._experiments)
            if len(identifiers) < len(self._experiments):
                logger.info(
                    "Selecting subset of %i datasets for cosym analysis: %s"
                    % (len(identifiers), str(identifiers))
                )
                self._experiments, self._reflections = select_datasets_on_identifiers(
                    self._experiments, self._reflections, use_datasets=identifiers
                )
                self.uuid_cache = [self.uuid_cache[int(id)] for id in identifiers]

        # Map experiments and reflections to minimum cell
        cb_ops = change_of_basis_ops_to_minimum_cell(
            self._experiments,
            params.lattice_symmetry_max_delta,
            params.relative_length_tolerance,
            params.absolute_angle_tolerance,
        )
        in_cb_ops = len(cb_ops)
        exclude = [
            expt.identifier
            for expt, cb_op in zip(self._experiments, cb_ops)
            if not cb_op
        ]
        if len(exclude):
            logger.info(
                "Rejecting {} datasets from cosym analysis "\
                "(couldn't determine consistent cb_op to minimum cell):\n"\
                "{}".format(len(exclude), exclude)
            )
            self._experiments, self._reflections = select_datasets_on_identifiers(
                self._experiments, self._reflections, exclude_datasets=exclude
            )
            assert len(cb_ops) == len(self.uuid_cache)
            self.uuid_cache = [
                x for i, x in enumerate(self.uuid_cache)
                if cb_ops[i] is not None
            ]
            cb_ops = list(filter(None, cb_ops))

        ex_cb_ops = len(cb_ops)

        #Normally we expect that all the cb_ops are the same (applicable for PSI with P63)
        assertion_dict = {}
        for cb_op in cb_ops:
          key_ = cb_op.as_hkl()
          assertion_dict[key_] = assertion_dict.get(key_, 0)
          assertion_dict[key_] += 1
        if len(assertion_dict) != 1:
          # unexpected, there is normally only 1 cb operator to minimum cell
          from libtbx.mpi4py import MPI
          mpi_rank = MPI.COMM_WORLD.Get_rank()
          mpi_size = MPI.COMM_WORLD.Get_size()
          print ("RANK %02d, # experiments %d, after exclusion %d, unexpectedly there are %d unique cb_ops: %s"%(
            mpi_rank, in_cb_ops, ex_cb_ops, len(assertion_dict),
            ", ".join(["%s:%d"%(key,assertion_dict[key]) for key in assertion_dict])))
          # revisit with different use cases later

          # In fact we need all cb_ops to match because the user might supply
          # a custom reindexing operator and we need to consistently tranform
          # it from the conventional basis into the minimum basis. Therefore,
          # force them all to match, but make sure user is aware.
          if not params.single_cb_op_to_minimum:
            raise RuntimeError('There are >1 different cb_ops to minimum and \
cosym.single_cb_op_to_minimum is not True')
          else:
            best_cb_op_str = max(assertion_dict, key=assertion_dict.get)
            best_cb_op = None
            for cb_op in cb_ops:
              if cb_op.as_hkl() == best_cb_op_str:
                best_cb_op = cb_op
                break
            assert best_cb_op is not None
            cb_ops = [best_cb_op] * len(cb_ops)

        self.cb_op_to_minimum = cb_ops



        # Eliminate reflections that are systematically absent due to centring
        # of the lattice, otherwise they would lead to non-integer miller indices
        # when reindexing to a primitive setting
        self._reflections = eliminate_sys_absent(self._experiments, self._reflections)

        self._experiments, self._reflections = apply_change_of_basis_ops(
            self._experiments, self._reflections, cb_ops
        )

        # transform models into miller arrays
        datasets = filtered_arrays_from_experiments_reflections(
            self.experiments,
            self.reflections,
            outlier_rejection_after_filter=False,
            partiality_threshold=params.partiality_threshold,
        )

        datasets = [
            ma.as_anomalous_array().merge_equivalents().array() for ma in datasets
        ]
        # opportunity here to subclass as defined above, instead of the dials-implemented version
        self.params.min_reflections = 0 # avoid any further filtering implemented in https://github.com/dials/dials/pull/2741
        self.cosym_analysis = CosymAnalysis(
            datasets,
            self.params,
            do_plot=do_plot,
            i_plot=i_plot,
            plot_fname=self.params.plot.filename,
            plot_format=self.params.plot.format,
            output_dir=output_dir,
            cb_op=cb_ops[0]
            )

    def _filter_min_reflections(self, experiments, reflections, uuid_cache_in):
        identifiers = []
        self.uuid_cache = []
        for expt, refl, uuid in zip(experiments, reflections, uuid_cache_in):
            if len(refl) >= self.params.min_reflections:
                identifiers.append(expt.identifier)
                self.uuid_cache.append(uuid)

        return select_datasets_on_identifiers(
            experiments, reflections, use_datasets=identifiers
        )


class TargetWithFastRij(Target):
  def __init__(self, *args, **kwargs):

    # nproc is an init arg that was removed from class
    # dials.algorithms.symmetry.cosym.target.Target in dials commit 1cd5afe4
    self._nproc = kwargs.pop('nproc', 1)

    # if test_data_path is provided, we are constructing this for a unit test
    test_data_path = kwargs.pop('test_data_path', None)
    if test_data_path is None:
      super(TargetWithFastRij, self).__init__(*args, **kwargs)
      return
    else:
      # This is only for unit testing
      import pickle
      import numpy as np
      self._nproc = 1
      with open(test_data_path, 'rb') as f:
        self._lattices = np.array(pickle.load(f))
        self.sym_ops = pickle.load(f)
        self._weights = pickle.load(f)
        self._data = pickle.load(f)
        self._patterson_group = pickle.load(f)
        self._min_pairs = 3 # minimum number of mutual miller indices for a match

        # truncate the input data to save time
        # self._lattices is the list of start index of each dataset
        self._lattices = self._lattices[:11]

        i_last = self._lattices[-1]
        self._data = self._data[:i_last]

        # now remove the last index in self._lattices as this was just used
        # to determine the start index of the next dataset (which does not
        # exist in the cut set)
        self._lattices = self._lattices[:-1]

  def _lattice_lower_upper_index(self, lattice_id):
       lower_index = int(self._lattices[lattice_id])
       upper_index = None
       if lattice_id < len(self._lattices) - 1:
           upper_index = int(self._lattices[lattice_id + 1])
       else:
           assert lattice_id == len(self._lattices) - 1
       return lower_index, upper_index

  def compute_gradients(self, x):
    grad = super(TargetWithFastRij, self).compute_gradients(x)
    return grad.A1

  def _compute_rij_wij(self, use_cache=True, use_super=False):

    if use_super:
      # for testing
      return super()._compute_rij_wij(use_cache=use_cache)

    def _compute_one_row(args):
      """
      Call the compute_one_row method of CC, which is a compute_rij_wij_detail
      instance. Args is a tuple that we unpack because easy_mp.parallel_map
      can only pass a single argument.
      """
      CC, i, NN = args
      rij_row, rij_col, rij_data, wij_row, wij_col, wij_data = [
          list(x) for x in CC.compute_one_row(self._lattices.size, i)
      ]
      rij = sparse.coo_matrix((rij_data, (rij_row, rij_col)), shape=(NN, NN))
      wij = sparse.coo_matrix((wij_data, (wij_row, wij_col)), shape=(NN, NN))
      return rij, wij

    n_lattices = self._lattices.size
    n_sym_ops = len(self.sym_ops)
    NN = n_lattices * n_sym_ops
    lower_i = flex.int()
    upper_i = flex.int()
    for lidx in range(self._lattices.size):
      LL,UU = self._lattice_lower_upper_index(lidx)
      lower_i.append(int(LL))
      if UU is None:  UU = self._data.data().size()
      upper_i.append(int(UU))
    indices = {}
    space_group_type = self._data.space_group().type()
    from xfel.merging import compute_rij_wij_detail
    CC = compute_rij_wij_detail(
        lower_i,
        upper_i,
        self._data.data(),
        self._min_pairs)
    for sym_op in self.sym_ops:
        cb_op = sgtbx.change_of_basis_op(sym_op)
        indices_reindexed = cb_op.apply(self._data.indices())
        miller.map_to_asu(space_group_type, False, indices_reindexed)
        indices[cb_op.as_xyz()] = indices_reindexed
        CC.set_indices(cb_op, indices_reindexed)

    assert self._nproc==1
    results = map(
        _compute_one_row,
        [(CC, i, NN) for i in range(n_lattices)]
    )

    rij_matrix = None
    wij_matrix = None
    for (rij, wij) in results:
        if rij_matrix is None:
            rij_matrix = rij
            wij_matrix = wij
        else:
            rij_matrix += rij
            wij_matrix += wij
    self.rij_matrix = rij_matrix.todense().astype(np.float64)
    self.wij_matrix = wij_matrix.todense().astype(np.float64)
    return self.rij_matrix, self.wij_matrix

class TargetWithCustomSymops(TargetWithFastRij):
  def __init__(
      self,
      intensities,
      lattice_ids,
      weights=None,
      min_pairs=3,
      lattice_group=None,
      dimensions=None,
      nproc=None,
      twin_axes=None,
      twin_angles=None,
      cb_op=None,
  ):
    '''
    A couple extra init arguments permit testing user-defined reindexing ops.
    twin_axes is a list of tuples, e.g. [(0,1,0)] means the twin axis is b.
    twin_angles is a corresponding list to define the rotations; 2 is a twofold
        rotation etc.
    cb_op is the previously determined transformation from the input cells to
        the minimum cell. The data have already been transformed by this
        operator, so we transform the twin operators before testing them.
    '''

    if nproc is not None:
      warnings.warn("nproc is deprecated", DeprecationWarning)
    self._nproc = 1

    if weights is not None:
      assert weights in ("count", "standard_error")
    self._weights = weights
    self._min_pairs = min_pairs

    data = intensities.customized_copy(anomalous_flag=False)
    cb_op_to_primitive = data.change_of_basis_op_to_primitive_setting()
    data = data.change_basis(cb_op_to_primitive).map_to_asu()

    # Convert to uint64 avoids crashes on Windows when later constructing
    # flex.size_t (https://github.com/cctbx/cctbx_project/issues/591)
    order = lattice_ids.argsort().astype(np.uint64)
    sorted_data = data.data().select(flex.size_t(order))
    sorted_indices = data.indices().select(flex.size_t(order))
    self._lattice_ids = lattice_ids[order]
    self._data = data.customized_copy(indices=sorted_indices, data=sorted_data)
    assert isinstance(self._data.indices(), type(cctbx_flex.miller_index()))
    assert isinstance(self._data.data(), type(cctbx_flex.double()))

    # construct a lookup for the separate lattices
    self._lattices = np.array(
        [
            np.where(self._lattice_ids == i)[0][0]
            for i in np.unique(self._lattice_ids)
        ]
    )

    self.sym_ops = OrderedSet(["x,y,z"])
    self._lattice_group = lattice_group
    auto_sym_ops = self._generate_twin_operators()
    if twin_axes is not None:
      assert len(twin_axes) == len(twin_angles)
      lds = [literal_description(cb_op.inverse().apply(op)) for op in auto_sym_ops]
      ld_tuples = [(ld.r_info.ev(), ld.r_info.type()) for ld in lds]
      i_symops_to_keep = []
      for i, (axis, angle) in enumerate(ld_tuples):
        if axis in twin_axes and angle in twin_angles:
          i_symops_to_keep.append(i)
      assert len(i_symops_to_keep) == len(twin_axes)
      sym_ops = [auto_sym_ops[i] for i in i_symops_to_keep]
    else:
      sym_ops = auto_sym_ops
    self.sym_ops.update(op.as_xyz() for op in sym_ops)
    if dimensions is None:
      dimensions = max(2, len(self.sym_ops))
    self.set_dimensions(dimensions)

    self._lattice_group = copy.deepcopy(self._data.space_group())
    for sym_op in self.sym_ops:
      self._lattice_group.expand_smx(sym_op)
    self._patterson_group = self._lattice_group.build_derived_patterson_group()

    logger.debug(
        "Lattice group: %s (%i symops)",
        self._lattice_group.info().symbol_and_number(),
        len(self._lattice_group),
    )
    logger.debug(
        "Patterson group: %s", self._patterson_group.info().symbol_and_number()
    )

    self.rij_matrix, self.wij_matrix = self._compute_rij_wij()
