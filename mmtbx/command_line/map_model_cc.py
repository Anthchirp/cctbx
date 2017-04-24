from __future__ import division
# LIBTBX_SET_DISPATCHER_NAME phenix.map_model_cc
# LIBTBX_SET_DISPATCHER_NAME phenix.model_map_cc

import sys
import iotbx.pdb
from libtbx import group_args
from libtbx.utils import Sorry
import mmtbx.utils
import mmtbx.maps.map_model_cc

master_params_str = """\
  map_file_name = None
    .type = str
    .help = Map file name
  model_file_name = None
    .type = str
    .help = Model file name
  include scope mmtbx.maps.map_model_cc.master_params
"""

def master_params():
  return iotbx.phil.parse(master_params_str, process_includes=True)


def broadcast(m, log):
  print >> log, "-"*79
  print >> log, m
  print >> log, "*"*len(m)

def get_inputs(args,
               log,
               master_params,
               need_map=True,
               need_model_hierarchy=True,
               need_crystal_symmetry=True):
  """
  Eventually, this will be centralized.
  """
  inputs = mmtbx.utils.process_command_line_args(
    args          = args,
    master_params = master_params)
  # Model
  pdb_file_name, pdb_hierarchy = None, None
  if(need_model_hierarchy):
    file_names = inputs.pdb_file_names
    if(len(file_names) != 1):
      raise Sorry("One model (PDB or mmCIF) required.")
    pdb_file_name = file_names[0]
    pdb_hierarchy = iotbx.pdb.input(
      file_name = pdb_file_name).construct_hierarchy()
  # Map
  ccp4_map_object = None
  if(need_map):
    if(inputs.ccp4_map is None):
      raise Sorry("Map file has to given.")
    ccp4_map_object = inputs.ccp4_map
  # Crystal symmetry
  crystal_symmetry = None
  if(need_crystal_symmetry):
    crystal_symmetry = inputs.crystal_symmetry
    if(crystal_symmetry is None):
      raise Sorry("No box (unit cell) info found.")
  #
  return group_args(
    params           = inputs.params.extract(),
    pdb_file_name    = pdb_file_name,
    pdb_hierarchy    = pdb_hierarchy,
    ccp4_map_object  = ccp4_map_object,
    crystal_symmetry = crystal_symmetry)

def run(args, log=sys.stdout):
  """phenix.map_model_cc or phenix.model_map_cc:
  Given PDB file and a map file calculate model-map coorelation.

How to run:
  phenix.map_model_cc model.pdb map.ccp4 resolution=3
  phenix.model_map_cc model.pdb map.ccp4 resolution=3

Feedback:
  PAfonine@lbl.gov
  phenixbb@phenix-online.org
  """
  assert len(locals().keys()) == 2 # intentional
  print >> log, "-"*79
  print >> log, run.__doc__
  print >> log, "-"*79
  # Get inputs
  inputs = get_inputs(
    args          = args,
    log           = log,
    master_params = master_params())
  # Model
  broadcast(m="Input PDB:", log=log)
  print >> log, inputs.pdb_file_name # ideally this should not be available here
  inputs.pdb_hierarchy.show(level_id="chain")
  # Crystal symmetry
  broadcast(m="Box (unit cell) info:", log=log)
  inputs.crystal_symmetry.show_summary(f=log)
  # Map
  broadcast(m="Input map:", log=log)
  inputs.ccp4_map_object.show_summary(prefix="  ")
  # Run task in 4 separate steps
  task_obj = mmtbx.maps.map_model_cc.map_model_cc(
    map_data         = inputs.ccp4_map_object.map_data(),
    pdb_hierarchy    = inputs.pdb_hierarchy,
    crystal_symmetry = inputs.crystal_symmetry,
    params           = inputs.params.map_model_cc)
  task_obj.validate()
  task_obj.run()
  results = task_obj.get_results()
  #
  broadcast(m="Map resolution:", log=log)
  print >> log, "  Resolution:", results.resolution
  broadcast(m="Map-model CC (overall):", log=log)
  print >> log, "  CC_mask  : %6.4f"%results.cc_mask
  print >> log, "  CC_volume: %6.4f"%results.cc_volume
  print >> log, "  CC_peaks : %6.4f"%results.cc_peaks
  broadcast(m="Model-map FSC:", log=log)
  results.fsc.show(prefix="  ")
  broadcast(m="Map-model CC (local):", log=log)
  # Per chain
  print >> log, "Per chain:"
  fmt = "%s %7.4f %8.3f %4.2f %d"
  for r in results.cc_per_chain:
    print fmt%(r.chain_id, r.cc, r.b_iso_mean, r.occ_mean, r.n_atoms)
  # Per residue
  print >> log, "Per residue:"
  fmt = "%s %s %s %7.4f %8.3f %4.2f"
  for r in results.cc_per_residue:
    print fmt%(r.chain_id, r.resname, r.resseq, r.cc, r.b_iso_mean, r.occ_mean)
  return None

if (__name__ == "__main__"):
  assert run(args=sys.argv[1:]) is None # assert here is intentional
