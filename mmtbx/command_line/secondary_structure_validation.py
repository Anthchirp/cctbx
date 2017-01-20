from __future__ import division
# LIBTBX_SET_DISPATCHER_NAME phenix.secondary_structure_validation

from mmtbx.secondary_structure import manager
import iotbx.pdb
import iotbx.phil
from scitbx.array_family import flex
from libtbx.utils import Sorry
import cStringIO
import sys
from libtbx import easy_mp
import mmtbx
from mmtbx.building.loop_closure.utils import get_phi_psi_atoms, get_rama_score, \
    rama_evaluate, get_pair_angles
from libtbx import group_args

import boost.python
ext = boost.python.import_ext("mmtbx_validation_ramachandran_ext")
from mmtbx_validation_ramachandran_ext import rama_eval
from mmtbx.validation import ramalyze

master_phil_str = """
show_all_params = False
  .type = bool
  .style = hidden
nproc = 1
  .type = int
file_name = None
  .type = path
  .multiple = True
  .optional = True
  .style = hidden
"""

master_phil = iotbx.phil.parse(master_phil_str, process_includes=True)

def show_usage():
  help_msg = """\
phenix.secondary_structure_validation: tool for validation of secondary
  structure annotations.

Usage examples:
  phenix.secondary_structure_validation model.pdb
  phenix.secondary_structure_validation model.cif
  phenix.secondary_structure_validation model.pdb nproc=7

Full scope of parameters:
  """
  print help_msg
  master_phil.show()

class gather_ss_stats(object):
  def __init__(self, pdb_h, atoms):
    self.pdb_h = pdb_h
    self.atoms = atoms
    self.r = rama_eval()

  def __call__(self, hsh_tuple):
    temp_annot = iotbx.pdb.secondary_structure.annotation(
        helices = hsh_tuple[0],
        sheets = hsh_tuple[1])
    helix = len(hsh_tuple[0]) > 0
    # print temp_annot.as_pdb_str().replace('\n',' '),
    ss_params = mmtbx.secondary_structure.default_params
    ss_params.secondary_structure.enabled=True
    ss_params.secondary_structure.protein.remove_outliers=False
    ss_params.secondary_structure.protein.helix=[]
    ss_params.secondary_structure.protein.sheet=[]
    ss_params.secondary_structure.nucleic_acid.enabled=False
    ss_m_log = cStringIO.StringIO()

    ss_manager = mmtbx.secondary_structure.manager(
        pdb_hierarchy=self.pdb_h,
        sec_str_from_pdb_file=temp_annot,
        params=ss_params.secondary_structure,
        log = ss_m_log)
    h_bond_proxies = ss_manager.create_protein_hbond_proxies(log=ss_m_log)

    cutoff_bad = 3.5
    cutoff_mediocre = 3.0
    n_hbonds = 0
    n_bad_hbonds = 0
    n_mediocre_hbonds = 0
    hb_lens = []
    for hb_p in h_bond_proxies:
      # print dir(hb_p)
      n_hbonds += 1
      hb_len = self.atoms[hb_p.i_seqs[0]].distance(self.atoms[hb_p.i_seqs[1]])
      hb_lens.append(hb_len)
      if hb_len > cutoff_bad:
        n_bad_hbonds += 1
      elif hb_len > cutoff_mediocre:
        n_mediocre_hbonds += 1

    # Ramachandran outliers and wrong areas
    sele = ss_manager.selection_cache.selection(temp_annot.as_atom_selections()[0])
    ss_h = self.pdb_h.select(sele)
    phi_psi_atoms = get_phi_psi_atoms(ss_h)

    n_outliers = 0
    n_wrong_region = 0
    for phi_psi_pair, rama_key in phi_psi_atoms:
      rama_score = get_rama_score(phi_psi_pair, self.r, rama_key)
      if rama_evaluate(phi_psi_pair, self.r, rama_key) == ramalyze.RAMALYZE_OUTLIER:
        n_outliers += 1
      else:
        reg = gather_ss_stats.helix_sheet_rama_region(phi_psi_pair)
        if (reg == 1 and not helix) or (reg == 2 and helix):
          n_wrong_region += 1
          # print "  Wrong region:", phi_psi_pair[0][2].id_str(), reg, helix

    del ss_manager
    del ss_params
    return n_hbonds, n_bad_hbonds, n_mediocre_hbonds, hb_lens, n_outliers, n_wrong_region

  @classmethod
  def helix_sheet_rama_region(cls,phi_psi_pair):
    # result 1 - helix, 2 - sheet, 0 - other
    # cutoff: phi < 70 - helix, phi>=70 - sheet
    phi_psi_angles = get_pair_angles(phi_psi_pair, round_coords=False)
    if phi_psi_angles[1] < 70:
      return 1
    else:
      return 2

def is_ca_and_something(pdb_h):
  asc = pdb_h.atom_selection_cache()
  n_N = asc.iselection("name N").size()
  n_O = asc.iselection("name O").size()
  n_CA = asc.iselection("name CA").size()
  water = asc.iselection("water").size()
  n_O -= water
  assert n_CA > 0
  if abs(n_CA-n_N)/n_CA > 0.5:
    return True
  if abs(n_CA-n_O)/n_CA > 0.5:
    return True
  return False

def some_chains_are_ca(pdb_h):
  for chain in pdb_h.only_model().chains():
    if chain.is_ca_only():
      return True

def run(args=None, pdb_inp=None, pdb_hierarchy=None, cs=None, nproc=None, params=None,
        out=sys.stdout, log=sys.stderr):
  if(pdb_hierarchy is None):
    assert args is not None
    # params keyword is for running program from GUI dialog
    if ( ((len(args) == 0) and (params is None)) or
         ((len(args) > 0) and ((args[0] == "-h") or (args[0] == "--help"))) ):
      show_usage()
      return
    # parse command-line arguments
    if (params is None):
      pcl = iotbx.phil.process_command_line_with_files(
        args=args,
        master_phil_string=master_phil_str,
        pdb_file_def="file_name")
      work_params = pcl.work.extract()
    # or use parameters defined by GUI
    else:
      work_params = params
    pdb_files = work_params.file_name

    pdb_combined = iotbx.pdb.combine_unique_pdb_files(file_names=pdb_files)
    pdb_structure = iotbx.pdb.input(source_info=None,
      lines=flex.std_string(pdb_combined.raw_records))
    pdb_h = pdb_structure.construct_hierarchy()
  else:
    work_params = master_phil.extract()
    if(nproc is not None): work_params.nproc = nproc
    pdb_h=pdb_hierarchy
    pdb_structure=pdb_inp
  atoms = pdb_h.atoms()
  ss_log = cStringIO.StringIO()
  try:
    ss_annot = pdb_structure.extract_secondary_structure(log=ss_log)
  except Sorry as e:
    print >> out, " Syntax error in SS: %s" % e.message
    return

  ss_log_cont = ss_log.getvalue()
  n_bad_helices = ss_log_cont.count("Bad HELIX")
  n_bad_sheets = ss_log_cont.count("Bad SHEET")
  if ss_annot is None or ss_annot.is_empty():
    print >> out, "No SS annotation, nothing to analyze"
    return
  if n_bad_helices > 0:
    print >> out, "Number of bad helices: %d" % n_bad_helices
  if n_bad_helices > 0:
    print >> out, "Number of bad sheets: %d" % n_bad_sheets
  if len(pdb_h.models()) != 1 :
    raise Sorry("Multiple models not supported.")
  if pdb_h.is_ca_only():
    print >> out, "Error: CA-only model"
    return
  if is_ca_and_something(pdb_h):
    print >> out, "CA-only and something model"
    return
  if some_chains_are_ca(pdb_h):
    print >> out, "some chains are CA-only"
    return

  corrupted_cs = False
  if cs is not None:
    if [cs.unit_cell(), cs.space_group()].count(None) > 0:
      corrupted_cs = True
      cs = None
    elif cs.unit_cell().volume() < 10:
      corrupted_cs = True
      cs = None

  if cs is None:
    if corrupted_cs:
      print >> out, "Symmetry information is corrupted, "
    else:
      print >> out, "Symmetry information was not found, "
    print >> out, "putting molecule in P1 box."
    from cctbx import uctbx
    atoms = pdb_structure.atoms()
    box = uctbx.non_crystallographic_unit_cell_with_the_sites_in_its_center(
      sites_cart=atoms.extract_xyz(),
      buffer_layer=3)
    atoms.set_xyz(new_xyz=box.sites_cart)
    cs = box.crystal_symmetry()

  n_total_helix_sheet_records = len(ss_annot.helices+ss_annot.sheets)
  n_bad_helix_sheet_records = 0
  # Empty stuff:
  empty_annots = ss_annot.remove_empty_annotations(pdb_h)
  number_of_empty_helices = empty_annots.get_n_helices()
  number_of_empty_sheets = empty_annots.get_n_sheets()
  n_bad_helix_sheet_records += (number_of_empty_helices+number_of_empty_sheets)
  if number_of_empty_helices > 0:
    print >> out, "Helices without corresponding atoms in the model (%d):" % number_of_empty_helices
    for h in empty_annots.helices:
      print >> out, "  ", h.as_pdb_str()
  if number_of_empty_sheets > 0:
    print >> out, "Sheets without corresponding atoms in the model (%d):" % number_of_empty_sheets
    for sh in empty_annots.sheets:
      print >> out, "  ", sh.as_pdb_str()

  print >> out, "Checking annotations thoroughly, use nproc=<number> if it is too slow..."

  hsh_tuples = []
  for h in ss_annot.helices:
    hsh_tuples.append(([h],[]))
  for sh in ss_annot.sheets:
    hsh_tuples.append(([],[sh]))
  calc_ss_stats = gather_ss_stats(pdb_h, atoms)
  results = easy_mp.pool_map(
      processes=work_params.nproc,
      fixed_func=calc_ss_stats,
      args=hsh_tuples)

  cumm_n_hbonds = 0
  cumm_n_bad_hbonds = 0
  cumm_n_mediocre_hbonds = 0
  cumm_n_rama_out = 0
  cumm_n_wrong_reg = 0
  #
  # Hydrogen Bonds in Proteins: Role and Strength
  # Roderick E Hubbard, Muhammad Kamran Haider
  # ENCYCLOPEDIA OF LIFE SCIENCES & 2010, John Wiley & Sons, Ltd. www.els.net
  #
  # See also: http://proteopedia.org/wiki/index.php/Hydrogen_bonds
  #
  for ss_elem, r in zip(ss_annot.helices+ss_annot.sheets, results):
    n_hbonds, n_bad_hbonds, n_mediocre_hbonds, hb_lens, n_outliers, n_wrong_region = r
    cumm_n_hbonds += n_hbonds
    cumm_n_bad_hbonds += n_bad_hbonds
    cumm_n_mediocre_hbonds += n_mediocre_hbonds
    cumm_n_rama_out += n_outliers
    cumm_n_wrong_reg += n_wrong_region
    if n_bad_hbonds + n_outliers + n_wrong_region > 0:
      n_bad_helix_sheet_records += 1
    if n_bad_hbonds + n_mediocre_hbonds + n_outliers + n_wrong_region > 0:
      # this is bad annotation, printing it to log with separate stats:
      print >> out, "Bad annotation found:"
      print >> out, "  %s" % ss_elem.as_pdb_str()
      print >> out, "  Total hb: %d, mediocre: %d, bad: %d, Rama outliers: %d, Rama wrong %d" % (
          n_hbonds, n_mediocre_hbonds, n_bad_hbonds, n_outliers, n_wrong_region)
      print >> out, "-"*80

  # for r in results:
  #   cumm_n_hbonds += r[0]
  #   cumm_n_bad_hbonds += r[1]
  #   cumm_n_mediocre_hbonds += r[2]
  #   cumm_n_rama_out += r[4]
  #   cumm_n_wrong_reg += r[5]
  #   print >> individ_log, "%d, %d, %d, %s, %d, %d" % (r[0], r[1], r[2], r[3], r[4], r[5])
  print >> out, "Overall info:"
  print >> out, "  Total HELIX+SHEET recods       :", n_total_helix_sheet_records
  print >> out, "  Total bad HELIX+SHEET recods   :", n_bad_helix_sheet_records
  print >> out, "  Total declared H-bonds         :", cumm_n_hbonds
  print >> out, "  Total mediocre H-bonds (3-3.5A):", cumm_n_mediocre_hbonds
  print >> out, "  Total bad H-bonds (> 3.5A)     :", cumm_n_bad_hbonds
  print >> out, "  Total Ramachandran outliers    :", cumm_n_rama_out
  print >> out, "  Total wrong Ramachandrans      :", cumm_n_wrong_reg
  return group_args(
    n_total_helix_sheet_records = n_total_helix_sheet_records,
    n_bad_helix_sheet_records   = n_bad_helix_sheet_records,
    n_hbonds                    = cumm_n_hbonds,
    n_mediocre_hbonds           = cumm_n_mediocre_hbonds,
    n_bad_hbonds                = cumm_n_bad_hbonds,
    n_rama_out                  = cumm_n_rama_out,
    n_wrong_reg                 = cumm_n_wrong_reg)
  print >> out, "All done."

if __name__ == "__main__" :
  run(sys.argv[1:])
