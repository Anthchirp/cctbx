##################################################################################
#                Copyright 2021  Richardson Lab at Duke University
# 
# Licensed under the Apache License, Version 2.0 (the "License"],
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##################################################################################
# This module exports functions that are helpful to create data structures
# needed by Probe.

import sys
import iotbx.map_model_manager
import iotbx.data_manager
import cctbx.maptbx.box
import mmtbx
import mmtbx_probe_ext as probe
from scitbx.array_family import flex
import mmtbx.probe.AtomTypes

def getBondedNeighborLists(atoms, bondProxies):
  """
    Helper function to produce a dictionary of lists that contain all bonded
    neighbors for each atom in a set of atoms.
    :param atoms: Flex array of atoms (could be obtained using model.get_atoms() if there
    are no chains with multiple conformations, must be a subset of the atoms including
    all in the base conformation and in a particular conformation otherwise).
    :param bondProxies: Flex array of bond proxies for the atoms.  This could be obtained
    using model.get_restraints_manager().geometry.get_all_bond_proxies(sites_cart =
    model.get_sites_cart())[0] if the model has only a single conformation.  Otherwise,
    it should be a flex array of atom positions for the atoms that are in the first argument.
    :returns a dictionary with one entry for each atom that contains a list of all of
    the atoms that are bonded to it.
  """
  atomDict = {}
  for a in atoms:
    atomDict[a.i_seq] = a
  bondedNeighbors = {}
  for a in atoms:
    bondedNeighbors[a] = []
  for bp in bondProxies:
    bondedNeighbors[atomDict[bp.i_seqs[0]]].append(atomDict[bp.i_seqs[1]])
    bondedNeighbors[atomDict[bp.i_seqs[1]]].append(atomDict[bp.i_seqs[0]])
  return bondedNeighbors

def getAtomsWithinNBonds(atom, bondedNeighborLists, N):
  """
    Helper function to produce a list of all of the atoms that are bonded to the
    specified atoms, or to one of the atoms bonded to the specified atom, recursively
    to a depth of N.  The atom itself will not be included in the list, so an atom that
    has no bonded neighbors will always have an empty result.  This can be used to
    produce a list of excluded atoms for dot scoring.
    :param atom: The atom to be tested.
    :param bondedNeighborLists: Dictionary of lists that contain all bonded neighbors for
    each atom in a set of atoms.  Should be obtained using
    mmtbx.probe.Helpers.getBondedNeighborLists().
    :param N: Depth of recursion.  N=1 will return the atoms bonded to atom.  N=2 will
    also return those bonded to these neighbors (but not the atom itself).
    :returns a list of all atoms that are bonded to atom within a depth of N.  The original
    atom is never on the list.
  """
  # Find all atoms to the specified depth
  atoms = {atom}            # Initialize the set with the atom itself
  for i in range(N):        # Repeat the recursion this many times
    current = list(atoms)   # Make a copy so we're not modifying the list we are traversing
    for a in current:       # Add all neighbors of all atoms in the current level
      for n in bondedNeighborLists[a]:
        atoms.add(n)

  # Remove the original atom from the result and turn the result into a list.
  atoms.discard(atom)
  return list(atoms)

class getExtraAtomInfoReturn:
  """
    Return type from getExtraAtomInfo() call.
      extraAtomInfo: ExtraAtomInfoMap with an entry for every atom in the model suitable for
                     passing to the scoring functions.
      warnings: a string that if not empty lists warnings that the person running the program
                might want to know about.  Suitable for printing or logging.
  """
  def __init__(self, extraAtomInfo, warnings):
    self.extraAtomInfo = extraAtomInfo
    self.warnings = warnings

def getExtraAtomInfo(model, useNeutronDistances = False):
  """
    Helper function to provide a mapper for ExtraAtomInfo needed by Probe when scoring
    models.  It first tries to find the information in CCTBX.  If it cannot, it looks
    the information up using the original C-code Probe tables and algorithms.
    :param model: Map Model Manager's Model containing all of the atoms to be described.
    PDB interpretation must have been done on the model, perhaps by calling
    model.process_input_model(make_restraints=True), with useNeutronDistances matching
    the parameter to this function.
    :param useNeutronDistances: Default is to use x-ray distances, but setting this to
    True uses neutron distances instead.  This must be set consistently with the
    PDB interpretation parameter used on the model.
    Can be obtained by calling iotbx.map_model_manager.map_model_manager().model().
    :returns a ExtraAtomInfoMap with an entry for every atom in the model suitable for
    passing to the scoring functions.
  """

  warnings = ""

  # Construct the AtomTypes object we're going to use, telling it whether to use neutron distances.
  at = mmtbx.probe.AtomTypes.AtomTypes(useNeutronDistances)

  # Traverse the hierarchy and look up the extra data to be filled in.
  extras = probe.ExtraAtomInfoMap([],[])
  mon_lib_srv = model.get_mon_lib_srv()
  ener_lib = mmtbx.monomer_library.server.ener_lib()
  ph = model.get_hierarchy()
  for m in ph.models():
    for chain in m.chains():
      for rg in chain.residue_groups():
        for ag in rg.atom_groups():
          md, ani = mon_lib_srv.get_comp_comp_id_and_atom_name_interpretation(
                residue_name=ag.resname, atom_names=ag.atoms().extract_name())
          atom_dict = md.atom_dict()

          for a in ag.atoms():
            extra = probe.ExtraAtomInfo()
            # See if we can find out about its Hydrogen-bonding status from the
            # model.  If so, we fill it and the vdwRadius information from
            # CCTBX.
            try:
              hb_type = model.get_specific_h_bond_type(a)
              if isinstance(hb_type, str):
                if hb_type == "A" or hb_type == "B":
                  extra.isAcceptor = True
                if hb_type == "D" or hb_type == "B":
                  extra.isDonor = True

                # For metallic atoms, the Richardsons determined in discussion with
                # Michael Prisant that we want to use the ionic radius rather than the
                # larger radius for all purposes.
                # @todo If CCTBX starts returning ionic radius as VdW radius, then we
                # can remove this check and always just use get_specific_vdw_radii()
                if isMetallic(a):
                  # @todo Replace this with CCTBX ionic radius function when Nigel adds it
                  extra.vdwRadius = at.FindProbeExtraAtomInfo(a).vdwRadius
                else:
                  extra.vdwRadius = model.get_specific_vdw_radii(a)

                # Mark aromatic ring N and C atoms as acceptors as a hack to enable the
                # ring itself to behave as an acceptor.
                # @todo Remove this once we have a better way to model the ring itself
                # as an acceptor, perhaps making it a cylinder or a sphere in the center
                # of the ring.
                if a.element in ['C','N']:
                  if mmtbx.probe.AtomTypes.IsAromaticAcceptor(ag.resname, a.name):
                    extra.IsAcceptor = True

                extras.setMappingFor(a, extra)
                continue

              # Did not find the information from CCTBX, so look it up using
              # the original Probe approach by dropping through to below
              else:
                warnings += "Could not find "+a.name+" in CCTBX, using Probe tables\n"
            except:
              # Warn and drop through to below.
              warnings += ("Could not look up "+a.name+" in CCTBX "+
                "(perhaps interpretation was not run on the model?), using Probe tables\n")

            # Did not find what we were looking for in CCTBX, so drop through to Probe
            extra, warn = at.FindProbeExtraAtomInfo(a)
            if len(warn) > 0:
              warnings += "  Probe says: "+warn+"\n"

            extras.setMappingFor(a, extra)

  return getExtraAtomInfoReturn(extras, warnings)

def isMetallic(atom):
  """
    Helper function to report whether a given atom is metallic.
    :param atom: iotbx.pdb.hierarchy.atom to check.
    :returns True if the atoms is metallic, False if it is not.  Bases this on
    the mmtbx.probe.AtomTypes _AtomTable.
  """
  # See if we've already made the set to look these up in to save time.
  element = atom.element.upper()
  try:
    return element in isMettalic.metallics
  except:
    # Build the set by filling in all of the entries in the atom table.
    at = mmtbx.probe.AtomTypes.AtomTypes()
    isMetallic.metallics = set()
    for e in at._AtomTable:
      if e[8] & mmtbx.probe.AtomTypes.AtomFlags.METALLIC_ATOM:
        isMetallic.metallics.add(e[1].upper())
    return element in isMetallic.metallics

def Test(inFileName = None):

  #========================================================================
  # Run unit test on getExtraAtomInfo().
  # @todo

  # Spot check that we're getting ionic radii for metals.
  # @todo

  #========================================================================
  # Run unit test on getBondedNeighborLists().  We use a specific PDB snippet
  # for which we know the answer and then we verify that the results are what
  # we expect.
  pdb_1xso_his_61 = (
"""
ATOM    442  N   HIS A  61      26.965  32.911   7.593  1.00  7.19           N  
ATOM    443  CA  HIS A  61      27.557  32.385   6.403  1.00  7.24           C  
ATOM    444  C   HIS A  61      28.929  31.763   6.641  1.00  7.38           C  
ATOM    445  O   HIS A  61      29.744  32.217   7.397  1.00  9.97           O  
ATOM    446  CB  HIS A  61      27.707  33.547   5.385  1.00  9.38           C  
ATOM    447  CG  HIS A  61      26.382  33.956   4.808  1.00  8.78           C  
ATOM    448  ND1 HIS A  61      26.168  34.981   3.980  1.00  9.06           N  
ATOM    449  CD2 HIS A  61      25.174  33.397   5.004  1.00 11.08           C  
ATOM    450  CE1 HIS A  61      24.867  35.060   3.688  1.00 12.84           C  
ATOM    451  NE2 HIS A  61      24.251  34.003   4.297  1.00 11.66           N  
END
"""
    )

  dm = iotbx.data_manager.DataManager(['model'])
  dm.process_model_str("1xso_snip.pdb",pdb_1xso_his_61)
  model = dm.get_model()
  model.process_input_model(make_restraints=True) # make restraints

  # Get the first model in the hierarchy.
  atoms = model.get_hierarchy().models()[0].atoms()

  # Get the Cartesian positions of all of the atoms we're considering for this alternate
  # conformation.
  carts = flex.vec3_double()
  for a in atoms:
    carts.append(a.xyz)

  # Get the bond proxies for the atoms in the model and conformation we're using and
  # use them to determine the bonded neighbor lists.
  bondProxies = model.get_restraints_manager().geometry.get_all_bond_proxies(sites_cart = carts)[0]
  bondedNeighborLists = getBondedNeighborLists(atoms, bondProxies)

  # Check the counts in the neighbor lists to make sure they match what we expect
  neighborCounts = {"N": 1, "CA": 3, "C": 2, "O": 1, "CB": 2,
                    "CG": 3, "ND1": 2, "CD2": 2, "CE1":2, "NE2": 2}
  for a in atoms:
    if len(bondedNeighborLists[a]) != neighborCounts[a.name.strip()]:
      return ("Helpers.Test(): Neighbor count for "+a.name.strip()+" was "+
        str(len(bondedNeighborLists[a]))+", expected "+str(neighborCounts[a.name.strip()]))

  #========================================================================
  # Run unit test on getAtomsWithinNBonds().
  # Get the atoms within N bounds for a range for the "N" atom and verify that the
  # counts match what is expected.
  # NOTE: This re-uses the bondedNeighborLists test results from above
  nestedNeighborsForN = [ None, 1, 3, 5, 7, 9, 9]
  for N in range(1,7):
    count = len(getAtomsWithinNBonds(atoms[0], bondedNeighborLists, N))
    if count != nestedNeighborsForN[N]:
      return ("Helpers.Test(): Nested count for "+atoms[0].name.strip()+
        " for N = "+str(N)+" was "+str(count)+", expected "+str(nestedNeighborsForN[N]))


  #========================================================================
  # Generate an example data model with a small molecule in it or else read
  # from the specified file.
  if inFileName is not None and len(inFileName) > 0:
    # Read a model from a file using the DataManager
    print('Reading model from',inFileName)
    dm = iotbx.data_manager.DataManager()
    dm.process_model_file(inFileName)
    model = dm.get_model(inFileName)
  else:
    # Generate a small-molecule model using the map model manager
    print('Generating model')
    # get an initialized instance of the map_model_manager
    mmm=iotbx.map_model_manager.map_model_manager()
    mmm.generate_map()     #   get a model from a generated small library model and calculate a map for it
    model = mmm.model()    #   get the model

  # Fix up bogus unit cell when it occurs by checking crystal symmetry.
  cs =model.crystal_symmetry()
  if (cs is None) or (cs.unit_cell() is None):
    model = cctbx.maptbx.box.shift_and_box_model(model = model)

  # Run PDB interpretation on the model to fill in the required CCTBX information.
  print('Interpreting model')
  p = mmtbx.model.manager.get_default_pdb_interpretation_params()
  p.pdb_interpretation.use_neutron_distances = False
  model.set_pdb_interpretation_params(params = p)
  model.process_input_model(make_restraints=True) # make restraints

  print('Getting extraAtomInfo')
  ret = getExtraAtomInfo(model)
  if len(ret.warnings) > 0:
    print('Warnings returned by getExtraAtomInfo():\n'+ret.warnings)

  # Run spot checks on isMetallic()
  a = iotbx.pdb.hierarchy.atom()
  a.element = "Li"
  if not isMetallic(a):
    return "Helpers.Test(): Lithium not listed as metallic"
  a.element = "He"
  if isMetallic(a):
    return "Helpers.Test(): Helium listed as metallic"

  return ""

if __name__ == '__main__':

  #==============================================================
  # Parse command-line arguments.  The 0th argument is the name
  # of the script. There can be the name of a PDB/CIF file to read.
  realParams = 0
  fileName = ""
  for i in range(1,len(sys.argv)):
    fileName = sys.argv[i]

  ret = Test(fileName)
  if len(ret) == 0:
    print('Success!')
  else:
    print(ret)

  assert (len(ret) == 0)