##################################################################################
#                Copyright 2021  Richardson Lab at Duke University
# 
# Licensed under the Apache License, Version 2.0 (the "License");
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

import argparse

import Movers
import InteractionGraph
from boost_adaptbx.graph import connected_component_algorithm as cca

from iotbx.map_model_manager import map_model_manager
from iotbx.data_manager import DataManager
import mmtbx
from scitbx.array_family import flex

from mmtbx.probe import AtomTypes
from mmtbx.probe import Helpers
import mmtbx_probe_ext as probeExt

# To enable addition of Hydrogens
# @todo See if we can remove the shift and box once reduce_hydrogen is complete
from cctbx.maptbx.box import shift_and_box_model
from mmtbx.hydrogens import reduce_hydrogen

##################################################################################
# This file includes a set of functions and classes that implement placement and optimization of
# Reduce's "Movers".

##################################################################################
# Module-scoped attributes that can be set to modify behavior.

# Default value of 1 reports standard information.
# Setting it to 0 removes all inforamtional messages.
# Setting it above 1 provides additional debugging information.
verbosity = 1
def _VerboseCheck(level, message):
  # Returns "" if the level is less than global verbosity level, message if it is at or above
  if verbosity >= level:
    return message
  else:
    return ""

##################################################################################
# Optimizers:
#     NullOptimer: This is a base Optimizer class that implements the placement and scoring
# of Movers but does not actually perform any optimization.  The useful Optimizers will all
# be derived from this base class.
#     BruteForceOptimizer: This tries all possible coarse and fine orientations of all
# Movers in all combinations.  It is too slow for any but very small numbers of Movers but it
# is guaranteed to find the minimum score across all combinations.  It is used mainly to
# provide results for test comparisons with the other optimizers.
#     CliqueOptimizer: This produces an InteractionGraph telling which Movers might possibly
# interact during their motion and runs brute-force optimization on each subgraph independently.
# This greatly reduces the number of calculations but is still much to slow for many molecules
# that have a large number of interacting Movers in their largest clique.
#     HyperGraphOptimizer: This uses a divide-and-conquer, dynamic-programming approach to break
# each clique into separate non-interacting subgraphs so that it only has to compute complete
# interactions for a small number of interacting Movers at a time.

class NullOptimizer:

  def __init__(self, model, modelIndex = 0, altID = "",
                bondedNeighborDepth = 3,
                probeRadius = 0.25,
                useNeutronDistances = False
              ):
    """Constructor for NullOptimizer.
    :param model: iotbx model (a group of hierarchy models).  Can be obtained using
    iotbx.map_model_manager.map_model_manager.model().  The model must have Hydrogens,
    which can be added using mmtbx.hydrogens.reduce_hydrogen.place_hydrogens().get_model().
    It must have a valid unit cell, which can be helped by calling
    cctbx.maptbx.box.shift_and_box_model().  It must have had PDB interpretation run on it,
    which can be done using model.process_input_model(make_restraints=True) with PDB
    interpretation parameters and hydrogen placement matching the value of the
    useNeutronDistances parameter described below.
    :param modelIndex: Identifies which index from the hierarchy is to be selected.
    If this value is None, optimization will be run sequentially on every model in the
    hierarchy.
    :param altID: The conformer alternate location specifier to use.  The value "" will
    cause it to run on the first conformer found in each model.  If this is None, optimization
    will be run sequentially for every conformer in the model, starting with the last and
    ending with "".  This will leave the initial conformer's values as the final location
    for atoms that are not inside a conformer.
    :param bondedNeighborDepth: How many hops to ignore bonding when doing Probe calculations.
    The default is to ignore interactions to a depth of 3 (my bonded neighbors and their bonded
    neighbors and their bonded neighbors).
    :param probeRadius: Radius of the probe to be used in Probe calculations (Angstroms).
    :param useNeutronDistances: Defaults to using X-ray/electron cloud distances.  If set to
    True, it will use neutron (nuclear) distances.  This must be set consistently with the
    values used to generate the hydrogens and to run PDB interpretation.
    """

    ################################################################################
    # Initialize my information string to empty.
    self._infoString = ""

    ################################################################################
    # Run optimization for every desired conformer and every desired model, calling
    # placement and then a derived-class single optimization routine for each.  When
    # the modelIndex or altID is None, that means to run over all available cases.
    # For alternates, if there is a non-empty ("" or " ") case, then we run backwards
    # from the last to the first but do not run for the empty case; if there are only
    # empty cases, then we run just once.  We run the models in order, all of them when
    # None is specified and the specified one if it is specified.

    startModelIndex = 0
    stopModelIndex = len(model.get_hierarchy().models())
    if modelIndex is not None:
      startModelIndex = modelIndex
      stopModelIndex = modelIndex + 1
    for mi in range(startModelIndex, stopModelIndex):
      # Get the specified model from the hierarchy.
      myModel = model.get_hierarchy().models()[mi]

      # Get the list of alternate conformation names present in all chains for this model.
      # If there is more than one result, remove the empty results and then sort them
      # in reverse order.
      alts = AlternatesInModel(myModel)
      if len(alts) > 1:
        alts.discard("")
        alts.discard(" ")
      alts = sorted(list(alts), reverse=True)

      # If there is a specified alternate, use it.
      if altID is not None:
        alts = [altID]

      for ai in alts:

        # Tell about the run we are currently doing.
        self._infoString += _VerboseCheck(1,"Running Reduce optimization on model index "+str(mi)+
          ", alternate '"+ai+"'\n")
        self._infoString += _VerboseCheck(1,"  bondedNeighborDepth = "+str(bondedNeighborDepth)+"\n")
        self._infoString += _VerboseCheck(1,"  probeRadius = "+str(probeRadius)+"\n")
        self._infoString += _VerboseCheck(1,"  useNeutronDistances = "+str(useNeutronDistances)+"\n")

        # Get the atoms from the specified conformer in the model (the empty string is the name
        # of the first conformation in the model; if there is no empty conformation, then it will
        # pick the first available conformation for each atom group.
        atoms = GetAtomsForConformer(myModel, ai)

        ################################################################################
        # Get the Cartesian positions of all of the atoms we're considering for this alternate
        # conformation.
        carts = flex.vec3_double()
        for a in atoms:
          carts.append(a.xyz)

        ################################################################################
        # Get the bond proxies for the atoms in the model and conformation we're using and
        # use them to determine the bonded neighbor lists.
        bondProxies = model.get_restraints_manager().geometry.get_all_bond_proxies(sites_cart = carts)[0]
        bondedNeighborLists = Helpers.getBondedNeighborLists(atoms, bondProxies)

        ################################################################################
        # Placement of water phantom Hydrogens, including adding them to our 'atoms' list but
        # not adding them to the hierarchy.  This must be done after the bond proxies are
        # constructed but before making the spatial-query data structure.  They should be
        # placed in an atom group that indicates that they are waters, but that group should
        # not be inserted into the hierarchy.
        # @todo

        ################################################################################
        # Construct the spatial-query information needed to quickly determine which atoms are nearby
        sq = probeExt.SpatialQuery(atoms)

        ################################################################################
        # Determine excluded atoms to a specified hop count for each atom that will
        # be moved.
        # @todo

        ################################################################################
        # Get the probeExt.ExtraAtomInfo needed to determine which atoms are potential acceptors.
        # @todo Ensure that waters are marked as acceptors and their phantom Hydrogens as donors.
        ret = Helpers.getExtraAtomInfo(model)
        extra = ret.extraAtomInfo
        self._infoString += ret.warnings

        ################################################################################
        # Test getting the list of Movers using the _PlaceMovers private function.
        ret = _PlaceMovers(atoms, model.rotatable_hd_selection(iselection=True),
                           bondedNeighborLists, sq, extra, AtomTypes.AtomTypes().MaximumVDWRadius())
        self._infoString += ret.infoString
        movers = ret.moverList
        self._infoString += _VerboseCheck(1,"Inserted "+str(len(movers))+" Movers\n")
        self._infoString += _VerboseCheck(1,'Marked '+str(len(ret.deleteAtoms))+' atoms for deletion\n')

        ################################################################################
        # Compute the interaction graph, of which each connected component is a Clique.
        # Get a list of singleton Cliques and a list of other Cliques.  Keep separate lists
        # of the singletons and the groups.
        interactionGraph = InteractionGraph.InteractionGraphAllPairs(movers, extra)
        components = cca.connected_components( graph = interactionGraph )
        maxLen = 0
        singletonCliques = []
        groupCliques = []
        for c in components:
          if len(c) == 1:
            singletonCliques.append(c)
          else:
            groupCliques.append(c)
          if len(c) > maxLen:
            maxLen = len(c)
        self._infoString += _VerboseCheck(1,"Found "+str(len(components))+" Cliques, "+
            str(len(singletonCliques))+" singletons; largest Clique size = "+
            str(maxLen)+"\n")

        ################################################################################
        # Call internal methods to optimize the single-element Cliques and then to optimize
        # the multi-element Cliques and then to do indepenedent fine adjustment of all
        # Cliques.  Subclasses should overload the called routines, but the global approach
        # taken here will be the same for all of them.  If we want to change the recipe
        # so that we can do global fine optimization, we'll do that here rather than in the
        # subclasses.
        # @todo

        ################################################################################
        # @todo Deletion of Hydrogens that were requested by Histidine FixUp()s.

  def getInfo(self):
    """
      Returns information that the user may care about regarding the processing.  The level of
      detail on this can be set by setting Optimizers.verbosity before creating or calling methods.
      :returns the information so far collected in the string.  Calling this function also clears
      the information, so that later calls will not repeat it.
    """
    ret = self._infoString
    self._infoString = ""
    return ret

##################################################################################
# Placement

class _PlaceMoversReturn:
  # Return type from PlaceMovers() call.  List of movers and then an information string
  # that may contain information the user would like to know (where Movers were placed,
  # failed Mover placements due to missing Hydrogens, etc.).  Also returns a list of
  # atoms that should be deleted as a result of situations determined during the
  # placement.
  def __init__(self, moverList, infoString, deleteAtoms):
    self.moverList = moverList
    self.infoString = infoString
    self.deleteAtoms = deleteAtoms

def _PlaceMovers(atoms, rotatableHydrogenIDs, bondedNeighborLists, spatialQuery, extraAtomInfo,
                  maxVDWRadius):
  """Produce a list of Movers for atoms in a pdb.hierarchy.conformer that has added Hydrogens.
  :param atoms: flex array of atoms to search.  This must have all Hydrogens needed by the
  Movers present in the structure already.
  :param rotateableHydrogenIDs: List of sequence IDs for single hydrogens that are rotatable.
  :param bondedNeighborLists: A dictionary that contains an entry for each atom in the
  structure that the atom from the first parameter interacts with that lists all of the
  bonded atoms.  Can be obtained by calling mmtbx.probe.Helpers.getBondedNeighborLists().
  :param spatialQuery: Probe.SpatialQuery structure to rapidly determine which atoms
  are within a specified distance of a location.
  :param extraAtomInfo: Probe.ExtraAtomInfo mapper that provides radius and other
  information about atoms beyond what is in the pdb.hierarchy.  Used here to determine
  which atoms may be acceptors.
  :param maxVDWRadius: Maximum VdW radius of an atom.  Can be obtained from
  mmtbx.probe.AtomTypes.AtomTypes().MaximumVDWRadius()
  :returns _PlaceMoversReturn giving the list of Movers found in the conformation and
  an error string that is empty if no errors are found during the process and which
  has a printable message in case one or more errors are found.
  """

  # List of Movers to return
  movers = []

  # List of atoms to delete
  deleteAtoms = []

  # Information string to return, filled in when we cannot place a Mover.
  infoString = ""

  # The radius of a Polar Hydrogen (one that forms Hydrogen bonds)
  # @todo Can we look this up from inside CCTBX?
  polarHydrogenRadius = 1.05

  # For each atom, check to see if it is the main atom from a Mover.  If so, attempt to
  # construct the appropriate Mover.  The construction may fail because not all required
  # atoms are present (especially the Hydrogens).  If the construction succeeds, add the
  # Mover to the list of those to return.  When they fail, add the output to the information
  # string.

  for a in atoms:
    # Find the stripped upper-case atom and residue names and the residue ID,
    # and an identifier string
    aName = a.name.strip().upper()
    resName = a.parent().resname.strip().upper()
    resID = str(a.parent().parent().resseq_as_int())
    resNameAndID = resName+" "+resID

    # See if we should construct a MoverSingleHydrogenRotator here.
    # @todo This is placing on atoms C and N in CYS 352; and on HE2 and CA in SER 500 of 4z4d
    if a.i_seq in rotatableHydrogenIDs:
      try:
        # Skip Hydrogens that are not bound to an atom that only has a single other
        # atom bound to it -- we will handle those in other cases.
        neighbor = bondedNeighborLists[a][0]
        if len(bondedNeighborLists[neighbor]) == 2:

          # Construct a list of nearby atoms that are potential acceptors.
          potentialAcceptors = []

          # Get the list of nearby atoms.  The center of the search is the atom that
          # the Hydrogen is bound to and its radius is 4 (these values are pulled from
          # the Reduce C++ code).
          maxDist = 4.0
          nearby = spatialQuery.neighbors(neighbor.xyz, extraAtomInfo.getMappingFor(neighbor).vdwRadius, maxDist)

          # Check each nearby atom to see if it distance from the neighbor is within
          # the sum of the hydrogen-bond length of the neighbor atom, the radius of
          # a polar Hydrogen, and the radius of the nearby atom, indicating potential
          # overlap.
          # O-H & N-H bond len == 1.0, S-H bond len == 1.3
          XHbondlen = 1.0
          if neighbor.element == "S":
            XHbondlen = 1.3
          candidates = []
          for n in nearby:
            d = (Movers._rvec3(neighbor.xyz) - Movers._rvec3(n.xyz)).length()
            if d <= XHbondlen + extraAtomInfo.getMappingFor(n).vdwRadius + polarHydrogenRadius:
              candidates.append(n)

          # See if each nearby atom is a potential acceptor or a flip partner from
          # Histidine or NH2 flips (which may be moved into that atom's position during a flip).
          # We check the partner (N's for GLN And ASN, C's for HIS) because if it is the O or
          # N atom, we will already be checking it as an acceptor now.
          for c in candidates:
            cName = c.name.strip().upper()
            resName = c.parent().resname.strip().upper()
            flipPartner = (
              (cName == 'ND2' and resName == 'ASN') or
              (cName == 'NE2' and resName == 'GLN') or
              (cName == 'CE1' and resName == 'HIS') or (cName == 'CD2' and resName == 'HIS') )
            acceptor = extraAtomInfo.getMappingFor(c).isAcceptor
            if acceptor or flipPartner:
              potentialAcceptors.append(c)

          movers.append(Movers.MoverSingleHydrogenRotator(a, bondedNeighborLists, potentialAcceptors))
          infoString += _VerboseCheck(1,"Added MoverSingleHydrogenRotator to "+resNameAndID+" "+aName+
            " with "+str(len(potentialAcceptors))+" potential nearby acceptors\n")
      except Exception as e:
        infoString += _VerboseCheck(0,"Could not add MoverSingleHydrogenRotator to "+resNameAndID+" "+aName+": "+str(e)+"\n")

    # See if we should construct a MoverNH3Rotator here.
    # Find any Nitrogen that has four total bonded neighbors, three of which are Hydrogens.
    # @todo Is this a valid way to search for them?
    if a.element == 'N' and len(bondedNeighborLists[a]) == 4:
      numH = 0
      for n in bondedNeighborLists[a]:
        if n.element == "H":
          numH += 1
      if numH == 3:
        try:
          movers.append(Movers.MoverNH3Rotator(a, bondedNeighborLists))
          infoString += _VerboseCheck(1,"Added MoverNH3Rotator to "+resNameAndID+"\n")
        except Exception as e:
          infoString += _VerboseCheck(0,"Could not add MoverNH3Rotator to "+resNameAndID+": "+str(e)+"\n")

    # See if we should construct a MoverAromaticMethylRotator or MoverTetrahedralMethylRotator here.
    # Find any Carbon that has four total bonded neighbors, three of which are Hydrogens.
    # @todo Is this a valid way to search for them?
    if a.element == 'C' and len(bondedNeighborLists[a]) == 4:
      numH = 0
      neighbor = None
      for n in bondedNeighborLists[a]:
        if n.element == "H":
          numH += 1
        else:
          neighbor = n
      if numH == 3:
        # See if the Carbon's other neighbor is attached to two other atoms (3 total).  If so,
        # then insert a MoverAromaticMethylRotator and if not, generate a MoverTetrahedralMethylRotator
        # so that they Hydrogens will be staggered but do not add it to those to be optimized.
        if len(bondedNeighborLists[neighbor]) == 3:
          try:
            movers.append(Movers.MoverAromaticMethylRotator(a, bondedNeighborLists))
            infoString += _VerboseCheck(1,"Added MoverAromaticMethylRotator to "+resNameAndID+" "+aName+"\n")
          except Exception as e:
            infoString += _VerboseCheck(0,"Could not add MoverAromaticMethylRotator to "+resNameAndID+" "+aName+": "+str(e)+"\n")
        else:
          try:
            ignored = Movers.MoverTetrahedralMethylRotator(a, bondedNeighborLists)
            infoString += _VerboseCheck(1,"Used MoverTetrahedralMethylRotator to stagger "+resNameAndID+" "+aName+"\n")
          except Exception as e:
            infoString += _VerboseCheck(0,"Could not add MoverTetrahedralMethylRotator to "+resNameAndID+" "+aName+": "+str(e)+"\n")

    # See if we should insert a MoverNH2Flip here.
    # @todo Is there a more general way than looking for specific names?
    if (aName == 'ND2' and resName == 'ASN') or (aName == 'NE2' and resName == 'GLN'):
      try:
        movers.append(Movers.MoverNH2Flip(a, "CA", bondedNeighborLists))
        infoString += _VerboseCheck(1,"Added MoverNH2Flip to "+resNameAndID+"\n")
      except Exception as e:
        infoString += _VerboseCheck(0,"Could not add MoverNH2Flip to "+resNameAndID+": "+str(e)+"\n")

    # See if we should insert a MoverHistidineFlip here.
    # @todo Is there a more general way than looking for specific names?
    if aName == 'NE2' and resName == 'HIS':
      try:
        # Get a potential Mover and test both of its Nitrogens in the original and flipped
        # locations.  If one or both of them are near enough to be ionically bound to an
        # ion, then we remove the Hydrogen(s) and lock the Histidine at that orientation
        # rather than inserting the Mover into the list of those to be optimized.
        hist = Movers.MoverHistidineFlip(a, bondedNeighborLists, extraAtomInfo)

        # Find the four positions to check for Nitrogen ionic bonds
        # The two atoms are NE2 (0th atom with its Hydrogen at atom 1) and
        # ND1 (4th atom with its Hydrogen at atom 5).
        cp = hist.CoarsePositions()
        ne2Orig = cp.positions[0][0]
        nd1Orig = cp.positions[0][4]
        ne2Flip = cp.positions[3][0]
        nd1Flip = cp.positions[3][4]

        # See if any are close enough to an ion to be ionically bonded.
        # If any are, record whether it is the original or
        # the flipped configuration.  Check the original configuration first.
        # Check out to the furthest distance of any atom's VdW radius.
        myRad = extraAtomInfo.getMappingFor(a).vdwRadius
        minDist = myRad
        maxDist = 0.25 + myRad + maxVDWRadius
        bondedConfig = None
        for i,pos in enumerate([ne2Orig, nd1Orig, ne2Flip, nd1Flip]):
          neighbors = spatialQuery.neighbors(pos, minDist, maxDist)
          for n in neighbors:
            if Helpers.isMetallic(n):
              dist = (Movers._rvec3(pos) - Movers._rvec3(n.xyz)).length()
              expected = myRad + extraAtomInfo.getMappingFor(n).vdwRadius
              infoString += _VerboseCheck(5,'Checking '+str(i)+' against '+n.name.strip()+' at '+str(n.xyz)+' from '+str(pos)+
                ' dist = '+str(dist)+', expected = '+str(expected)+'; N rad = '+str(myRad)+
                ', '+n.name.strip()+' rad = '+str(extraAtomInfo.getMappingFor(n).vdwRadius)+'\n')
              if dist >= (expected - 0.55) and dist <= (expected + 0.25):
                # The first two elements come from configuration 0 and the second two from configuration 3
                bondedConfig = (i // 2) * 3
                infoString += _VerboseCheck(5,'Found ionic bond in coarse configuration '+str(bondedConfig)+'\n')
                break
          if bondedConfig is not None:
            # We want the first configuration that is found, so we don't flip if we don't need to.
            break

        # If one of the bonded configurations has at least one Ionic bond, then check each of
        # the Nitrogens in that configuration, removing its Hydrogen if it is bonded to an ion.
        # Set the histidine to that flip state; it will not be inserted as a Mover.
        if bondedConfig is not None:
          # Set the histidine in that flip state
          fixUp = hist.FixUp(bondedConfig)
          coarsePositions = hist.CoarsePositions().positions[bondedConfig]
          infoString += _VerboseCheck(5,'XXX Before fixup\n')
          for i,a in enumerate(fixUp.atoms):
            if i < len(fixUp.positions):
              a.xyz = fixUp.positions[i]
            if i < len(fixUp.extraInfos):
              extraAtomInfo.setMappingFor(a, fixUp.extraInfos[i])

          # See if we should remove the Hydrogen from each of the two potentially-bonded
          # Nitrogens and make each an acceptor if we do remove its Hydrogen.  The two atoms
          # are NE2 (0th atom with its Hydrogen at atom 1) and ND1 (4th atom with
          # its Hydrogen at atom 5).
          # NOTE that we must check the position of the atom in its coarse configuration rather
          # than its FixUp configuration because that is the location that we use to determine if
          # we have an ionic bond.
          def _modifyIfNeeded(nitro, coarseNitroPos, hydro):
            # Helper function to check and change things for one of the Nitrogens.
            myRad = extraAtomInfo.getMappingFor(nitro).vdwRadius
            minDist = myRad
            maxDist = 0.25 + myRad + maxVDWRadius
            neighbors = spatialQuery.neighbors(coarseNitroPos, minDist, maxDist)
            for n in neighbors:
              if Helpers.isMetallic(n):
                dist = (Movers._rvec3(coarseNitroPos) - Movers._rvec3(n.xyz)).length()
                expected = myRad + extraAtomInfo.getMappingFor(n).vdwRadius
                if dist >= (expected - 0.55) and dist <= (expected + 0.25):
                  nonlocal infoString
                  infoString += _VerboseCheck(1,'Removing Hydrogen from '+resNameAndID+nitro.name+' and marking as an acceptor '+
                    '(ionic bond to '+n.name.strip()+')\n')
                  extra = extraAtomInfo.getMappingFor(nitro)
                  extra.isAcceptor = True
                  extraAtomInfo.setMappingFor(nitro, extra)
                  deleteAtoms.append(hydro)
                  break

          _modifyIfNeeded(fixUp.atoms[0], coarsePositions[0], fixUp.atoms[1])
          _modifyIfNeeded(fixUp.atoms[4], coarsePositions[4], fixUp.atoms[5])

          infoString += _VerboseCheck(1,"Set MoverHistidineFlip on "+resNameAndID+" to state "+str(bondedConfig)+"\n")
        else:
          movers.append(hist)
          infoString += _VerboseCheck(1,"Added MoverHistidineFlip to "+resNameAndID+"\n")
      except Exception as e:
        infoString += _VerboseCheck(0,"Could not add MoverHistidineFlip to "+resNameAndID+": "+str(e)+"\n")

  return _PlaceMoversReturn(movers, infoString, deleteAtoms)

##################################################################################
# Helper functions

def AlternatesInModel(model):
  """Returns a set of altloc names of all conformers in all chains.
  :returns Set of strings.  The set is will include only the empty string if no
  chains in the model have alternates.  It has an additional entry for every altloc
  found in every chain.
  """
  ret = set()
  for c in model.chains():
    for alt in c.conformers():
      ret.add(alt.altloc)
  return ret

def GetAtomsForConformer(model, conf):
  """Returns a list of atoms in the named conformer.  It also includes atoms from
  the first conformation in chains that do not have this conformation, to provide a
  complete model.  For a model whose chains only have one conformer, it will return
  all of the atoms.
  :param model: pdb.hierarchy.model to search for atoms.
  :param conf: String name of the conformation to find.  Can be "".
  :returns List of atoms consistent with the specified conformer.
  """
  ret = []
  for ch in model.chains():
    confs = ch.conformers()
    which = 0
    for i in range(1,len(confs)):
      if confs[i].altloc == conf:
        which = i
        break
    ret.extend(ch.atoms())
  return ret

##################################################################################
# Test function and associated data and helpers to verify that all functions behave properly.

def Test(inFileName = None):
  """Test function for all functions provided above.
  :param inFileName: Name of a PDB or CIF file to load (default makes a small molecule)
  :returns Empty string on success, string describing the problem on failure.
  """

  #========================================================================
  # Test the AlternatesInModel() function.
  # @todo

  #========================================================================
  # Test the GetAtomsForConformer() function.
  # @todo

  ################################################################################
  # Test using snippet from 1xso to ensure that the Histidine placement code will lock down the
  # Histidine, set its Nitrogen states, and mark its Hydrogens for deletion.  To
  # ensure that hydrogen placement puts them there in the first place, we first
  # move the CU and ZN far from the Histidine before adding Hydrogens, then move
  # them back before building the spatial hierarchy and testing.
  # @todo Why are CU and ZN not being found in CCTBX for this snippet?  Missing ligands?  Print info to see.
  #   (Running 1sxo produces this as well).
  pdb_1xso_his_61_and_ions = (
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
HETATM 2190 CU    CU A   1      22.291  33.388   3.996  1.00 13.22          CU  
HETATM 2191 ZN    ZN A 152      27.539  36.010   2.881  1.00  9.34          ZN  
END
"""
    )
  dm = DataManager(['model'])
  dm.process_model_str("1xso_snip.pdb",pdb_1xso_his_61_and_ions)
  model = dm.get_model()

  # Make sure we have a valid unit cell.  Do this before we add hydrogens to the model
  # to make sure we have a valid unit cell.
  model = shift_and_box_model(model = model)

  # Find and move the Copper and Zinc atoms far from the Histidine so that
  # Hydrogen placement and bond proxies won't consider them to be bonded.
  # This tests the algorithm behavior for the case of all Hydrogens present
  # in all cases.
  for a in model.get_hierarchy().models()[0].atoms():
    if a.element.upper() == "CU":
      origPositionCU = a.xyz
      a.xyz = (1000, 1000, 1000)
    if a.element.upper() == "ZN":
      origPositionZN = a.xyz
      a.xyz = (1000, 1000, 1000)

  # Add Hydrogens to the model
  reduce_add_h_obj = reduce_hydrogen.place_hydrogens(model = model)
  reduce_add_h_obj.run()
  model = reduce_add_h_obj.get_model()

  # Interpret the model after adding Hydrogens to it so that
  # all of the needed fields are filled in when we use them below.
  # @todo Remove this once place_hydrogens() does all the interpretation we need.
  p = mmtbx.model.manager.get_default_pdb_interpretation_params()
  model.set_pdb_interpretation_params(params = p)
  model.process_input_model(make_restraints=True) # make restraints

  # Get the first model in the hierarchy.
  firstModel = model.get_hierarchy().models()[0]

  # Get the list of alternate conformation names present in all chains for this model.
  alts = AlternatesInModel(firstModel)

  # Get the atoms from the first conformer in the first model (the empty string is the name
  # of the first conformation in the model; if there is no empty conformation, then it will
  # pick the first available conformation for each atom group.
  atoms = GetAtomsForConformer(firstModel, "")

  # Get the Cartesian positions of all of the atoms we're considering for this alternate
  # conformation.
  carts = flex.vec3_double()
  for a in atoms:
    carts.append(a.xyz)

  # Get the bond proxies for the atoms in the model and conformation we're using and
  # use them to determine the bonded neighbor lists.
  bondProxies = model.get_restraints_manager().geometry.get_all_bond_proxies(sites_cart = carts)[0]
  bondedNeighborLists = Helpers.getBondedNeighborLists(atoms, bondProxies)

  # Put the Copper and Zinc back in their original positions before we build the
  # spatial-query structure.  This will make them close enough to be bonded to
  # the Nitrogens and should cause Hydrogen removal and marking of the Nitrogens
  # as acceptors.
  for a in model.get_hierarchy().models()[0].atoms():
    if a.element.upper() == "CU":
      a.xyz = origPositionCU
    if a.element.upper() == "ZN":
      a.xyz = origPositionZN

  # Get the spatial-query information needed to quickly determine which atoms are nearby
  sq = probeExt.SpatialQuery(atoms)

  # Get the probeExt.ExtraAtomInfo needed to determine which atoms are potential acceptors.
  ret = Helpers.getExtraAtomInfo(model)
  extra = ret.extraAtomInfo

  # Place the movers, which should include only an NH3 rotator because the Histidine flip
  # will be constrained by the ionic bonds.
  ret = _PlaceMovers(atoms, model.rotatable_hd_selection(iselection=True),
                     bondedNeighborLists, sq, extra, AtomTypes.AtomTypes().MaximumVDWRadius())
  movers = ret.moverList
  if len(movers) != 1:
    return "Optimizers.Test(): Incorrect number of Movers for 1xso Histidine test"

  # Make sure that the two ring Nitrogens have been marked as acceptors.
  # Make sure that the two hydrogens have been marked for deletion.
  for a in model.get_hierarchy().models()[0].atoms():
    name = a.name.strip()
    if name in ["ND1", "NE2"]:
      if not extra.getMappingFor(a).isAcceptor:
        return "Optimizers.Test(): '+a+' in 1xso Histidine test was not an acceptor"
    if name in ["HD1", "HE2"]:
      if not a in ret.deleteAtoms:
        return "Optimizers.Test(): '+a+' in 1xso Histidine test was not set for deletion"

  #========================================================================
  # Generate an example data model with a small molecule in it or else read
  # from the specified file.
  infoString = ""
  if inFileName is not None and len(inFileName) > 0:
    # Read a model from a file using the DataManager
    print('Reading model from',inFileName)
    dm = DataManager()
    dm.process_model_file(inFileName)
    model = dm.get_model(inFileName)
  else:
    # Generate a small-molecule model using the map model manager
    print('Generating model')
    mmm=map_model_manager()         #   get an initialized instance of the map_model_manager
    mmm.generate_map()              #   get a model from a generated small library model and calculate a map for it
    model = mmm.model()             #   get the model

  # Make sure we have a valid unit cell.  Do this before we add hydrogens to the model
  # to make sure we have a valid unit cell.
  model = shift_and_box_model(model = model)

  # Add Hydrogens to the model
  print('Adding Hydrogens')
  reduce_add_h_obj = reduce_hydrogen.place_hydrogens(model = model)
  reduce_add_h_obj.run()
  model = reduce_add_h_obj.get_model()

  # Interpret the model after shifting and adding Hydrogens to it so that
  # all of the needed fields are filled in when we use them below.
  # @todo Remove this once place_hydrogens() does all the interpretation we need.
  print('Interpreting model')
  p = mmtbx.model.manager.get_default_pdb_interpretation_params()
  model.set_pdb_interpretation_params(params = p)
  model.process_input_model(make_restraints=True) # make restraints

  print('Constructing Optimizer')
  # @todo Let the caller specify the model index and altID rather than doing only the first.
  nullOpt = NullOptimizer(model)
  info = nullOpt.getInfo()
  print('XXX info:\n'+info)

  # @todo

  #========================================================================
  # Unit tests for each type of Optimizer.
  # @todo

  # @todo Unit test a multi-model case, a multi-alternate case, and singles of each.

  return ""

##################################################################################
# If we're run on the command line, test our classes and functions.
if __name__ == '__main__':

  #==============================================================
  # Parse command-line arguments.  The 0th argument is the name
  # of the script. There can be the name of a PDB file to read.
  parser = argparse.ArgumentParser(description='Test mmtbx.reduce.Optimizers.')
  parser.add_argument('inputFile', nargs='?', default="")
  args = parser.parse_args()

  ret = Test(args.inputFile)
  if len(ret) == 0:
    print('Success!')
  else:
    print(ret)

  assert (len(ret) == 0)