from __future__ import absolute_import, division, print_function
from turtle import window_width
from libtbx.program_template import ProgramTemplate
import iotbx.pdb
import mmtbx.secondary_structure
from pathlib import Path
import numpy as np
from mmtbx.kinemage.validation import get_chain_color
from mmtbx.kinemage.ribbons import find_contiguous_protein_residues, find_contiguous_nucleic_acid_residues
from mmtbx.kinemage.ribbons import make_protein_guidepoints, make_nucleic_acid_guidepoints
from mmtbx.kinemage.ribbons import untwist_ribbon, swap_edge_and_face, non_CA_atoms_present, _FindNamedAtomInResidue
from mmtbx.kinemage.nrubs import Triple, NRUBS
from copy import copy

version = "1.0.0"

master_phil_str = '''
do_protein = True
  .type = bool
  .short_caption = Construct protein ribbons
  .help = Construct ribbons for the protein sections of the model
do_nucleic_acid = True
  .type = bool
  .short_caption = Construct nucleic acid ribbons
  .help = Construct ribbons for the nucleic acid sections of the model
untwist_ribbons = True
  .type = bool
  .short_caption = Untwist ribbons
  .help = Remove excess twist from ribbons by making the neighboring dot products positive
DNA_style = False
  .type = bool
  .short_caption = DNA style ribbons
  .help = Use the DNA style ribbons instead of the default RNA style ribbons (rotates them by 90 degrees)
color_by = *rainbow secondary_structure
  .type = choice(multi=False)
  .short_caption = How to color the ribbons
  .help = How to color the ribbons
do_plain_coils = False
  .type = bool
  .short_caption = Do plain coils
  .help = Do plain coils (no halo) even for the helix and sheet parts of the protein
coil_width = 1.0
  .type = float
  .short_caption = Coil width
  .help = Width of the coil part of the ribbon
'''

# ------------------------------------------------------------------------------

class Program(ProgramTemplate):
  description = '''
mmtbx.ribbons version {}: Given PDB file produce Kinemage file with ribbons representation.

This program produces a kinemage file with ribbons representation of the input file using the
same algorithm as the Richardson lab's MolProbity server.  The code is based on the Java code
within the javadev repository.

How to run:
  mmtbx.ribbons model.pdb
  
Output:
  If neither output.file_name nor output.filename is specified, it will write
  to a file with the same name as the input model file name but with the
  extension replaced with with '.kin'.

'''.format(version)
  datatypes = ['model', 'phil']
  master_phil_str = master_phil_str

# ------------------------------------------------------------------------------

  def validate(self):
    self.data_manager.has_models(raise_sorry=True)

    if self.params.output.filename is None:
      # If the output file name is not specified, use the same root as the
      # input file and replace the suffix with .kin.
      suffix = '.kin'
      inName = self.data_manager.get_default_model_name()
      p = Path(inName)
      self.params.output.filename = str(p.with_suffix(suffix))
      print('Setting output.filename Phil parameter to',self.params.output.filename)

# ------------------------------------------------------------------------------

  def splineInterplate(self, pts, nIntervals):
    # Returns a list of interpolated points between the points using a spline interpolation.
    # Based on https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#parametric-spline-curves
    # @param pts: The guidepoints to interpolate between (a list of 3D points with at least 6 points in it).
    # The first and last points are duplicates to form knots, so are not interpolated between.
    # @param nIntervals: The number of intervals to interpolate between each pair of guidepoints, skipping the first and last.
    # @return: The list of interpolated points

    nrubs = NRUBS()
    points = []
    for pt in pts:
      points.append( Triple(pt[0], pt[1], pt[2]) )
    res = nrubs.spline(points, nIntervals)
    ret = []
    for r in res:
      ret.append( np.array((r.x, r.y, r.z)) )
    return ret

    # Adjust the points to match what is expected by the Java code.
    # The Java code expects the first and last points to be duplicates of the second and second-to-last points.
    points = pts[1:-1]

# ------------------------------------------------------------------------------

  # The original code constructed a NONE crayon for the fancy print, which did not add color to
  # the output, re-using the colors specified by model or chain.  It sometimes also specified
  # the RAINBOW crayon, which overwrites the colors with a rainbow color scheme across each chain.
  # For the edges, it specified a constant crayon whose string value is deadblack
  # and it adds "U " to make it un-pickable. @todo Add unpickable somewhere.
  # We pull that logic into here by implementing the forRibbon() and shouldPrint() and
  # getKinString() methods in our own class.
  class Crayon:
    def __init__(self, doRainBow, color=None):
      self._doRainBow = doRainBow
      self._color = color
      self._rainbowColors = [ "blue", "sky", "cyan", "sea", "green", "lime", "yellow" ,"gold" ,"orange" ,"red" ]

    # Rainbow color map changes the color once across the rainbow for each chain (the chain goes from
    # blue to red).
    # The non-rainbow, constant-colored map does not change.
    def forRibbon(self, startGuide):
      if self._doRainBow:
        res = startGuide.prevRes
        chain = res.parent()
        firstChainResID = chain.residue_groups()[0].resseq
        lastChainResID = chain.residue_groups()[-1].resseq
        normRes = (int(res.resseq) - int(firstChainResID)) / (int(lastChainResID) - int(firstChainResID))
        scaledIndex = int(normRes * len(self._rainbowColors))
        if scaledIndex == len(self._rainbowColors):
          scaledIndex -= 1
        self._color = self._rainbowColors[scaledIndex]

    def getKinString(self):
      if self._color is None:
        return ""
      return self._color

    # All of our elements should be visible in the output.
    def shouldPrint(self):
      return True

  # Commented out the fields and did not include methods that we don't need for our implementation
  # Switched getType() to making type public.
  # Its fields should be based on the PDB secondary structure records.
  class Range:
    def __init__(self, pType = 'COIL', pChainId = '', pInit = 0, pEnd = 0, pSense = 0, pStrand = 1, pSheet = ' ', pFlipped = False):
      # self._rangeIndex = 0
      self.type = pType
      self._chainId = pChainId
      self.initSeqNum = pInit
      self.endSeqNum = pEnd
      #self._initICode = ' '
      #self._endICode = ' '
      self.sense = pSense    # 0 if first strand, 1 if parallel, -1 if anti-parallel
      self.strand = pStrand   # Starts at 1 for each strand within a sheet and increases by 1
      self.sheetID = pSheet
      self.flipped = pFlipped  # Used by printing code to make the tops within a set of ribbons all point the same way
      self.previous = None
      self.next = None
      self.duplicateOf = None

    def __str__(self):
      return 'Range: type={}, init={}, end={}, sense={}, strand={}, sheet={}, flipped={}'.format(
        self.type, self.initSeqNum, self.endSeqNum, self.sense, self.strand, self.sheetID, self.flipped)

  def ConsolidateSheets(self):
    uniqueStrands = {}

    # For each ribbon, record its predecessor in sheet, and check for duplicates
    for rng in self.secondaryStructure.values():
      if not rng.type == 'SHEET':
        continue

      key = str(int(rng.initSeqNum)) + rng._chainId
      if not key in uniqueStrands:
        uniqueStrands[key] = rng
      else:
        rng.duplicateOf = uniqueStrands[key]

      # Now find this strand's previous and next strand.
      for rng2 in self.secondaryStructure.values():
        if not rng2.type == 'SHEET':
          continue
        if rng2.sheetID == rng.sheetID and rng2.strand == rng.strand-1:
          rng.previous = rng2
          rng2.next = rng

      # Now go through and reassign previous/next fields in duplicates
      for rng in self.secondaryStructure.values():
        if not rng.type == 'SHEET':
          continue
        if rng.duplicateOf is None and rng.next is not None and rng.next.duplicateOf is not None:
          rng.next.duplicateOf.previous = rng
        if rng.duplicateOf is None and rng.previous is not None and rng.previous.duplicateOf is not None:
          rng.previous = rng.previous.duplicateOf

  class RibbonElement:
    def __init__(self, other=None, setRange=None):
      self.start = 0
      self.end = 0
      self.type = 'COIL'
      self.range = setRange
      if other is not None:
        self.like(other)
      if self.range is not None:
        if self.range.type is None or self.range.type == 'TURN':
          self.type = 'COIL'
        else:
          self.type = self.range.type

    def sameSSE(self, that):
      return self.type == that.type and self.range == that.range

    def like(self, that):
      self.start = that.start
      self.end = that.end
      self.type = that.type
      self.range = that.range

    def __lt__(self, that):
      a = 0
      b = 0
      if self.range is not None:
        a = self.range.strand
      if that.range is not None:
        b = that.range.strand
      return a < b

  def getPointID(self, point, start, end, interval, nIntervals):
    res = start.nextRes
    if self.params.DNA_style and interval <= nIntervals // 2:
      res = start.prevRes   # == first res, for RNA/DNA only

    buf = res.atom_groups()[0].resname.strip() + " " + res.parent().id.strip() + " " + res.resseq.strip() + res.icode
    return buf.lower().strip()

  def printFancy(self, guides, splines, i, lineBreak = False):
    # Prints a fancy ribbon element with the given guidepoints and splines.
    # @param guides: The guidepoints for the ribbon
    # @param splines: The interpolated points for the ribbon
    # @param i: The index of the guidepoint to print

    ret = ""
    # Not self.nIntervals, we want a local variable here
    nIntervals = (len(splines) - 1) // (len(guides) - 3)
    interval = i % nIntervals
    startGuide = (i // nIntervals) + 1
    ret += "{"
    ret += self.getPointID(splines[i], guides[startGuide], guides[startGuide + 1], interval, nIntervals)
    ret += "}"
    if lineBreak:
      ret += "P "
    self.crayon.forRibbon(guides[startGuide])
    ret += self.crayon.getKinString()
    ret += " "
    ret += str(splines[i][0]) + " " + str(splines[i][1]) + " " + str(splines[i][2]) + "\n"

    return ret

  def printFancyRibbon(self, guides, widthAlpha, widthBeta, listAlpha, listBeta, listCoil, listCoilOutline, doRainbow):
    # Constructs a kinemage string for a ribbon representation of a protein or nucleic acid chain.
    # Makes a triangulated ribbon with arrowheads, etc.
    # @param guides: The guidepoints for the ribbon
    # @param widthAlpha: The width of the alpha helix part of the ribbon
    # @param widthBeta: The width of the beta sheet part of the ribbon
    # @param listAlpha: The kinemage string describing the alpha helix part of the ribbon
    # @param listBeta: The kinemage string describing the beta sheet part of the ribbon
    # @param listCoil: The kinemage string describing the coil part of the ribbon
    # @param listCoilOutline: The kinemage string describing the outline of the coil part of the ribbon
    # @param doRainbow: Whether or not to color the ribbon by rainbow
    # If false (the default), the protein style will be used instead.
    # @return: The kinemage string for the ribbon representation

    ret = ""

    # Initialize local references
    widthCoil = self.params.coil_width
    secStruct = self.secondaryStructure

    # Seven strands of guidpoints: coil, +/-alpha, +/-beta, +/-beta arrowheads.
    # Each strand is a list of 3D points and there is a strand for each offset from the center spline.
    halfWidths = [0.0, -widthAlpha/2, widthAlpha/2, -widthBeta/2, widthBeta/2, -widthBeta, widthBeta]
    # Make a different empty list for each strand
    strands = []
    for _ in range(len(halfWidths)):
      strands.append([])
    for g in guides:
      for i in range(len(halfWidths)):
        strands[i].append(g.pos + g.dvec * halfWidths[i])

    # Seven strands of interpolated points
    splinepts = []
    for i in range(len(strands)):
      splinepts.append(self.splineInterplate(strands[i], self.nIntervals))

    # Discovery of ribbon elements: ribbons, ropes, and arrows.
    # We skip the regions associated with the first and last points, which are present to cause the
    # curve to pass through those points (like knots) but are not interpolated between.
    # We do that by looping through fewer entries and by adding 1 to the index.
    ribbonElements = []
    ribElement = self.RibbonElement()
    prevRibElt = self.RibbonElement()
    ribbonElements.append(ribElement)
    # Element that is reused and copied when creating a new list entry
    ribElement.type = None
    for i in range(len(guides) - 1 - 2):
      g1 = guides[i + 1]
      g2 = guides[i + 2]

      # The Java code reports that we're not really using ribbon elements, just reusing the
      # class for convenience.  So not all of the member variables may be filled in.
      currSS = self.RibbonElement(setRange=secStruct[int(g1.nextRes.resseq)])
      nextSS = self.RibbonElement(setRange=secStruct[int(g2.nextRes.resseq)])

      # Otherwise, we get one unit of coil before alpha or beta at start
      if ribElement.type is None:
        ribElement.like(currSS)

      if not ribElement.sameSSE(currSS): # Helix / sheet starting
        if currSS.type == 'HELIX' or currSS.type == 'SHEET':
          ribElement.end = self.nIntervals*i + 1
          ribElement = self.RibbonElement(currSS)
          ribbonElements.append(ribElement)
          # Every helix or sheet starts with a coil; see below
          ribElement.start = self.nIntervals*i + 1
      if not ribElement.sameSSE(nextSS): # Helix / sheet ending
        if currSS.type == 'HELIX' or currSS.type == 'SHEET':
          end = self.nIntervals*i + 0
          if currSS.type == 'SHEET':
            end += self.nIntervals - 1
          ribElement.end = end
          ribElement = self.RibbonElement()
          ribbonElements.append(ribElement)
          # Every helix or sheet flows into coil
          ribElement.type = 'COIL'
          ribElement.start = end
    ribElement.end = len(splinepts[0]) - 1

    # "Crayons" to use for coloring the ribbon, juggling them around to keep the edges black.
    normalCrayon = self.Crayon(doRainbow)
    edgeCrayon = self.Crayon(False)  # Default no color; the deadblack will be set on the vectorlist

    # Sort the ribbon elements by strand number
    ribbonElements.sort()

    # Create a list of just sheets (Java STRANDs) for searching through
    strands = [r for r in ribbonElements if r.type == 'SHEET']

    for ribElement in ribbonElements:
      stGuide = (ribElement.start // self.nIntervals) + 2
      endGuide = (ribElement.end // self.nIntervals) - 1

      if ribElement.type == 'HELIX':

        k = (ribElement.start + ribElement.end) // 2
        pt = splinepts[2][k]
        v1 = splinepts[1][k] - pt
        v2 = splinepts[1][k+1] - pt
        cross = np.cross(v1, v2)
        dot = np.dot( np.linalg.norm(cross), np.linalg.norm(np.array(guides[k//self.nIntervals+1].cvec)))

        self.crayon = normalCrayon
        ret += "@ribbonlist {fancy helix} " + listAlpha + "\n"

        # @todo The Java code has a "P X " at the start of the first line in the ribbon
        for i in range(ribElement.start, ribElement.end):
          if dot > 0:
            # Flip the normals (for sidedness) by switching the order of these two lines.
            ret += self.printFancy(guides, splinepts[2], i)
            ret += self.printFancy(guides, splinepts[1], i)
          else:
            ret += self.printFancy(guides, splinepts[1], i)
            ret += self.printFancy(guides, splinepts[2], i)
        ret += self.printFancy(guides, splinepts[0], ribElement.end)  # Angled tip at end of helix
        self.crayon = edgeCrayon
        ret += "@vectorlist {fancy helix edges} width=1 " + listAlpha + " color= deadblack\n"
        # Black edge, left side
        ret += self.printFancy(guides, splinepts[0], ribElement.start, True)
        for i in range(ribElement.start, ribElement.end):
          ret += self.printFancy(guides, splinepts[1], i)
        ret += self.printFancy(guides, splinepts[0], ribElement.end)
        # Black edge, right side
        ret += self.printFancy(guides, splinepts[0], ribElement.start, True)
        for i in range(ribElement.start, ribElement.end):
          ret += self.printFancy(guides, splinepts[2], i)
        ret += self.printFancy(guides, splinepts[0], ribElement.end)

      elif ribElement.type == 'SHEET':

        dot = 0.0     # Used to determine sidedness

        # Don't do for the first strand in the sheet
        if ribElement.range.strand != 1:
          # Look for previous strand
          for i in range(len(strands)):
            curElt = strands[i]
            if curElt.range == ribElement.range.previous:
              prevRibElt = curElt
              break

          prevRange = prevRibElt.range
          if prevRange is None:
            prevRibElt = ribElement

          # Detrmine sidedness using splinepts and normals
          prevStGuide = (prevRibElt.start // self.nIntervals) + 2
          prevEndGuide = (prevRibElt.end // self.nIntervals) + 1
          curClosest = stGuide      # Find the pair of guidepoints closest to each other
          prevClosest = prevStGuide # (one on each strand) to perform the test
          closeDist = 1e10
          for i in range(stGuide, endGuide+2 + 1):
            for j in range(prevStGuide, endGuide+2 + 1):
              # Look for closest pair of H-bonding partners
              try:
                O = _FindNamedAtomInResidue(guides[i].prevRes, "O")
                N = _FindNamedAtomInResidue(guides[i].prevRes, "N")
                dist = np.linalg.norm(O.pos - N.pos)
                if dist < closeDist:
                  closeDist = dist
                  curClosest = i
                  prevClosest = j
              except:
                pass
          kCur = min(self.nIntervals*(curClosest-1), len(splinepts[4]))
          kPrev = min(self.nIntervals*(prevClosest-1), len(splinepts[4]))
          ptCur = splinepts[4][kCur]
          v1Cur = splinepts[3][kCur] - ptCur
          # VBC Hack to get 1jj2 at least generating ribbons kins.  Doesn't seem to 
          # correctly generate beta sides though.  Error is kCur+1 generates an ArrayIndexOutOfBoundsException in splinepts[3]
          if kCur+1 < len(splinepts[3]):
            v2Cur = splinepts[3][kCur+1] - ptCur
            crossCur = np.cross(v1Cur, v2Cur)
            ptPrev = splinepts[4][kPrev]
            v1Prev = splinepts[3][kPrev] - ptPrev
            v2Prev = splinepts[3][kPrev+1] - ptPrev
            crossPrev = np.cross(v1Prev, v2Prev)
            dot = np.dot( crossCur, crossPrev)

        self.crayon = normalCrayon
        ret += "@ribbonlist {fancy sheet} " + listBeta + "\n"
        for i in range(ribElement.start, ribElement.end - 1):
          # If strands are not "facing" the same way,
          # flip the normals (for sidedness) by switching the order of these two lines, (ARK Spring2010)
          if (dot < 0 and not prevRibElt.range.flipped) or (dot > 0 and prevRibElt.range.flipped):
            ret += self.printFancy(guides, splinepts[4], i)
            ret += self.printFancy(guides, splinepts[3], i)
            ribElement.range.flipped = True
          else:
            ret += self.printFancy(guides, splinepts[3], i)
            ret += self.printFancy(guides, splinepts[4], i)

        # Ending *exactly* like this is critical to avoiding a break
        # between the arrow body and arrow head!
        if (dot < 0 and not prevRibElt.range.flipped) or (dot > 0 and prevRibElt.range.flipped):
          ret += self.printFancy(guides, splinepts[6], ribElement.end - 2)
          ret += self.printFancy(guides, splinepts[5], ribElement.end - 2)
          ret += self.printFancy(guides, splinepts[0], ribElement.end)
        else:
          ret += self.printFancy(guides, splinepts[5], ribElement.end - 2)
          ret += self.printFancy(guides, splinepts[6], ribElement.end - 2)
          ret += self.printFancy(guides, splinepts[0], ribElement.end)
        # Borders
        self.crayon = edgeCrayon
        ret += "@vectorlist {fancy sheet edges} width=1 " + listBeta + " color= deadblack\n"
        # Black edge, left side
        ret += self.printFancy(guides, splinepts[0], ribElement.start, True)
        for i in range(ribElement.start, ribElement.end - 1):
          ret += self.printFancy(guides, splinepts[3], i)
        ret += self.printFancy(guides, splinepts[5], ribElement.end - 2)
        ret += self.printFancy(guides, splinepts[0], ribElement.end)
        # Black edge, right side
        ret += self.printFancy(guides, splinepts[0], ribElement.start, True)
        for i in range(ribElement.start, ribElement.end - 1):
          ret += self.printFancy(guides, splinepts[4], i)
        ret += self.printFancy(guides, splinepts[6], ribElement.end - 2)
        ret += self.printFancy(guides, splinepts[0], ribElement.end)

      else: # Coil

        # for black outlines on coils
        if listCoilOutline is not None:
          self.crayon = normalCrayon
          ret += "@vectorlist {fancy coil edges} " + listCoilOutline + "\n"
          for i in range(ribElement.start, ribElement.end+1):
            ret += self.printFancy(guides, splinepts[0], i)

        self.crayon = normalCrayon
        ret += "@vectorlist {fancy coil} " + listCoil + "\n"
        for i in range(ribElement.start, ribElement.end+1):
          ret += self.printFancy(guides, splinepts[0], i)

      # Reset the crayon for the next pass
      self.crayon = normalCrayon

    return ret

# ------------------------------------------------------------------------------

  def run(self):
    self.model = self.data_manager.get_model()
    self.nIntervals = 4

    # Analyze the secondary structure and make a dictionary that maps from residue sequence number to secondary structure type
    # by filling in 'COIL' as a default value for each and then parsing all of the secondary structure records in the
    # model and filling in the relevant values for them.
    # @todo Can pass it an atom selection cache, so maybe we do this after we have selected the atoms we want to use?
    print('Finding secondary structure:')
    ss_manager = mmtbx.secondary_structure.manager(self.model.get_hierarchy())
    self.secondaryStructure = {}
    for model in self.model.get_hierarchy().models():
      for chain in model.chains():
        for residue_group in chain.residue_groups():
          self.secondaryStructure[int(residue_group.resseq)] = self.Range()
    for line in ss_manager.records_for_pdb_file().splitlines():
      r = self.Range(line[0:5].strip())
      if r.type == 'HELIX':
        r.sheetID = line[11:14].strip()
        r._chainId = line[19:21].strip()
        r.initSeqNum = int(line[22:26].strip())
        r.endSeqNum = int(line[33:38].strip())
      elif r.type == 'SHEET':  # Java code marks this internally as STRAND but we leave it as SHEET
        r.sheetID = line[11:14].strip()
        r._chainId = line[21:22].strip()
        r.initSeqNum = int(line[23:28].strip())
        r.endSeqNum = int(line[33:38].strip())
        r.sense = int(line[38:40].strip())
        r.strand = int(line[7:10].strip())
      elif r.type == 'TURN':
        # In fact, we turn turns int coils, so we really don't care.
        # r.sheetID = line[11:14].strip()
        r._chainId = line[20:21].strip()
        r.initSeqNum = int(line[22:26].strip())
        r.endSeqNum = int(line[33:37].strip())
      for i in range(r.initSeqNum,r.endSeqNum+1):
        self.secondaryStructure[i] = r

    self.ConsolidateSheets()

    # The name of the structure, which is the root of the input file name (no path, no extension)
    self.idCode = self.data_manager.get_default_model_name().split('.')[0]
    if self.idCode is None or self.idCode == "":
      self.idCode = "macromol"

    # Create the output string that will be written to the output file.  It will be filled in during the run.
    outString = ""

    # Fill in the header information
    outString += "@kinemage 1\n"
    outString += "@onewidth\n"
    
    # Handle multiple models
    groupByModel = self.model.get_hierarchy().models_size() > 1
    for model in self.model.get_hierarchy().models():
      modelID = model.id
      if modelID == "":
        modelID = "_"
      print('Processing model', modelID, 'with', len(model.chains()), 'chains')
      if groupByModel:
        # @todo Test and figure out what to fill in for the "@todo" name based on what the original code produces
        outString += "@group {{{} {}}} dominant master= {{all models}}\n".format(self.idCode, "@todo")
 
      # Make a list of all the chain names in the model with only one entry per name.
      # Use this to make a dictionary to look up the color that is used for each chain name.
      # This ensures that the chains are colored the same no matter their order or repeat in the file.  
      chainNames = set()
      for chain in model.chains():
        chainNames.add(chain.id)
      chainNames = sorted(list(chainNames))
      chainCount = 0
      chainColors = {}
      for name in chainNames:
        # Backbone color by model ID if we have multiple models, or by chain ID if we have multiple chains in a model.
        if groupByModel:
          c = get_chain_color(modelID)
        else:
          c = get_chain_color(chainCount)
        chainColors[name] = c
        chainCount += 1

      # Cycle over all chains in the model and make a group or subgroup for each chain
      # depending on whether we are grouping by model or not.
      for chain in model.chains():
        print('Processing chain',chain.id)

        if self.params.do_protein:
          # Find the contiguous protein residues by CA distance
          contiguous_residue_lists = find_contiguous_protein_residues(chain)
          print('Found {} contiguous protein residue lists'.format(len(contiguous_residue_lists)))

          if len(contiguous_residue_lists) > 0:
            if groupByModel:
              outString += "@subgroup {{chain{}}} dominant master= {chain {}}}\n".format(chain.id, chain.id)
            else:
              outString += "@group {{{} {}}} dominant\n".format(self.idCode, chain.id)
            outString += "@subgroup {ribbon}\n"

            bbColor = chainColors[chain.id]
            if bbColor == "white":
              # Distinguish between the different types of secondary structure for the first chain
              outString += "@colorset {{alph{}}} red\n".format(chain.id)
              outString += "@colorset {{beta{}}} lime\n".format(chain.id)
              outString += "@colorset {{coil{}}} white\n".format(chain.id)
            else:
              # Do all secondary structure in the same color for all but the first chain to clean up the display
              outString += "@colorset {{alph{}}} {}\n".format(chain.id, bbColor)
              outString += "@colorset {{beta{}}} {}\n".format(chain.id, bbColor)
              outString += "@colorset {{coil{}}} {}\n".format(chain.id, bbColor)

            for contig in contiguous_residue_lists:
              guidepoints = make_protein_guidepoints(contig)
              print(' Made {} protein guidepoints for {} residues'.format(len(guidepoints),len(contig)))
              if self.params.untwist_ribbons:
                print('  Untwisted ribbon')
                untwist_ribbon(guidepoints)
              # There is always secondary structure looked up for protein residues, so we skip the case from the Java code
              # where it can be missing.
              if self.params.do_plain_coils:
                outString += self.printFancyRibbon(guidepoints, 2, 2.2,
                      "color= {alph"+chain.id+"} master= {protein} master= {ribbon} master= {alpha}",
                      "color= {beta"+chain.id+"} master= {protein} master= {ribbon} master= {beta}",
                      "width= 4 color= {coil"+chain.id+"} master= {protein} master= {ribbon} master= {coil}",
                      self.params.color_by == "rainbow");
              else:
                outString += self.printFancyRibbon(guidepoints, 2, 2.2,
                      "color= {alph"+chain.id+"} master= {protein} master= {ribbon} master= {alpha}",
                      "color= {beta"+chain.id+"} master= {protein} master= {ribbon} master= {beta}",
                      "width= 4 fore color= {coil"+chain.id+"} master= {protein} master= {ribbon} master= {coil}",
                      "width= 6 rear color= deadblack master= {protein} master= {ribbon} master= {coil}",
                      self.params.color_by == "rainbow")
      
        if self.params.do_nucleic_acid:
          # Find the contiguous nucleic acid residues by CA distance
          contiguous_residue_lists = find_contiguous_nucleic_acid_residues(chain)
          print('Found {} contiguous nucleic acid residue lists'.format(len(contiguous_residue_lists)))

          if len(contiguous_residue_lists) > 0:
            if groupByModel:
              outString += "@subgroup {{chain{}}} dominant master= {chain {}}}\n".format(chain.id, chain.id)
            else:
              outString += "@group {{{} {}}} dominant\n".format(self.idCode, chain.id)
            outString += "@subgroup {ribbon}\n"

            bbColor = chainColors[chain.id]
            if bbColor == "white":
              outString += "@colorset {{nucl{}}} lime\n".format(chain.id)
              outString += "@colorset {{ncoi{}}} white\n".format(chain.id)
            else:
              outString += "@colorset {{nucl{}}} {}\n".format(chain.id, bbColor)
              outString += "@colorset {{ncoi{}}} {}\n".format(chain.id, bbColor)

            for contig in contiguous_residue_lists:
              guidepoints = make_nucleic_acid_guidepoints(contig)
              print(' Made {} NA guidepoints for {} residues'.format(len(guidepoints),len(contig)))
              if self.params.untwist_ribbons:
                print('  Untwisted ribbon')
                untwist_ribbon(guidepoints)
              if self.params.DNA_style:
                print('  Swapped edge and face (DNA style)')
                swap_edge_and_face(guidepoints)
              else:
                print('  Using RNA style ribbons')

              outString += self.printFancyRibbon(guidepoints, 3.0, 3.0,
                    "color= {nucl"+chain.id+"} master= {nucleic acid} master= {ribbon} master= {RNA helix?}",
                    "color= {nucl"+chain.id+"} master= {nucleic acid} master= {ribbon} master= {A-form}",
                    "width= 4 color= {ncoi"+chain.id+"} master= {nucleic acid} master= {ribbon} master= {coil}",
                    None,
                    self.params.color_by == "rainbow")

    # Write the output to the specified file.
    self.data_manager._write_text("Text", outString, self.params.output.filename)
