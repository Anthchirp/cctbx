from __future__ import division
"""
The script attempts to simulate diffraction patterns
for symmetrically equivalent crystal orientations, and tests if the intensities change.
If changes in intensity are detected, the stdout will indicate Failure.
"""
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("sym", type=str, choices=["C","I","F","P1","P2","P3","P4","P6"], help="crystal system")
parser.add_argument("--cuda", action="store_true", help="Uses Giles nanoBragg CUDA kernel")
parser.add_argument("--fix", action="store_true", help="fix using symmetric mosaic blocks (exaFEL fix)")
parser.add_argument("--mos", action="store_true", help="add mosaic spread")
parser.add_argument("--plot", action="store_true", help="display plots at the end")
parser.add_argument("--gauss", action="store_true", help="Use the original Gaussian RELP shape")
parser.add_argument("--square", action="store_true", help="Use the Square RELP shape")
args = parser.parse_args()
from scipy.spatial.transform import Rotation
import numpy as np
from simtbx.diffBragg import utils as db_utils

from simtbx.nanoBragg import nanoBragg, shapetype
from scitbx.matrix import sqr
from dxtbx.model import Crystal
from dxtbx.model import DetectorFactory, BeamFactory
from cctbx import sgtbx, miller, crystal, uctbx
from cctbx.array_family import flex
from cctbx.sgtbx.literal_description import literal_description


# ===================
# prepare the miller array for nanoBragg
def prep_miller_array(mill_arr):
    """prep for nanoBragg"""
    # TODO: is this correct order of things?
    cb_op = mill_arr.space_group_info().change_of_basis_op_to_primitive_setting()
    mill_arr = mill_arr.expand_to_p1()
    mill_arr = mill_arr.generate_bijvoet_mates()
    dtrm = cb_op.c().r().determinant()
    if not dtrm == 1:
        mill_arr = mill_arr.change_basis(cb_op)
    return mill_arr


# simple method to check whether the structure factors in nanoBragg obey the symmetry
def sanity_check_Fs(SIM, sg, ucell_p1):
    """
    SIM: nanoBragg instance
    sg: space group operator
    ucell_p1: p1 unit cell params
    """
    inds, amps = SIM.Fhkl_tuple
    print(ucell_p1)

    sym = crystal.symmetry(ucell_p1, "P1")
    mset = miller.set(sym, flex.miller_index(inds), True)
    ma = miller.array(mset, flex.double(amps)).set_observation_type_xray_amplitude().resolution_filter(-1,
                                                                                                       2)
    print(sg.info())
    ops = [o for o in sg.build_derived_laue_group().all_ops() if o.r().determinant() == 1]
    for o in ops:
        print("Op=", o.as_xyz())
        ma2 = ma.change_basis(sgtbx.change_of_basis_op(o))
        r = ma.r1_factor(ma2, assume_index_matching=True)
        print("R=", r, "\n")
        assert r == 0
# ===================

add_spots_func="add_nanoBragg_spots"
if args.cuda:
    add_spots_func = "add_nanoBragg_spots_cuda"

syms = {
 'C':  # 1hk5
     [(190.14, 39.04, 96.23, 90, 105.07, 90),
    "C 1 2 1"],
 'I':  # 3dll
     [(169.7, 410, 695, 90, 90, 90),
     "I 2 2 2"],
 'F':  # 1r03
     [(181.41, 181.41, 181.41, 90, 90, 90),
     "F 4 3 2"],
 'P6':  # 4pgu
     [(62.207, 62.207, 293.474, 90, 90, 120),
     "P 65 2 2"],
 'P3':  # 3dxj
     [(235.09, 235.09, 250.88, 90, 90, 120),
     "P 32"],
 'P4':  # 1z35
     [(136.8, 136.8, 136.8, 90, 90, 90),
     "P 41 3 2"],
 'P2':  # 3nxs
     [(78.63, 91.46, 57.47, 90, 90, 90),
     "P 21 21 2"],
 'P1':  # 3l89
     [(94.53, 107.71, 154.1, 90.01, 90.1, 104.7),
     "P 1"]}

ucell_p, symbol = syms[args.sym]
ucell = uctbx.unit_cell(ucell_p)
F = db_utils.make_miller_array(unit_cell=ucell_p, symbol=symbol)

sg = F.space_group()
sgi = sg.info()
print("unit cell, space group:\n", F, "\n")

O = ucell.orthogonalization_matrix()
# real space vectors
a = O[0], O[3], O[6]
b = O[1], O[4], O[7]
c = O[2], O[5], O[8]
C_sg = Crystal(a,b,c,sg)
# this should have the identity as a Umat
assert np.allclose(C_sg.get_U(), (1,0,0,0,1,0,0,0,1))

# convert crystal to P1
to_p1_op = sgi.change_of_basis_op_to_primitive_setting()
C_p1 = C_sg.change_basis(to_p1_op)

# random orientation
Misset = sqr(Rotation.random(random_state=1).as_matrix().flatten())
# reorient the P1 crystal
U_p1 = sqr(C_p1.get_U())
C_p1.set_U(Misset*U_p1)

# make a detector and a beam
beam = BeamFactory.simple(wavelength=1)
detector = DetectorFactory.simple(
    sensor='PAD',
    distance=200,
    beam_centre=(51.25,51.25),
    fast_direction='+x',
    slow_direction='-y',
    pixel_size=(.1,.1),
    image_size=(1024,1024))

SIM = nanoBragg(detector=detector, beam=beam)
#SIM.printout_pixel_fastslow = 504, 422
SIM.xtal_shape = shapetype.Gauss_star
if args.gauss and args.square:
    raise RuntimeError("Can only have gauss, square or neither (default, which is Gauss_star)")
if args.gauss:
    SIM.xtal_shape = shapetype.Gauss
if args.square:
    SIM.xtal_shape = shapetype.Square

SIM.oversample = 1
SIM.interpolate = 0
# convert to P1 (and change basis depending on space group)
F_p1 = prep_miller_array(F)
SIM.Fhkl_tuple = F_p1.indices(), F_p1.data()
SIM.Amatrix = sqr(C_p1.get_A()).transpose()
if args.mos:
    SIM.mosaic_spread_deg = 1
    SIM.mosaic_domains = 10
assert np.allclose(SIM.unit_cell_tuple, C_p1.get_unit_cell().parameters())
SIM.Ncells_abc = 12, 12, 12
if args.fix:
    SIM.set_mosaic_blocks_sym(C_p1, reference_symbol=sgi.type().lookup_symbol(), orig_mos_domains=1)
getattr(SIM, add_spots_func)()
reference = SIM.raw_pixels.as_numpy_array()
print(SIM.fudge)
#exit()

#sanity_check_Fs(SIM, sg, C_p1.get_unit_cell().parameters())
print("USINGSHAPE:", SIM.xtal_shape)

ops = sg.build_derived_laue_group().all_ops()
ops = [o for o in ops if o.r().determinant() == 1]

results = []
for o in ops:
    sg_op = sgtbx.change_of_basis_op(o)
    C_o = C_sg.change_basis(sg_op).change_basis(to_p1_op)
    U = sqr(C_o.get_U())
    C_o.set_U(Misset*U)
    Amat = sqr(C_o.get_A())
    if args.fix:
        SIM.set_mosaic_blocks_sym(C_o, sgi.type().lookup_symbol(), orig_mos_domains=1)
    SIM.raw_pixels *= 0
    SIM.Amatrix = Amat.transpose()

    print("operator:", o.as_xyz())
    forms = literal_description(o)
    print(forms.long_form())
    getattr(SIM, add_spots_func)()
    img = SIM.raw_pixels.as_numpy_array()
    if not np.allclose(img, reference, atol=1e-7):
        passed=False
        print("FAILED\n")
        SIM.raw_pixels *= 0
        getattr(SIM, add_spots_func)()
    else:
        passed=True
        print("PASSED\n")
    results.append((passed, img, o))

SIM.free_all()
nfail = sum([not passed for passed,_,_ in results ])
nops = len(results)
if nfail > 0:
    print("Test failed for %d / %d ops." %(nfail, len(results)))
else:
    print("All tests pass (%d ops) ! " %(len(results)))
    print("Ok!")

if args.plot:
    try:
        from pylab import *
        from itertools import cycle
        imgs = cycle(results)
        m = reference[reference >0].mean()
        s = reference[reference>0].std()
        vmax=m+0.5*s
        fig, (ax1,ax2) = subplots(nrows=1, ncols=2, layout='constrained')
        fig.set_size_inches((8,4))
        ax1.imshow(reference, vmin=0, vmax=vmax)
        ax2.imshow(results[0][1], vmin=0, vmax=vmax)
        ax1.set_title("+x,+y,+z", fontsize=16)
        suptitle("%s" % str(sgi), fontsize=16)
        for ax in (ax1, ax2):
            ax.set_xlim(300, 700)
            ax.set_ylim(700, 300)
        while 1:
            if not fignum_exists(fig.number):
                break
            passed, img, op = next(imgs)
            ax2.images[0].set_data(img)
            pass_s = "passed" if passed else "failed"
            def check_sign(x):
                if not x.startswith("-"):
                    x = "+"+x
                return x
            x,y,z = map(check_sign, op.as_xyz().split(','))
            xyz = ",".join([x,y,z])

            ax2.set_title("%s (%s)" % (xyz, pass_s), fontsize=16)
            plt.draw()
            plt.pause(2)

    except KeyboardInterrupt:
        close()
        exit()
