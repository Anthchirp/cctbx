import os

def classify_histidine(hierarchy, resname='HIS'):
  from mmtbx.validation.rotalyze import rotalyze
  result = rotalyze(
      pdb_hierarchy=hierarchy,
      # data_version="8000",#was 'params.data_version', no options currently
      # show_errors=self.params.show_errors,
      # outliers_only=self.params.outliers_only,
      # use_parent=self.params.use_parent,
      # out=self.logger,
      quiet=False)
  names = []
  for rot in result.results:
    if rot.resname!=resname: continue
    names.append(rot.rotamer_name)
  hs=0
  ha=None
  for atom in hierarchy.atoms():
    if atom.parent().resname!=resname: continue
    if atom.name.strip() in ['HD1', 'HE2']:
      hs+=1
      ha=atom.name.strip()
  assert len(names)==1
  if hs==2: ha = 'HD1, HE2'
  return names[0], ha

def run_hbond(args):
  from iotbx.cli_parser import run_program
  from mmtbx.programs.hbond import Program
  hbonds = run_program(program_class=Program,
                       args=tuple(args),
                       )
  return hbonds

def run_serial_or_parallel(func, argstuples, nproc=1, log=None):
  import time
  from libtbx import easy_mp
  rc = []
  if nproc==1:
    for i, args in enumerate(argstuples):
      t0=time.time()
      print('  Running job %d' % (i+1), file=log)
      res = func(*args)
      rc.append(res)
      print('    Time : %0.1fs' % (time.time()-t0))
  elif nproc>1:
    print('  Running %d jobs on %d procs' % (len(argstuples), nproc), file=log)
    i=0
    t0=time.time()
    for args, res, err_str in easy_mp.multi_core_run( func,
                                                      argstuples,
                                                      nproc,
                                                      keep_input_order=True):
      assert not err_str, '\n\nDebug in serial :\n%s' % err_str
      print('  Running job %d : %0.1fs' % (i+1, time.time()-t0), file=log)
      rc.append(res)
      i+=1
  return rc

def get_hbonds_via_filenames(filenames, nq_or_h, restraint_filenames=None):
  argstuples = []
  for i, filename in enumerate(filenames):
    assert os.path.exists(filename), '"%s"' % filename
    argstuples.append([[filename,
                       '--quiet',
                       'output_pymol_file=True',
                       'output_restraint_file=False',
                       'output_skew_kurtosis_plot=False',
                       'prefix=%s' % filename.replace('.pdb',''),
                       ]])
    if restraint_filenames:
      argstuples[-1][-1]+=self.restraint_filenames
  # print('  Running %d jobs in %d procs' % (len(argstuples), nproc), file=log)

  rc = run_serial_or_parallel(run_hbond, argstuples, nproc=6)

  i=0
  hbondss=[]
  pymols = ''
  for i, filename in enumerate(filenames):
    hbondss.append(rc[i])
    pf = filename.replace('.pdb', '.pml')
    assert os.path.exists(pf), 'file not found %s' % pf
    f=open(pf, 'a')
    f.write('\n')
    f.write('show sticks, resn %s\n' % nq_or_h)
    del f
    pymols += '  phenix.pymol %s &\n' % pf
  return hbondss, pymols

def get_rotamers_via_filenames(filenames, selection):
  from iotbx import pdb
  rotamers=[]
  for i, filename in enumerate(filenames):
    hierarchy = pdb.input(filename).construct_hierarchy()
    asc1 = hierarchy.atom_selection_cache()
    sel = asc1.selection(selection)
    hierarchy = hierarchy.select(sel)
    rc = classify_histidine(hierarchy)
    rotamers.append(rc[0])
    i+=1
  return rotamers
