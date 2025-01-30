from __future__ import division

from libtbx.phil import parse
from libtbx.utils import Sorry
from matplotlib import pyplot as plt
from serialtbx.util.energy_scan_notch_finder import notch_phil_string, find_notch, plot_notches, calibrate_energy

"""When an energy scan is conducted at LCLS, we acquire FEE spectra of the incident beam with a varying, known, narrow energy band removed -- the "notch". Energy calibration is the process of identifying the notch in each scan and using the known pixel-energy pairs to generate a function (linear fit) returning the energy for any given pixel position on the spectrometer (reported FEE energy in xtc streams). This file automates this process. Helper functions are located in serialtbx in case energy calibration can be useful outside the LCLS use case."""

fee_phil_string = """
experiment = None
  .type = str
  .help = experiment identifier at LCLS, e.g. mfxl1013621
verbose = False
  .type = bool
  .help = print all possible output
output_phil = None
  .type = path
  .help = path where calibrated values should be written as a phil file
max_events = 1000
  .type = int
  .help = use at most max_events per run
"""

phil_scope = parse(fee_phil_string + notch_phil_string)

def tally_fee_data(experiment, runs, plot=True, verbose=True, max_events=None):
  """Check each event of each requested run in the specified experiment for a FEE spectrometer event. Report how many events are missing. Return spectrometer data if present."""
  import psana

  good = 0
  bad = 0
  events = []
  rundata = []

  for r in runs:
    print(f"Processing run {r}...".format())
    ds = psana.DataSource(f'exp={experiment}:run={r}:idx'.format())
    d = psana.Detector('FEE-SPEC0')
    pr = list(ds.runs())[0]
    times = pr.times()
    data = None
    total = 0
    if max_events is None:
      iterable = range(len(times))
    else:
      iterable = range(min(len(times), max_events))
    for i in iterable:
      e = pr.event(times[i])
      f = d.get(e)
      if f:
        if verbose:
          print(r, i)
        good += 1
        events.append(1)
      else:
        if verbose:
          print(r, i, 'no fee')
        bad += 1
        events.append(0)
        continue
      if data is None:
        data = f.hproj().astype(float)
      else:
        data += f.hproj().astype(float)
      total += 1
    if verbose:
      print(total)
    data /= total
    rundata.append(data)
    print(f"Found {good} events with FEE, {bad} events without ({good+bad} total)".format())
  if plot:
    plt.plot(range(len(events)), events, '-')
    plt.title("FEE presence over time")
    plt.xlabel("Event number")
    plt.ylabel("FEE present (1 yes, 0 no)")
    plt.figure()
  return rundata

def run(args):
  user_phil = []
  runs = []
  energies = []
  for arg in args:
    if ':' in arg: # interpret as tuple of run number and known notch energy
      try:
        srun, senergy = arg.split(':')
        runs.append(int(srun))
        energy = energies.append(int(senergy))
      except Exception:
        raise Sorry("Run numbers and known energies must be supplied as colon-separated pairs (without spaces), e.g. \"5:9415 6:9405 7:9395\"")
    else:
      try:
        user_phil.append(parse(arg))
      except Exception:
        raise Sorry("Unrecognized argument %s"%arg)
  if not runs:
    raise Sorry("Run numbers and known energies must be supplied as colon-separated pairs (without spaces), e.g. \"5:9415 6:9405 7:9395\"")

  params = phil_scope.fetch(sources=user_phil).extract()
  rundata = tally_fee_data(params.experiment, runs, verbose=params.verbose, max_events=params.max_events)
  notches = [find_notch(range(len(data)),
                        data,
                        params.kernel_size,
                        params.fit_half_range,
                        params.baseline_cutoff,
                        ref_spectrum=params.reference_spectrum)
             for data in rundata]
  plot_notches(runs, rundata, notches, params.per_run_plots)
  try:
    eV_offset, eV_per_pixel = calibrate_energy(notches, energies)
    args_str = ' '.join(args)
    with open('fee_calib.out', 'a') as outfile:
      outfile.write(f'using {args_str}, eV_offset={eV_offset} eV_per_pixel={eV_per_pixel}\n')
    print('wrote calibrated values to fee_calib.out')
    if params.output_phil:
      with open(params.output_phil, 'w') as outfile:
        outfile.write(f'spectrum_eV_offset={eV_offset}\n')
        outfile.write(f'spectrum_eV_per_pixel={eV_per_pixel}\n')
      print(f'wrote calibrated values to {params.output_phil}')
  except SystemError as e:
    print(e)
  if params.per_run_plots:
    plt.show()

if __name__ == "__main__":
  import sys
  run(sys.argv[1:])
