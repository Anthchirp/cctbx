from __future__ import absolute_import, division, print_function

import os, time
import libtbx.load_env
from libtbx.utils import Sorry

from xfel.ui.db.trial import Trial
from xfel.ui.db.run import Run
from xfel.ui.db.rungroup import Rungroup
from xfel.ui.db.tag import Tag
from xfel.ui.db.job import Job, JobFactory
from xfel.ui.db.stats import Stats
from xfel.ui.db.experiment import Cell, Bin, Isoform, Event
from xfel.ui.db.dataset import Dataset, DatasetVersion
from xfel.ui.db.task import Task

from xfel.ui.db import get_db_connection
from six.moves import range
import six
from six.moves import zip

from xfel.command_line.experiment_manager import initialize as initialize_base

CACHED_CONNECT_TIMEOUT = 300

class initialize(initialize_base):
  expected_tables = ["run", "job", "rungroup", "trial", "tag", "run_tag", "event", "trial_rungroup",
                     "imageset", "imageset_event", "beam", "detector", "experiment",
                     "crystal", "cell", "cell_bin", "bin", "isoform", "rungroup_run",
                     "dataset", "dataset_version", "dataset_version_job", "dataset_tag",
                     "dataset_task", "task"]

  def __getattr__(self, prop):
    if prop == "dbobj":
      return get_db_connection(self.params)
    raise AttributeError("%s not found"%prop)

  def __setattr__(self, prop, val):
    if prop == "dbobj": pass
    return super(initialize, self).__setattr__(prop, val)

  def __init__(self, params, dbobj):
    initialize_base.__init__(self, params, dbobj, interactive = False, drop_tables = None)

  def create_tables(self, sql_path = None):
    if sql_path is None:
      sql_path = os.path.join(libtbx.env.find_in_repositories("xfel/ui/db"), "schema.sql")

    return initialize_base.create_tables(self, sql_path)

  def verify_tables(self):
    self.expected_tables.pop(self.expected_tables.index('rungroup_run'))
    tables_ok = super(initialize, self).verify_tables()
    self.expected_tables.append('rungroup_run')

    if tables_ok:
      # Maintain backwards compatibility with SQL tables v2: 09/24/16
      query = "SHOW columns FROM `%s_event`"%self.params.experiment_tag
      cursor = self.dbobj.cursor()
      cursor.execute(query)
      columns = cursor.fetchall()
      column_names = list(zip(*columns))[0]
      if 'two_theta_low' not in column_names and 'two_theta_high' not in column_names:
        query = """
          ALTER TABLE %s_event
          ADD COLUMN two_theta_low DOUBLE NULL,
          ADD COLUMN two_theta_high DOUBLE NULL
        """%self.params.experiment_tag
        cursor.execute(query)
      elif 'two_theta_low' in column_names and 'two_theta_high' in column_names:
        pass
      else:
        assert False

      # Maintain backwards compatibility with SQL tables v2: 09/28/16
      query = "SHOW columns FROM `%s_job`"%self.params.experiment_tag
      cursor = self.dbobj.cursor()
      cursor.execute(query)
      columns = cursor.fetchall()
      column_names = list(zip(*columns))[0]
      if 'submission_id' not in column_names:
        query = """
          ALTER TABLE %s_job
          ADD COLUMN submission_id VARCHAR(45) NULL
        """%self.params.experiment_tag
        cursor.execute(query)

      # Maintain backwards compatibility with SQL tables v2: 12/12/16
      query = "SHOW columns FROM `%s_rungroup`"%self.params.experiment_tag
      cursor = self.dbobj.cursor()
      cursor.execute(query)
      columns = cursor.fetchall()
      column_names = list(zip(*columns))[0]
      for needed_column, column_format in zip(['format', 'two_theta_low', 'two_theta_high'],
                                              ["VARCHAR(45) NOT NULL DEFAULT 'pickle'",
                                               "DOUBLE NULL", "DOUBLE NULL"]):
        if needed_column not in column_names:
          query = """
            ALTER TABLE %s_rungroup
            ADD COLUMN %s %s
          """%(self.params.experiment_tag, needed_column, column_format)
          cursor.execute(query)

      # Maintain backwards compatibility with SQL tables v2: 06/23/17
      query = "SHOW columns FROM `%s_trial`"%self.params.experiment_tag
      cursor = self.dbobj.cursor()
      cursor.execute(query)
      columns = cursor.fetchall()
      column_names = list(zip(*columns))[0]
      if 'd_min' not in column_names:
        query = """
          ALTER TABLE `%s_trial`
          ADD COLUMN d_min FLOAT NULL
        """%self.params.experiment_tag
        cursor.execute(query)
      query = "SHOW columns FROM `%s_cell`"%self.params.experiment_tag
      cursor = self.dbobj.cursor()
      cursor.execute(query)
      columns = cursor.fetchall()
      column_names = list(zip(*columns))[0]
      if 'trial_id' not in column_names:
        query = """
          ALTER TABLE `%s_cell`
          ADD COLUMN trial_id INT NULL,
          ADD CONSTRAINT fk_cell_trial1 FOREIGN KEY (trial_id) REFERENCES `%s_trial` (id) ON DELETE NO ACTION,
          ADD INDEX fk_cell_trial1_idx (trial_id ASC)
        """%(self.params.experiment_tag, self.params.experiment_tag)
        cursor.execute(query)

      # Maintain backwards compatibility with SQL tables v3: 02/1/19
      query = "SHOW TABLES LIKE '%s_rungroup_run'"%(self.params.experiment_tag)
      cursor.execute(query)
      if cursor.rowcount == 0:
        print("Upgrading to version 4 of mysql database schema")
        query = """
        CREATE TABLE IF NOT EXISTS `%s`.`%s_rungroup_run` (
          `rungroup_id` INT NOT NULL,
          `run_id` INT NOT NULL,
          PRIMARY KEY (`rungroup_id`, `run_id`),
          INDEX `fk_rungroup_has_run_run1_idx` (`run_id` ASC),
          INDEX `fk_rungroup_has_run_rungroup1_idx` (`rungroup_id` ASC),
          CONSTRAINT `fk_%s_rungroup_has_run_rungroup1`
            FOREIGN KEY (`rungroup_id`)
            REFERENCES `%s`.`%s_rungroup` (`id`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION,
          CONSTRAINT `fk_%s_rungroup_has_run_run1`
            FOREIGN KEY (`run_id`)
            REFERENCES `%s`.`%s_run` (`id`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION)
        ENGINE = InnoDB;
        """%(self.params.db.name, self.params.experiment_tag, self.params.experiment_tag,
             self.params.db.name, self.params.experiment_tag, self.params.experiment_tag,
             self.params.db.name, self.params.experiment_tag)
        cursor.execute(query)
        # Convert from startrun/endrun to linked table connecting run and rungroup and an 'open' flag in rungroup
        query = "ALTER TABLE `%s_rungroup` ADD COLUMN open TINYINT(1) NOT NULL DEFAULT 0"%self.params.experiment_tag
        cursor.execute(query)
        query = "SELECT id, startrun, endrun FROM `%s_rungroup`"%self.params.experiment_tag
        cursor.execute(query)
        for rungroup_id, startrun, endrun in cursor.fetchall():
          if endrun is None:
            # This run is 'open', so get the last run available and update the open bit for the rungroup
            query = 'SELECT run FROM `%s_run`'%self.params.experiment_tag
            cursor.execute(query)
            endrun = max(list(zip(*cursor.fetchall()))[0])
            query = 'UPDATE `%s_rungroup` set open = 1 where id = %d'%(self.params.experiment_tag, rungroup_id)
            cursor.execute(query)
          # Add all thr runs to the rungroup
          for run in range(startrun, endrun+1):
            query = 'SELECT id FROM `%s_run` run WHERE run.run = %d'%(self.params.experiment_tag, run)
            cursor.execute(query)
            rows = cursor.fetchall(); assert len(rows) <= 1
            if len(rows) > 0:
              run_id = rows[0][0]
              query = "INSERT INTO `%s_rungroup_run` (rungroup_id, run_id) VALUES (%d, %d)" % ( \
                self.params.experiment_tag, rungroup_id, run_id)
              cursor.execute(query)
        self.dbobj.commit()
        query = "ALTER TABLE `%s_rungroup` DROP COLUMN startrun, DROP COLUMN endrun"%self.params.experiment_tag
        cursor.execute(query)
        # remove NOT NULL (and default for format column)
        query = "ALTER TABLE `%s_rungroup` MODIFY COLUMN format varchar(45)"%self.params.experiment_tag
        cursor.execute(query)
        query = "ALTER TABLE `%s_rungroup` MODIFY COLUMN detector_address varchar(100)"%self.params.experiment_tag
        cursor.execute(query)
        query = "ALTER TABLE `%s_rungroup` MODIFY COLUMN detz_parameter double"%self.params.experiment_tag
        cursor.execute(query)
        # Retype and add new columns
        query = "ALTER TABLE `%s_run` MODIFY COLUMN run varchar(45) NOT NULL"%self.params.experiment_tag
        cursor.execute(query)
        query = "ALTER TABLE `%s_run` ADD COLUMN path varchar(4097)"%self.params.experiment_tag
        cursor.execute(query)
        column_names = ['number', 'd_min', 'd_max', 'total_hkl']
        column_types = ['int', 'double', 'double', 'int']
        for column_name, column_type in zip(column_names, column_types):
          query = "ALTER TABLE `%s_bin` MODIFY COLUMN %s %s"%(self.params.experiment_tag, column_name, column_type)
          cursor.execute(query)

      # Maintain backwards compatibility with SQL tables v4: 11/06/19
      query = "SHOW columns FROM `%s_job`"%self.params.experiment_tag
      cursor = self.dbobj.cursor()
      cursor.execute(query)
      columns = cursor.fetchall()
      column_names = list(zip(*columns))[0]
      if 'dataset_id' not in column_names:
        print("Upgrading to version 5 of mysql database schema")
        query = "ALTER TABLE `%s_job` DROP PRIMARY KEY, ADD PRIMARY KEY (`id`)"%self.params.experiment_tag
        cursor.execute(query)
        query = "ALTER TABLE `%s_job` MODIFY COLUMN run_id INT NULL"%self.params.experiment_tag
        cursor.execute(query)
        query = "ALTER TABLE `%s_job` MODIFY COLUMN rungroup_id INT NULL"%self.params.experiment_tag
        cursor.execute(query)
        query = "ALTER TABLE `%s_job` MODIFY COLUMN trial_id INT NULL"%self.params.experiment_tag
        cursor.execute(query)
        query = "ALTER TABLE `%s_job` ADD COLUMN task_id INT NULL"%self.params.experiment_tag
        cursor.execute(query)
        query = "ALTER TABLE `%s_job` ADD COLUMN dataset_id INT NULL"%self.params.experiment_tag
        cursor.execute(query)
        query = """
        ALTER TABLE `%s_job`
          ADD CONSTRAINT `fk_job_task1`
            FOREIGN KEY (`task_id`)
            REFERENCES `%s`.`%s_task` (`id`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION
        """%(self.params.experiment_tag, self.params.db.name, self.params.experiment_tag)
        cursor.execute(query)
        query = """
        ALTER TABLE `%s_job`
          ADD CONSTRAINT `fk_job_dataset1`
            FOREIGN KEY (`dataset_id`)
            REFERENCES `%s`.`%s_dataset` (`id`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION
        """%(self.params.experiment_tag, self.params.db.name, self.params.experiment_tag)
        cursor.execute(query)
        query = "ALTER TABLE `%s_job` ADD INDEX `fk_job_task1_idx` (`task_id` ASC)"%self.params.experiment_tag
        cursor.execute(query)
        query = "ALTER TABLE `%s_job` ADD INDEX `fk_job_dataset1_idx` (`dataset_id` ASC)"%self.params.experiment_tag
        cursor.execute(query)

      # Maintain backwards compatibility with SQL tables v5: 10/09/20
      query = "SHOW columns FROM `%s_rungroup`"%self.params.experiment_tag
      cursor = self.dbobj.cursor()
      cursor.execute(query)
      columns = cursor.fetchall()
      column_names = list(zip(*columns))[0]
      if 'wavelength_offset' not in column_names:
        print("Upgrading to version 5.1 of mysql database schema")
        query = "ALTER TABLE `%s_rungroup` ADD COLUMN wavelength_offset DOUBLE NULL"%self.params.experiment_tag
        cursor.execute(query)

      # Maintain backwards compatibility with SQL tables v5: 12/11/20
      query = "SHOW columns FROM `%s_rungroup`"%self.params.experiment_tag
      cursor = self.dbobj.cursor()
      cursor.execute(query)
      columns = cursor.fetchall()
      column_names = list(zip(*columns))[0]
      if 'spectrum_eV_per_pixel' not in column_names:
        print("Upgrading to version 5.2 of mysql database schema")
        query = "ALTER TABLE `%s_rungroup` ADD COLUMN spectrum_eV_per_pixel DOUBLE NULL"%self.params.experiment_tag
        cursor.execute(query)
        query = "ALTER TABLE `%s_rungroup` ADD COLUMN spectrum_eV_offset DOUBLE NULL"%self.params.experiment_tag
        cursor.execute(query)

      # Maintain backwards compatibility with SQL tables v5.3: 07/23/21
      if 'extra_format_str' not in column_names:
        print("Upgrading to version 5.3 of mysql database schema")
        query = "ALTER TABLE `%s_rungroup` ADD COLUMN extra_format_str TEXT NULL"%self.params.experiment_tag
        cursor.execute(query)
        query = "ALTER TABLE `%s_job` MODIFY COLUMN submission_id TEXT NULL"%self.params.experiment_tag
        cursor.execute(query)
    return tables_ok

  def set_up_columns_dict(self, app):
    columns_dict = {}
    for table in self.expected_tables:
      table_name = "%s_%s" % (self.params.experiment_tag, table)
      query = "SHOW COLUMNS FROM `%s`" % (table_name)
      cursor = app.execute_query(query)
      columns_dict[table_name] = [c[0] for c in cursor.fetchall() if c[0] != 'id']
    return columns_dict

class dummy_cursor(object):
  def __init__(self, sql_cursor):
    self.rowcount = sql_cursor.rowcount
    self.lastrowid = sql_cursor.lastrowid
    self.prefetched = sql_cursor.fetchall()
  def fetchall(self):
    return self.prefetched

class db_application(object):
  def __init__(self, params, cache_connection = True, mode = 'execute'):
    self.params = params
    self.dbobj = None
    self.dbobj_refreshed_time = None
    self.cache_connection = cache_connection
    self.query_count = 0
    self.mode = mode
    self.last_query = None

  def __setattr__(self, prop, val):
    if prop == "mode":
      assert val in ['execute', 'cache_commits']
    return super(db_application, self).__setattr__(prop, val)

  def execute_query(self, query, commit=True):
    from MySQLdb import OperationalError

    if self.mode == 'cache_commits' and commit:
      self.last_query = query
      return

    if self.params.db.verbose:
      st = time.time()
      self.query_count += 1

    retry_count = 0
    retry_max = 10
    sleep_time = 0.1
    while retry_count < retry_max:
      try:

        # Get the (maybe cached) connection
        # We enable autocommit on the connection by default, to avoid stale
        # reads arising from unclosed transactions. See:
        # https://stackoverflow.com/questions/1617637/pythons-mysqldb-not-getting-updated-row
        if not commit: # connection caching is not attempted if commit=False
          dbobj = get_db_connection(self.params, autocommit=False)
        elif (
            self.dbobj is None
            or time.time() - self.dbobj_refreshed_time > CACHED_CONNECT_TIMEOUT
        ):
          dbobj = get_db_connection(self.params, autocommit=True)
          self.dbobj_refreshed_time = time.time()
          if self.cache_connection:
            self.dbobj = dbobj
        else:
          dbobj = self.dbobj

        sql_cursor = dbobj.cursor()
        sql_cursor.execute(query)
        cursor = dummy_cursor(sql_cursor)
        sql_cursor.close()

        if self.params.db.verbose:
          et = time.time() - st
          if et > 1:
            print('Query % 6d SQLTime Taken = % 10.6f seconds' % (self.query_count, et), query[:min(len(query),145)])
        return cursor
      except OperationalError as e:
        reconnect_strings = [
            "MySQL server has gone away",
            "max_user_connections",
            "is not allowed to connect to this MariaDB server",
        ]
        retry_strings = [
            "Can't connect to MySQL server",
            "Lost connection to MySQL server",
            "Deadlock found when trying to get lock",
            "WSREP has not yet prepared node for application use",
        ]
        if any([s in str(e) for s in reconnect_strings]):
          self.dbobj = None
        elif all([s not in str(e) for s in retry_strings]):
          print(query)
          raise e
        retry_count += 1
        print("Couldn't connect to MYSQL, retry", retry_count)
        time.sleep(sleep_time)
        sleep_time *= 2
      except Exception as e:
        print("Couldn't execute MYSQL query.  Query:")
        print(query)
        print("Exception:")
        print(str(e))
        raise e
    raise Sorry("Couldn't execute MYSQL query. Too many reconnects. Query: %s"%query)

class xfel_db_application(db_application):
  def __init__(self, params, drop_tables = False, verify_tables = False, **kwargs):
    super(xfel_db_application, self).__init__(params, **kwargs)
    dbobj = get_db_connection(params)
    init_tables = initialize(params, dbobj)

    if drop_tables:
      init_tables.drop_tables()

    if verify_tables and not init_tables.verify_tables():
      init_tables.create_tables()
      print('Creating experiment tables...')
      if not init_tables.verify_tables():
        raise Sorry("Couldn't create experiment tables")

    self.columns_dict = init_tables.set_up_columns_dict(self)

  def list_lcls_runs(self):
    if self.params.facility.lcls.web.location is None or len(self.params.facility.lcls.web.location) == 0:
      from xfel.command_line.auto_submit import match_runs
      import os
      exp_prefix = self.params.facility.lcls.experiment[0:3].upper()
      xtc_dir = os.path.join(os.environ.get('SIT_PSDM_DATA', '/reg/d/psdm'), exp_prefix, self.params.facility.lcls.experiment, 'xtc')
      return [{'run':str(r.id)} for r in sorted(match_runs(xtc_dir, False), key=lambda x:x.id)]
    else:
      import json
      from six.moves import urllib
      basequery = f"{self.params.facility.lcls.api.protocol}://{self.params.facility.lcls.api.host}/ws/lgbk/lgbk/%s/ws/files_for_live_mode_at_location?location=%s"
      query = basequery%(self.params.facility.lcls.experiment, self.params.facility.lcls.web.location)
      R = urllib.request.urlopen(query)
      if R.getcode() != 200:
        print ("Couldn't connect to LCLS webservice to list runs, code", R.getcode())
        return []

      j = json.loads(R.read())
      if not j.get('success'):
        print("Web service query to list runs failed")
        return []

      present_runs = []
      for r in sorted(j['value'], key=lambda x:x['run_num']):
        if r['all_present']:
          is_good = True
        else:
          if not self.params.facility.lcls.web.enforce80:
            for item_idx, item in enumerate(r['files']):
              if '-s80-' in item['path']:
                item['is_present'] = True
          if not self.params.facility.lcls.web.enforce81:
            for item_idx, item in enumerate(r['files']):
              if '-s81-' in item['path']:
                item['is_present'] = True
          is_good = all([f['is_present'] for f in r['files']])
        if is_good:
          present_runs.append({'run':str(int(r['run_num']))})

      return present_runs


  def create_trial(self, d_min = 1.5, n_bins = 10, **kwargs):
    # d_min and n_bins only used if isoforms are in this trial

    trial = Trial(self, d_min = d_min, **kwargs)
    if trial.target_phil_str is not None:
      from iotbx.phil import parse
      dispatcher = self.params.dispatcher
      if dispatcher == 'cxi.xtc_process':
        from spotfinder.applications.xfel import cxi_phil
        trial_params = cxi_phil.cxi_versioned_extract().persist.phil_scope.fetch(parse(trial.target_phil_str)).extract()
        isoforms = trial_params.isoforms
      else:
        from xfel.ui import load_phil_scope_from_dispatcher
        phil_scope = load_phil_scope_from_dispatcher(self.params.dispatcher)
        trial_params = phil_scope.fetch(parse(trial.target_phil_str)).extract()
        isoforms = trial_params.indexing.stills.isoforms
      if len(isoforms) > 0:
        for isoform in isoforms:
          print("Creating isoform", isoform.name)
          db_isoform = Isoform(self,
                               name = isoform.name,
                               trial_id = trial.id)
          a, b, c, alpha, beta, gamma = isoform.cell.parameters()
          cell = self.create_cell(cell_a = a, cell_b = b, cell_c = c,
                                  cell_alpha = alpha, cell_beta = beta, cell_gamma = gamma,
                                  lookup_symbol = isoform.lookup_symbol,
                                  isoform_id = db_isoform.id)
          from cctbx.crystal import symmetry

          cs = symmetry(unit_cell = isoform.cell,space_group_symbol=str(isoform.lookup_symbol))
          mset = cs.build_miller_set(anomalous_flag=False, d_min=d_min)
          binner = mset.setup_binner(n_bins=n_bins)
          for i in binner.range_used():
            d_max, d_min = binner.bin_d_range(i)
            Bin(self, number = i, d_min = d_min, d_max = d_max,
                total_hkl = binner.counts_complete()[i], cell_id = cell.id)
      elif dispatcher == 'cxi.xtc_process':
        pass # TODO: labelit target
      else:
        if trial_params.indexing.known_symmetry.unit_cell is not None and \
            trial_params.indexing.known_symmetry.space_group is not None:
          print("Creating target cell")
          unit_cell = trial_params.indexing.known_symmetry.unit_cell
          symbol = str(trial_params.indexing.known_symmetry.space_group)
          a, b, c, alpha, beta, gamma = unit_cell.parameters()
          cell = self.create_cell(cell_a = a, cell_b = b, cell_c = c,
                                  cell_alpha = alpha, cell_beta = beta, cell_gamma = gamma,
                                  lookup_symbol = symbol,
                                  trial_id = trial.id)
          from cctbx.crystal import symmetry

          cs = symmetry(unit_cell = unit_cell, space_group_symbol = symbol)
          mset = cs.build_miller_set(anomalous_flag=False, d_min=d_min)
          binner = mset.setup_binner(n_bins=n_bins)
          for i in binner.range_used():
            d_max, d_min = binner.bin_d_range(i)
            Bin(self, number = i, d_min = d_min, d_max = d_max,
                total_hkl = binner.counts_complete()[i], cell_id = cell.id)
    return trial

  def get_trial_isoforms(self, trial_id):
    where = "WHERE trial_id = %d"%trial_id
    return self.get_all_x(Isoform, "isoform", where)

  def get_trial_cell(self, trial_id):
    where = "WHERE trial_id = %d"%trial_id
    cells = self.get_all_x(Cell, "cell", where)
    assert len(cells) <= 1
    if len(cells) == 0:
      return None
    else:
      return cells[0]

  def get_trial_cells(self, trial_id, rungroup_id = None, run_id = None):
    # Use big queries to assist listing lots of cells. Start with list of cells for this trial
    tag = self.params.experiment_tag
    if rungroup_id is not None or run_id is not None:
      assert rungroup_id is not None and run_id is not None
      extra_where = "AND evt.run_id = %d AND evt.rungroup_id = %d"%(run_id, rungroup_id)
    else:
      extra_where = ""

    where  = """JOIN `%s_crystal` crystal ON crystal.cell_id = cell.id
               JOIN `%s_experiment` expt ON expt.crystal_id = crystal.id
               JOIN `%s_imageset` imgset ON imgset.id = expt.imageset_id
               JOIN `%s_imageset_event` is_e ON is_e.imageset_id = imgset.id
               JOIN `%s_event` evt ON evt.id = is_e.event_id
               JOIN `%s_trial` trial ON evt.trial_id = trial.id
               JOIN `%s_rungroup` rg ON evt.rungroup_id = rg.id
               WHERE trial.id = %d AND rg.active = True %s""" % (
               tag, tag, tag, tag, tag, tag, tag, trial_id, extra_where)
    cells = self.get_all_x(Cell, 'cell', where)
    where = " JOIN `%s_cell` cell ON bin.cell_id = cell.id "%(tag) + where
    return self.link_cell_bins(cells, where = where)

  def link_cell_bins(self, cells, where = None):
    tag = self.params.experiment_tag
    cells_d = {cell.id:cell for cell in cells}

    # Get all the bin ids for bins associated with these cells and assemble the bin objects
    if where is None:
      cell_ids = ", ".join([str(key) for key in cells_d.keys()])
      where = """ WHERE bin.cell_id IN (%s)""" % (cell_ids)
    bins = self.get_all_x(Bin, 'bin', where=where)

    # Link in all the bins
    for bin in bins:
      if bin.cell_id in cells_d: # might not be there if new data has arrived
        cells_d[bin.cell_id]._bins.append(bin)

    for cell in cells:
      cell._bins_set = True

    return cells

  def create_cell(self, **kwargs):
    return Cell(self, **kwargs)

  def get_cell(self, cell_id = None, name = None):
    assert [cell_id, name].count(None) == 1
    if name is not None:
      query = "SELECT id FROM `%s_cell` WHERE name = '%s'"%(self.params.experiment_tag, name)
      cursor = self.execute_query(query)
      results = cursor.fetchall()
      assert len(results) in [0,1]
      if len(results) == 0:
        return None
      cell_id = int(results[0][0])

    return Cell(self, cell_id=cell_id)

  def get_cell_bins(self, cell_id):
    query = "SELECT id FROM `%s_bin` WHERE cell_id = %d" % \
            (self.params.experiment_tag, cell_id)
    cursor = self.execute_query(query)
    ids = [str(i[0]) for i in cursor.fetchall()]
    if len(ids) == 0:
      return []
    where = "WHERE id IN (%s)" % ", ".join(ids)
    return self.get_all_x(Bin, 'bin', where)

  def get_all_x(self, cls, name, where = None):
    table_name = "%s_%s" % (self.params.experiment_tag, name)
    columns = ["%s.%s"%(name, c) for c in self.columns_dict[table_name]]
    query = "SELECT %s.id, %s FROM `%s` %s" % (name, ", ".join(columns), table_name, name)
    if where is not None:
      query += " " + where
    cursor = self.execute_query(query)
    results = []
    for row in cursor.fetchall():
      d = {key:value for key, value in zip(self.columns_dict[table_name], row[1:])}
      d["%s_id"%name] = row[0]
      results.append(cls(self, **d))
    return results

  def get_all_x_with_subitems(self, cls, name, where = None, sub_items = None):
    """ Assemble a list of db_proxy objects, where each one references a sub object
        @param cls principal class
        @param name table name not including experiment tag
        @param where optional constraints on what to get
        @param sub_items: array of tuples (cls,name) of items this table references
        @return list of assembled db_proxy objects"""

    # Assemble a list of all columns to be extracted via a big join. Then entries in the
    # columns array will be of the form "name.column", IE: tag.comment
    table_name = "%s_%s" % (self.params.experiment_tag, name)
    columns = ["%s.%s"%(name, c) for c in self.columns_dict[table_name]]
    columns.append("%s.id"%name)

    if sub_items is None:
      sub_items = []
    sub_table_names = ["%s_%s"%(self.params.experiment_tag, i[1]) for i in sub_items]
    for i, sub_item in enumerate(sub_items):
      scls, sname, required = sub_item
      columns.extend(["%s.%s"%(sname, c) for c in self.columns_dict[sub_table_names[i]]])
      columns.append("%s.id"%sname)

    # the main item being extracted is in the FROM statement and is given a nickname which
    # is the table name without the experiment tag
    query = "SELECT %s FROM `%s` %s" % (", ".join(columns), table_name, name)

    # Join statements to bring in the sub tables
    for i, sub_item in enumerate(sub_items):
      scls, sname, required = sub_item
      if required:
        join = " JOIN "
      else:
        join = " LEFT OUTER JOIN " # allows nulls
      query += join + "`%s` %s ON %s.id = %s.%s_id"% (
        sub_table_names[i], sname, sname, name, sname)

    if where is not None:
      query += " " + where
    cursor = self.execute_query(query)

    results = []
    sub_ds = {sub_item[1]:(sub_item[0], {}) for sub_item in sub_items}
    sub_reqds = {sub_item[1]:sub_item[2] for sub_item in sub_items}

    for row in cursor.fetchall():
      # Each row will be a complete item and sub items in column form. Assemble one
      # dictionary (d) for the main item and a dictionary of dictionaries (sub_ds)
      # for each of the sub items
      d = {}
      for key, value in zip(columns, row):
        n, c = key.split('.') # nickname n, column name c
        if n == name:
          d[c] = value # this column came from the main table
        else:
          sub_ds[n][1][c] = value # this column came from a sub table

      if 'task' in sub_ds:
        d['task_type'] = sub_ds['task'][1]['type']

      # pop the id column as it is passed as name_id to the db_proxy class (ie Job(job_id = 2))
      _id = d.pop("id")
      d["%s_id"%name] = _id
      results.append(cls(self, **d)) # instantiate the main class
      for sub_d_n, sub_d in six.iteritems(sub_ds):
        _id = sub_d[1].pop("id")
        if _id is None:
          assert not sub_reqds[sub_d_n]
          continue
        sub_d[1]["%s_id"%sub_d_n] = _id
        setattr(results[-1], sub_d_n, sub_d[0](self, **sub_d[1])) # instantiate the sub items
    return results

  def get_trial(self, trial_id = None, trial_number = None):
    assert [trial_id, trial_number].count(None) == 1
    if trial_id is None:
      trials = [t for t in self.get_all_trials() if t.trial == trial_number]
      assert len(trials) == 1
      return trials[0]
    else:
      return Trial(self, trial_id)

  def get_trial_rungroups(self, trial_id, only_active = False):
    tag = self.params.experiment_tag
    if only_active:
      query = """SELECT t_rg.rungroup_id FROM `%s_trial_rungroup` t_rg
                 JOIN `%s_rungroup` rg ON rg.id = t_rg.rungroup_id
                 WHERE t_rg.trial_id = %d AND rg.active = True""" % (tag, tag, trial_id)
    else:
      query = """SELECT t_rg.rungroup_id FROM `%s_trial_rungroup` t_rg
                 WHERE t_rg.trial_id = %d""" % (tag, trial_id)
    cursor = self.execute_query(query)
    rungroup_ids = ["%d"%i[0] for i in cursor.fetchall()]
    if len(rungroup_ids) == 0:
      return []
    return self.get_all_x(Rungroup, "rungroup", where = "WHERE rungroup.id IN (%s)"%",".join(rungroup_ids))

  def get_trial_runs(self, trial_id):
    tag = self.params.experiment_tag
    query = """SELECT run.id FROM `%s_run` run
               JOIN `%s_rungroup_run` rgr ON run.id = rgr.run_id
               JOIN `%s_rungroup` rg ON rg.id = rgr.rungroup_id
               JOIN `%s_trial_rungroup` t_rg on t_rg.rungroup_id = rg.id
               JOIN `%s_trial` trial ON trial.id = t_rg.trial_id
               WHERE trial.id=%d AND rg.active=True
               """ %(tag, tag, tag, tag, tag, trial_id)
    cursor = self.execute_query(query)
    run_ids = ["%d"%i[0] for i in cursor.fetchall()]
    if len(run_ids) == 0:
      return []

    use_ids = self.params.facility.name not in ['lcls']
    if use_ids:
      key = lambda x: x.id
    else:
      key = lambda x: int(x.run)

    return sorted(self.get_all_x(Run, "run", where = "WHERE run.id IN (%s)"%",".join(run_ids)), key=key)

  def get_trial_tags(self, trial_id):
    tag = self.params.experiment_tag
    query = """SELECT tag.id FROM `%s_tag` tag
               JOIN `%s_run_tag` rt ON rt.tag_id = tag.id
               JOIN `%s_run` run ON run.id = rt.run_id
               JOIN `%s_rungroup_run` rgr ON run.id = rgr.run_id
               JOIN `%s_rungroup` rg ON rg.id = rgr.rungroup_id
               JOIN `%s_trial_rungroup` t_rg ON t_rg.rungroup_id = rg.id
               JOIN `%s_trial` trial ON trial.id = t_rg.trial_id
               WHERE trial.id=%d AND rg.active=True
               """ % (tag, tag, tag, tag, tag, tag, tag, trial_id)
    cursor = self.execute_query(query)
    tag_ids = ["%d"%i[0] for i in cursor.fetchall()]
    if len(tag_ids) == 0:
      return []
    return self.get_all_x(Tag, "tag", where = "WHERE tag.id IN (%s)"%",".join(tag_ids))

  def get_all_trials(self, only_active = False):
    if only_active:
      return [t for t in self.get_all_x(Trial, "trial") if t.active]
    else:
      return self.get_all_x(Trial, "trial")

  def create_run(self, **kwargs):
    return Run(self, **kwargs)

  def get_run(self, run_id = None, run_number = None):
    assert [run_id, run_number].count(None) == 1
    if run_id is not None:
      return Run(self, run_id)
    runs = [r for r in self.get_all_runs() if r.run == str(run_number)]
    if len(runs) == 0:
      raise Sorry("Couldn't find run %s"%run_number)
    assert len(runs) == 1
    return runs[0]

  def get_all_runs(self):
    return self.get_all_x(Run, "run")

  def get_rungroup_runs_by_tags(self, rungroup, tags, mode):
    tag = self.params.experiment_tag
    tagids = [t.id for t in tags]
    if mode == "union":
      # Union is done by using a series of OR statements testing the tag ids
      return self.get_all_x(Run, 'run', where = """
        JOIN `%s_rungroup_run` rg_r ON rg_r.run_id = run.id
        JOIN `%s_rungroup` rg ON rg.id = rg_r.rungroup_id
        JOIN `%s_run_tag` rt ON rt.run_id = run.id
        JOIN `%s_tag` tag ON tag.id = rt.tag_id
        WHERE rg.id = %d AND (%s)
         """%(tag, tag, tag, tag, rungroup.id, " OR ".join(["tag.id = %d"%i for i in tagids])))
    elif mode == "intersection":
      # Intersection is done using a series of INNER JOINS, one full set for each tag
      return self.get_all_x(Run, 'run', where = """
        %s
        WHERE %s
        """%("".join(["""
                      INNER JOIN `%s_rungroup_run` rg_r%d ON rg_r%d.run_id = run.id
                      INNER JOIN `%s_rungroup` rg%d ON rg%d.id = rg_r%d.rungroup_id
                      INNER JOIN `%s_run_tag` rt%d ON rt%d.run_id = run.id
                      INNER JOIN `%s_tag` tag%d ON tag%d.id = rt%d.tag_id AND tag%d.id = %d"""%(
                      tag, i, i, tag, i, i, i, tag, i, i, tag, i, i, i, i, tid) for i, tid in enumerate(tagids)]),
        " AND ".join(["rg%d.id = %d"%(i, rungroup.id) for i in range(len(tagids))])))

  def create_rungroup(self, **kwargs):
    return Rungroup(self, **kwargs)

  def get_rungroup(self, rungroup_id):
    return Rungroup(self, rungroup_id)

  def get_rungroup_runs(self, rungroup_id):
    query = """SELECT run.id FROM `%s_run` run
               JOIN `%s_rungroup_run` rgr ON run.id = rgr.run_id
               WHERE rgr.rungroup_id = %d"""%(self.params.experiment_tag,
                 self.params.experiment_tag, rungroup_id)
    cursor = self.execute_query(query)
    run_ids = ["%d"%i[0] for i in cursor.fetchall()]
    if len(run_ids) == 0:
      return []
    return self.get_all_x(Run, "run", where = "WHERE run.id IN (%s)"%",".join(run_ids))

  def get_all_rungroups(self, only_active = False):
    if only_active:
      return [rg for rg in self.get_all_x(Rungroup, "rungroup") if rg.active]
    else:
      return self.get_all_x(Rungroup, "rungroup")

  def create_tag(self, **kwargs):
    return Tag(self, **kwargs)

  def get_tag(self, tag_id):
    return Tag(self, tag_id)

  def get_run_tags(self, run_id):
    query = "SELECT tag_id from `%s_run_tag` WHERE `%s_run_tag`.run_id = %d" % \
            (self.params.experiment_tag, self.params.experiment_tag, run_id)
    cursor = self.execute_query(query)
    tag_ids = [str(i[0]) for i in cursor.fetchall()]
    if len(tag_ids) == 0:
      return []
    where = "WHERE id IN (%s)" % ", ".join(tag_ids)
    return self.get_all_x(Tag, 'tag', where)

  def get_all_tags(self):
    return self.get_all_x(Tag, "tag")

  def delete_x(self, item, item_id):
    query = "DELETE FROM `%s` WHERE id = %d"%(item.table_name, item_id)
    self.execute_query(query, commit = True)

  def delete_tag(self, tag = None, tag_id = None):
    assert [tag, tag_id].count(None) == 1

    if tag_id is None:
      tag_id = tag.tag_id
    else:
      tag = self.get_tag(tag_id)

    query = "DELETE FROM `%s_run_tag` WHERE tag_id = %d" % (self.params.experiment_tag, tag_id)
    self.execute_query(query, commit=True)

    self.delete_x(tag, tag_id)

  # Replaced by JobFactory in job.py
  #def create_job(self, **kwargs):
  #  return Job(self, **kwargs)

  def get_job(self, job_id):
    return JobFactory.from_args(self, job_id)

  def get_all_jobs(self, active = False, where = None):
    if active:
      if where is None:
        where = "LEFT OUTER JOIN `%s_dataset_task` dataset_task ON job.dataset_id = dataset_task.dataset_id " % self.params.experiment_tag
        where += "WHERE (trial.active = True AND rungroup.active = True) OR (dataset.active = True AND job.task_id = dataset_task.task_id AND dataset_task.task_id IS NOT NULL)"
    return self.get_all_x_with_subitems(JobFactory.from_args, "job", sub_items = [(Trial, 'trial', False),
                                                                                  (Run, 'run', False),
                                                                                  (Rungroup, 'rungroup', False),
                                                                                  (Task, 'task', False),
                                                                                  (Dataset, 'dataset', False)], where = where)

  def delete_job(self, job = None, job_id = None):
    assert [job, job_id].count(None) == 1
    if job_id is None:
      job_id = job.job_id
    else:
      job = self.get_job(job_id)

    self.delete_x(job, job_id)

  def get_all_events(self, trial = None, runs = None, only_indexed = True, isoform = None, where = None):
    tag = self.params.experiment_tag
    if where is None:
      final_where = ""
    else:
      final_where = where.strip()
    where = ""
    if only_indexed or isoform is not None:
      where = " JOIN `%s_imageset_event` is_e ON event.id = is_e.event_id"%tag
    if trial is not None:
      if runs is None:
        runs = trial.runs
      if len(runs) == 0:
        return []
      if isoform is None:
        where += " WHERE"
      else:
        where += """ JOIN `%s_imageset` imgset ON imgset.id = is_e.imageset_id
                     JOIN `%s_experiment` exp ON exp.imageset_id = imgset.id
                     JOIN `%s_crystal` crystal ON crystal.id = exp.crystal_id
                     JOIN `%s_cell` cell ON cell.id = crystal.cell_id
                     JOIN `%s_isoform` isoform ON isoform.id = cell.isoform_id
                     WHERE isoform.id = %s AND"""%(tag, tag, tag, tag, tag, isoform.id)
      where += " event.trial_id = %d AND event.run_id in (%s)" % (
        trial.id, ", ".join([str(r.id) for r in runs]))

      if 'rungroup_id' in self.columns_dict["%s_%s" % (tag, 'event')]: # some backwards compatibility, as event.rungroup_id was added late to the schema
        rungroups = ", ".join([str(rg.id) for rg in trial.rungroups])
        if len(rungroups) > 0:
          where += " AND event.rungroup_id in (%s)"%rungroups

    if len(where) > 0 and len(final_where) > 0:
      final_where = "AND " + final_where.lstrip("WHERE")
    return self.get_all_x(Event, "event", where + " " + final_where)

  def get_stats(self, **kwargs):
    return Stats(self, **kwargs)

  def create_dataset(self, **kwargs):
    return Dataset(self, **kwargs)

  def get_dataset(self, dataset_id):
    return Dataset(self, dataset_id)

  def get_all_datasets(self):
    return self.get_all_x(Dataset, "dataset")

  def get_dataset_tasks(self, dataset_id):
    tag = self.params.experiment_tag
    query = """SELECT d_t.task_id FROM `%s_dataset_task` d_t
               WHERE d_t.dataset_id = %d""" % (tag, dataset_id)
    cursor = self.execute_query(query)
    task_ids = ["%d"%i[0] for i in cursor.fetchall()]
    if len(task_ids) == 0:
      return []
    return self.get_all_x(Task, "task", where = "WHERE task.id IN (%s)"%",".join(task_ids))

  def create_dataset_version(self, **kwargs):
    return DatasetVersion(self, **kwargs)

  def get_dataset_version(self, dataset_version_id):
    return DatasetVersion(self, dataset_version_id)

  def get_dataset_versions(self, dataset_id, latest = False):
    tag = self.params.experiment_tag
    if latest:
      query = """SELECT MAX(dv.id) FROM `%s_dataset_version` dv
                 WHERE dv.dataset_id = %d"""%(tag, dataset_id)
      cursor = self.execute_query(query)
      rows = cursor.fetchall()
      if not rows: return []
      if rows[0][0] is None: return []
      dataset_version_ids = [i[0] for i in rows]
      if not dataset_version_ids : return []
      assert len(dataset_version_ids) == 1
      where = "WHERE dataset_version.id = %d"%dataset_version_ids[0]
    else:
      where = "WHERE dataset_version.dataset_id = %d"%dataset_id
    return self.get_all_x_with_subitems(DatasetVersion, "dataset_version", where = where, sub_items=[(Dataset, "dataset", True)])

  def get_job_dataset_version(self, job_id):
    tag = self.params.experiment_tag
    query = """SELECT dvj.dataset_version_id FROM `%s_dataset_version_job` dvj
               WHERE dvj.job_id = %d""" % (tag, job_id)
    cursor = self.execute_query(query)
    dataset_version_ids = [i[0] for i in cursor.fetchall()]
    if not dataset_version_ids:
      return None
    assert len(dataset_version_ids) == 1
    return self.get_dataset_version(dataset_version_ids[0])

  def get_dataset_version_jobs(self, dataset_version_id):
    tag = self.params.experiment_tag
    query = """SELECT dvj.job_id FROM `%s_dataset_version_job` dvj
               WHERE dvj.dataset_version_id = %d""" % (tag, dataset_version_id)
    cursor = self.execute_query(query)
    job_ids = ["%d"%i[0] for i in cursor.fetchall()]
    if len(job_ids) == 0:
      return []
    return self.get_all_jobs(where = "WHERE job.id IN (%s)"%",".join(job_ids))

  def get_dataset_tags(self, dataset_id):
    tag = self.params.experiment_tag
    query = """SELECT d_t.tag_id FROM `%s_dataset_tag` d_t
               WHERE d_t.dataset_id = %d""" % (tag, dataset_id)
    cursor = self.execute_query(query)
    tag_ids = ["%d"%i[0] for i in cursor.fetchall()]
    if len(tag_ids) == 0:
      return []
    return self.get_all_x(Tag, "tag", where = "WHERE tag.id IN (%s)"%",".join(tag_ids))

  def create_task(self, **kwargs):
    return Task(self, **kwargs)

  def get_task(self, task_id):
    return Task(self, task_id)

  def get_all_tasks(self):
    return self.get_all_x(Task, "task")

# Deprecated, but preserved here in case it proves useful later
"""
class sacla_run_finder(object):
  def __init__(self, params):
    self.params = params
    self.known_runs = []

  def list_runs(self):
    import dbpy
    assert self.params.facility.sacla.start_run is not None
    runs = []
    run = self.params.facility.sacla.start_run
    while True:
      if run in self.known_runs:
        run += 1
        continue
      try:
        info = dbpy.read_runinfo(self.params.facility.sacla.beamline, run)
      except dbpy.APIError:
        break
      if info['runstatus'] == 0:
        runs.append(run)
        self.known_runs.append(run)
      run += 1

    return self.known_runs
"""

class standalone_run_finder(object):
  def __init__(self, params):
    self.params = params

  def list_runs(self):
    import glob

    runs = []
    if self.params.facility.standalone.monitor_for == 'folders':
      for foldername in sorted(os.listdir(self.params.facility.standalone.data_dir)):
        path = os.path.join(self.params.facility.standalone.data_dir, foldername)
        if not os.path.isdir(path): continue
        if self.params.facility.standalone.composite_files:
          for filepath in sorted(glob.glob(os.path.join(path, self.params.facility.standalone.template))):
            filename = os.path.basename(filepath)
            runs.append((foldername + "_" + os.path.splitext(filename)[0], filepath))
        else:
          files = sorted(glob.glob(os.path.join(path, self.params.facility.standalone.template)))
          if len(files) >= self.params.facility.standalone.folders.n_files_needed:
            runs.append((foldername, os.path.join(path, self.params.facility.standalone.template)))
    elif self.params.facility.standalone.monitor_for == 'files':
      if not self.params.facility.standalone.composite_files:
        print("Warning, monitoring a single folder for single image files is inefficient")
      path = self.params.facility.standalone.data_dir
      for filepath in sorted(glob.glob(os.path.join(path, self.params.facility.standalone.template))):
        if self.params.facility.standalone.files.last_modified > 0:
          if time.time() - os.path.getmtime(filepath) < self.params.facility.standalone.files.last_modified:
            continue
        if self.params.facility.standalone.files.minimum_file_size > 0:
          statinfo = os.stat(filepath)
          if statinfo.st_size < self.params.facility.standalone.files.minimum_file_size:
            continue
        filename = os.path.basename(filepath)
        runs.append((os.path.splitext(filename)[0], filepath))

    return runs

class cheetah_run_finder(standalone_run_finder):
  def __init__(self, params):
    super(cheetah_run_finder, self).__init__(params)
    self.known_runs = []
    self.known_bad_runs = []

  def list_runs(self):
    runs = super(cheetah_run_finder, self).list_runs()

    good_runs = []
    for name, path in runs:
      if name in self.known_runs:
        good_runs.append((name, path))
      elif name not in self.known_bad_runs:
        status_file = os.path.join(os.path.dirname(path), 'status.txt')
        if not os.path.exists(status_file): continue
        status_lines = [line for line in open(status_file).readlines() if 'Status' in line]
        if len(status_lines) != 1: continue
        l = status_lines[0].strip()
        status = l.split(',')[-1].split('=')[-1]
        print(name, status)
        if status == 'Finished':
          good_runs.append((name, path))
          self.known_runs.append(name)
        elif 'error' in status.lower():
          self.known_bad_runs.append(name)

    return good_runs
