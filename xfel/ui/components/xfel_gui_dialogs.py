from __future__ import absolute_import, division, print_function
from six.moves import range
import six

'''
Author      : Lyubimov, A.Y.
Created     : 06/04/2016
Last Changed: 06/04/2016
Description : XFEL UI Custom Dialogs
'''

import time
import os
import wx
from wx.lib.mixins.listctrl import TextEditMixin, getListCtrlSelection
from wx.lib.scrolledpanel import ScrolledPanel
from xfel.ui.db.task import task_types
import numpy as np

import xfel.ui.components.xfel_gui_controls as gctr
from xfel.ui.components.submission_tracker import QueueInterrogator

icons = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons/')

# Platform-specific stuff
# TODO: Will need to test this on Windows at some point
if wx.Platform == '__WXGTK__':
  norm_font_size = 10
  button_font_size = 12
  LABEL_SIZE = 14
  CAPTION_SIZE = 12
elif wx.Platform == '__WXMAC__':
  norm_font_size = 12
  button_font_size = 14
  LABEL_SIZE = 14
  CAPTION_SIZE = 12
elif (wx.Platform == '__WXMSW__'):
  norm_font_size = 9
  button_font_size = 11
  LABEL_SIZE = 11
  CAPTION_SIZE = 9


class EdListCtrl(wx.ListCtrl, TextEditMixin):
  ''' TextEditMixin allows any column to be edited. '''

  # ----------------------------------------------------------------------
  def __init__(self, parent, ID=wx.ID_ANY, pos=wx.DefaultPosition,
               size=wx.DefaultSize, style=0):
    """Constructor"""
    wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
    TextEditMixin.__init__(self)
    self.curRow = -1

  def OnLeftDown(self, evt=None):
    try:
      return super(EdListCtrl, self).OnLeftDown(evt)
    except wx._core.wxAssertionError:
      pass

class BaseDialog(wx.Dialog):
  def __init__(self, parent,
               label_style='bold',
               content_style='normal',
               *args, **kwargs):
    wx.Dialog.__init__(self, parent, *args, **kwargs)

    self.envelope = wx.BoxSizer(wx.VERTICAL)
    self.main_sizer = wx.BoxSizer(wx.VERTICAL)
    self.envelope.Add(self.main_sizer, flag=wx.EXPAND | wx.ALL, border=5)
    self.SetSizer(self.envelope)

    if label_style == 'normal':
      self.font = wx.Font(norm_font_size, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
    elif label_style == 'bold':
      self.font = wx.Font(norm_font_size, wx.DEFAULT, wx.NORMAL, wx.BOLD)
    elif label_style == 'italic':
      self.font = wx.Font(norm_font_size, wx.DEFAULT, wx.ITALIC, wx.NORMAL)
    elif label_style == 'italic_bold':
      self.font = wx.Font(norm_font_size, wx.DEFAULT, wx.ITALIC, wx.BOLD)

    if content_style == 'normal':
      self.cfont = wx.Font(norm_font_size, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
    elif content_style == 'bold':
      self.cfont = wx.Font(norm_font_size, wx.DEFAULT, wx.NORMAL, wx.BOLD)
    elif content_style == 'italic':
      self.cfont = wx.Font(norm_font_size, wx.DEFAULT, wx.ITALIC, wx.NORMAL)
    elif content_style == 'italic_bold':
      self.cfont = wx.Font(norm_font_size, wx.DEFAULT, wx.ITALIC, wx.BOLD)


# --------------------------------- Dialogs ---------------------------------- #

class SettingsDialog(BaseDialog):
  ''' Initial settings for cctbx.xfel; accepts DB credentials (may be
  separate dialog), populates experiment name / tag, separate dialog for
  multiprocessing and other settings; starts sentinels on OK '''

  def __init__(self, parent, params,
               label_style='bold',
               content_style='normal',
               *args, **kwargs):

    BaseDialog.__init__(self, parent,
                        label_style=label_style,
                        content_style=content_style,
                        size=(600, -1),
                        *args, **kwargs)

    self.params = params
    self.drop_tables = False

    self.main_sizer = wx.BoxSizer(wx.VERTICAL)

    # Experiment tag and DB Credentials button
    self.db_cred = gctr.TextButtonCtrl(self,
                                       name='db_cred',
                                       label='Experiment Tag',
                                       label_style='bold',
                                       label_size=(150, -1),
                                       big_button=True,
                                       big_button_label='DB Credentials...',
                                       big_button_size=(130, -1),
                                       value=self.params.experiment_tag if self.params.experiment_tag is not None else "")
    self.main_sizer.Add(self.db_cred,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    # Facility control
    self.facility_sizer = wx.BoxSizer(wx.HORIZONTAL)

    choices = ['LCLS', 'Standalone']
    lower_choices = [f.lower() for f in choices]
    self.facility = gctr.ChoiceCtrl(self,
                                    name = 'facility',
                                    label='Facility',
                                    label_size=(150, -1),
                                    ctrl_size=(180, -1),
                                    label_style='bold',
                                    choices=choices)
    self.facility_sizer.Add(self.facility, flag=wx.EXPAND | wx.ALL, border=10)
    try:
      self.facility.ctr.SetSelection(lower_choices.index(params.facility.name))
    except ValueError:
      pass

    self.btn_facility_options = gctr.Button(self, name = 'btn_facility_options', label='Options...')
    self.facility_sizer.Add(self.btn_facility_options, flag=wx.EXPAND | wx.ALL, border=10)

    self.main_sizer.Add(self.facility_sizer, flag=wx.EXPAND | wx.ALL)

    self.lcls_api_sizer = wx.BoxSizer(wx.HORIZONTAL)

    lcls_api_host = None
    if self.params.facility.name == 'lcls':
      lcls_api_host = self.params.facility.lcls.api.host
    if lcls_api_host is None: lcls_api_host = ''

    self.lcls_api_host = gctr.TextButtonCtrl(self,
                                             name='lcls_api_host',
                                             label='LCLS API Host',
                                             label_size=(150, -1),
                                             label_style='bold',
                                             big_button=False,
                                             value=lcls_api_host)
    self.lcls_api_sizer.Add(self.lcls_api_host, flag=wx.EXPAND | wx.ALL, border=10)

    choices = ["http", "https"]
    self.lcls_api_protocol = gctr.ChoiceCtrl(self,
                                             name='lcls_api_protocol',
                                             label='Protocol',
                                             label_size=(50, -1),
                                             ctrl_size=(80, -1),
                                             label_style='bold',
                                             choices=choices)
    try:
      self.lcls_api_protocol.ctr.SetSelection(choices.index(params.facility.lcls.api.protocol))
    except ValueError:
      pass

    self.lcls_api_sizer.Add(self.lcls_api_protocol, flag=wx.EXPAND | wx.ALL, border=10)

    self.main_sizer.Add(self.lcls_api_sizer, flag=wx.EXPAND | wx.ALL)

    # Experiment name control
    experiment = None
    if self.params.facility.name == 'lcls': experiment = self.params.facility.lcls.experiment
    if experiment is None: experiment = ''
    self.experiment = gctr.TextButtonCtrl(self,
                                          name='experiment',
                                          label='Experiment',
                                          label_style='bold',
                                          label_size=(150, -1),
                                          big_button_size=(130, -1),
                                          value=experiment)
    self.main_sizer.Add(self.experiment,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    # Output folder text control w/ Browse / magnifying glass button
    if self.params.output_folder == '':
      current_folder = os.path.abspath(os.curdir)
    else:
      current_folder = self.params.output_folder
    self.output = gctr.TextButtonCtrl(self,
                                      name='output',
                                      label='Output',
                                      label_style='bold',
                                      label_size=(150, -1),
                                      big_button=True,
                                      big_button_label='Browse...',
                                      big_button_size=(120, -1),
                                      value=current_folder)
    self.main_sizer.Add(self.output,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    self.btn_op = gctr.Button(self, name='advanced', label='Advanced Settings...')
    self.btn_OK = wx.Button(self, label="OK", id=wx.ID_OK)
    self.btn_OK.SetDefault()
    self.btn_cancel = wx.Button(self, label="Cancel", id=wx.ID_CANCEL)

    button_sizer = wx.FlexGridSizer(1, 4, 0, 10)
    button_sizer.AddMany([#(self.btn_mp),
                          (self.btn_op),
                          (0,0),
                          (self.btn_OK),
                          (self.btn_cancel)])

    button_sizer.AddGrowableCol(1)
    self.main_sizer.Add(button_sizer,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)
    self.SetSizerAndFit(self.main_sizer)

    self.Bind(wx.EVT_BUTTON, self.onDBCredentialsButton, id=self.db_cred.btn_big.GetId())
    self.Bind(wx.EVT_BUTTON, self.onAdvanced, id=self.btn_op.GetId())
    self.Bind(wx.EVT_BUTTON, self.onOK, id=self.btn_OK.GetId())
    self.Bind(wx.EVT_BUTTON, self.onBrowse, id=self.output.btn_big.GetId())
    self.Bind(wx.EVT_CHOICE, self.onFacilityChoice)
    self.Bind(wx.EVT_BUTTON, self.onFacilityOptions, id=self.btn_facility_options.GetId())

    self.setup_facility_options()

  def onBrowse(self, e):
    dlg = wx.DirDialog(self, "Choose the input directory:",
                       style=wx.DD_DEFAULT_STYLE)

    if dlg.ShowModal() == wx.ID_OK:
      self.output.ctr.SetValue(dlg.GetPath())
    dlg.Destroy()

  def onFacilityChoice(self, e):
    self.params.facility.name = self.facility.ctr.GetStringSelection().lower()
    self.setup_facility_options()

  def setup_facility_options(self):
    if self.params.facility.name in ['lcls']:
      self.experiment.Enable()
      self.lcls_api_host.Enable()
      self.lcls_api_protocol.Enable()
    else:
      self.experiment.Disable()
      self.lcls_api_host.Disable()
      self.lcls_api_protocol.Disable()

  def onFacilityOptions(self, e):
    if self.params.facility.name == 'lcls':
      opts = LCLSFacilityOptions(self, self.params)
      opts.Fit()
      opts.Center()
      opts.ShowModal()
    elif self.params.facility.name == 'standalone':
      opts = StandaloneOptions(self, self.params)
      opts.Fit()
      opts.Center()
      opts.ShowModal()

  def onAdvanced(self, e):
    adv = AdvancedSettingsDialog(self, self.params)
    adv.Fit()
    adv.Center()

    adv.ShowModal()

  def onDBCredentialsButton(self, e):
    self.update_settings()
    creds = DBCredentialsDialog(self, self.params)
    creds.Center()
    if (creds.ShowModal() == wx.ID_OK):
      self.params.db.host     = creds.db_host.ctr.GetValue()
      self.params.db.port     = int(creds.db_port.ctr.GetValue())
      self.params.db.name     = creds.db_name.ctr.GetValue()
      self.params.db.user     = creds.db_user.ctr.GetValue()
      self.params.db.password = creds.db_password.ctr.GetValue()
      if self.params.facility.name == 'lcls':
        self.params.facility.lcls.web.location = creds.web_location.ctr.GetValue()

      self.drop_tables = creds.chk_drop_tables.GetValue()

  def update_settings(self):
    self.params.facility.name = self.facility.ctr.GetStringSelection().lower()
    self.params.experiment_tag = self.db_cred.ctr.GetValue()
    self.params.output_folder = self.output.ctr.GetValue()
    if self.params.facility.name == 'lcls':
      self.params.facility.lcls.experiment = self.experiment.ctr.GetValue()
      self.params.facility.lcls.api.host = self.lcls_api_host.ctr.GetValue()
      self.params.facility.lcls.api.protocol = self.lcls_api_protocol.ctr.GetStringSelection().lower()

  def onOK(self, e):
    self.update_settings()
    e.Skip()


class DBCredentialsDialog(BaseDialog):
  ''' DB credentials entry '''

  def __init__(self, parent, params,
               label_style='bold',
               content_style='normal',
               *args, **kwargs):

    self.params = params
    BaseDialog.__init__(self, parent,
                        label_style=label_style,
                        content_style=content_style,
                        style=wx.NO_BORDER,
                        *args, **kwargs)

    # Host name
    self.db_host = gctr.TextButtonCtrl(self,
                                       name='db_host',
                                       label='DB Host name',
                                       label_style='bold',
                                       label_size=(150, -1),
                                       big_button_size=(130, -1),
                                       value=params.db.host)
    self.main_sizer.Add(self.db_host, flag=wx.EXPAND | wx.ALL, border=10)

    # Host name
    self.db_port = gctr.TextButtonCtrl(self,
                                       name='db_port',
                                       label='DB Port number',
                                       label_style='bold',
                                       label_size=(150, -1),
                                       big_button_size=(130, -1),
                                       value=str(params.db.port))
    self.main_sizer.Add(self.db_port, flag=wx.EXPAND | wx.ALL, border=10)

    # Database name
    self.db_name = gctr.TextButtonCtrl(self,
                                       name='db_name',
                                       label='DB name',
                                       label_style='bold',
                                       label_size=(150, -1),
                                       big_button_size=(130, -1),
                                       value=params.db.name)
    self.main_sizer.Add(self.db_name, flag=wx.EXPAND | wx.ALL, border=10)

    # User name
    self.db_user = gctr.TextButtonCtrl(self,
                                       name='db_user',
                                       label='DB user name',
                                       label_style='bold',
                                       label_size=(150, -1),
                                       big_button_size=(130, -1),
                                       value=params.db.user)
    self.main_sizer.Add(self.db_user, flag=wx.EXPAND | wx.ALL, border=10)

    # Password
    self.db_password = gctr.TextButtonCtrl(self,
                                           name='db_password',
                                           label='DB Password',
                                           label_style='bold',
                                           label_size=(150, -1),
                                           text_style=wx.TE_PASSWORD,
                                           big_button_size=(130, -1),
                                           value=params.db.password)
    self.main_sizer.Add(self.db_password, flag=wx.EXPAND | wx.ALL, border=10)



    # Drop tables button
    self.chk_drop_tables = wx.CheckBox(self,
                                       label='Delete and regenerate all tables')
    self.main_sizer.Add(self.chk_drop_tables, flag=wx.ALL, border=10)

    if params.facility.name == 'lcls':
      self.main_sizer.Add(wx.StaticLine(self), flag=wx.EXPAND | wx.ALL, border=10)
      # LCLS user name
      self.web_location = gctr.TextButtonCtrl(self,
                                              name='web_location',
                                              label='XTC stream location\n(SLAC or NERSC)',
                                              label_style='bold',
                                              label_size=(150, -1),
                                              big_button_size=(130, -1),
                                              value=params.facility.lcls.web.location)
      self.main_sizer.Add(self.web_location, flag=wx.EXPAND | wx.ALL, border=10)

    # Dialog control

    self.btn_db_start = gctr.Button(self, name='start_db', label='Start DB Server')
    self.db_btn_OK = wx.Button(self, name='db_OK', label="OK", id=wx.ID_OK)
    self.db_btn_OK.SetDefault()
    self.db_btn_cancel = wx.Button(self, label="Cancel", id=wx.ID_CANCEL)
    self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)
    self.button_sizer.Add(self.btn_db_start)
    self.button_sizer.Add(-1,-1, proportion=1)
    self.button_sizer.Add(self.db_btn_OK)
    self.button_sizer.Add(self.db_btn_cancel)
    self.main_sizer.Add(self.button_sizer,flag=wx.EXPAND | wx.ALL, border=10)


    self.Bind(wx.EVT_CHECKBOX, self.onDropTables, self.chk_drop_tables)
    self.Bind(wx.EVT_BUTTON, self.onStartDB, self.btn_db_start)

    self.Fit()
    self.SetTitle('Database Credentials')

  def onDropTables(self, e):
    if self.chk_drop_tables.GetValue():
      msg = wx.MessageDialog(self,
                             message='Are you sure?',
                             caption='Warning',
                             style=wx.YES_NO |  wx.ICON_EXCLAMATION)

      if (msg.ShowModal() == wx.ID_NO):
        self.chk_drop_tables.SetValue(False)
    e.Skip()

  def onStartDB(self, e):
    self.start_db_dialog = StartDBDialog(self, self.params)
    if (self.start_db_dialog.ShowModal() == wx.ID_OK):
      self.params.db.server.root_password = self.start_db_dialog.get_db_root_psswd.ctr.GetValue()
      self.params.db.server.basedir = self.start_db_dialog.get_db_basedir.ctr.GetValue()

      def _submit_start_server_job(params):
        from xfel.command_line.cxi_mpi_submit import do_submit
        assert self.params.db.user is not None, "DB User not defined!"
        assert self.params.db.password is not None, "Password for DB User not defined!"
        assert self.params.db.server.root_password is not None, "Root password for DB not defined!"
        assert self.params.db.server.basedir is not None, "Base directory for DB not defined!"

        try:
          import copy
          new_params = copy.deepcopy(params)
          new_params.mp.use_mpi = False
          new_params.mp.extra_args = ["db.port=%d db.server.basedir=%s db.user=%s db.name=%s db.server.root_password=%s" %(params.db.port, params.db.server.basedir, params.db.user, params.db.name, params.db.server.root_password)]
          submit_path = os.path.join(params.output_folder, "launch_server_submit.sh")
          submission_id = do_submit('cctbx.xfel.ui_server',
                                    submit_path,
                                    new_params.output_folder,
                                    new_params.mp,
                                    log_name="my_SQL.log",
                                    err_name="my_SQL.err",
                                    job_name='cctbx_start_mysql'
                                   )
          #remove root password from params
          if submission_id:
            if (self.params.mp.method == 'slurm') or (self.params.mp.method == 'shifter') or (self.params.mp.method == 'sfapi'):
              attempts = 10
              q = QueueInterrogator(self.params.mp.method)
              for i in range(attempts):
                status = q.query(submission_id)
                if status == 'RUN':
                  hostname = q.get_mysql_server_hostname(submission_id)
                  if hostname:
                    self.params.db.host = hostname
                  else:
                    print("Unable to find hostname running MySQL server from SLURM. Submission ID: ", submission_id)
                else:
                  print("Waiting for job to start. Submission ID: %s, status: %s"%(submission_id, status))
                  time.sleep(1)
            elif self.params.mp.method == 'local':
              self.params.db.host = params.db.host
            else:
              print("Unable to find hostname running MySQL server on ", self.params.mp.method)
              print("Submission ID: ", submission_id)

            self.params.db.port = int(params.db.port)
            self.params.db.name = params.db.name
            self.params.db.user = params.db.user
            self.params.db.password = params.db.password
            self.params.db.server.root_password = ''
            self.params.db.server.basedir = params.db.server.basedir

            os.remove(os.path.join(self.params.output_folder, "launch_server_submit.sh"))
            os.remove(os.path.join(self.params.output_folder, "launch_server_submit_submit.sh"))
          else:
            print('couldn\'t submit job')
        except RuntimeError:
          print("Couldn\'t submit job to start MySql DB.")
          print("Check if all phil parameters required to launch jobs exists.")

      _submit_start_server_job(self.params)
      db_file_location = self.params.db.server.basedir
      self.launch_db_sizer = wx.BoxSizer(wx.HORIZONTAL)
      msg_text = "DB will be located in\n" + str(db_file_location)
      self.db_start_box = wx.MessageBox(msg_text,"DB Info", wx.OK | wx.ICON_INFORMATION)
      print("Started DB")
      self.Close()

class StartDBDialog(BaseDialog):
  ''' Dialog to start DB '''

  def __init__(self, parent, params,
               label_style='bold',
               content_style='normal',
               *args, **kwargs):

    self.params = params
    self.start_db_sizer = wx.BoxSizer(wx.HORIZONTAL)
    self.start_db_sizer2 = wx.BoxSizer(wx.HORIZONTAL)
    self.start_db_sizer3 = wx.BoxSizer(wx.HORIZONTAL)
    self.start_db_sizer4 = wx.BoxSizer(wx.HORIZONTAL)
    self.vsiz = wx.BoxSizer(wx.VERTICAL)
    BaseDialog.__init__(self, parent,
                        label_style=label_style,
                        content_style=content_style,
                        style=wx.DEFAULT_FRAME_STYLE,
                        *args, **kwargs)
    warn_icon = wx.ArtProvider.GetBitmap(wx.ART_WARNING, wx.ART_OTHER, (50, 50))
    self.staticbmp = wx.StaticBitmap(self, -1, warn_icon, pos=(1, 1))
    self.start_db_sizer.Add(self.staticbmp, flag=wx.ALL)
    self.vsiz.Add(self.start_db_sizer, 0)
    font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL, False)
    warning_label1 = wx.StaticText(self, wx.ID_STATIC, 'This will start the server! Make sure the server is not already running!')
    warning_label1.SetFont(font)
    self.start_db_sizer.Add(warning_label1, 0, wx.ALL | wx.RIGHT)
    self.vsiz.Add(self.start_db_sizer, 0, wx.ALL, 20)
    self.get_db_basedir = gctr.TextButtonCtrl(self,
                                              name='basedir',
                                              label='DB Base Directory',
                                              label_style='bold',
                                              label_size=(150, -1),
                                              big_button_size=(130, -1),
                                              value=os.path.join(self.params.output_folder, 'MySql')
                                              )
    self.start_db_sizer2.Add(self.get_db_basedir)
    self.vsiz.Add(self.start_db_sizer2, 0, wx.ALL, 40)
    self.get_db_root_psswd = gctr.TextButtonCtrl(self,
                                                 name='db_root_password',
                                                 label='DB Root Password',
                                                 label_style='bold',
                                                 label_size=(150, -1),
                                                 text_style=wx.TE_PASSWORD,
                                                )

    self.start_db_sizer3.Add(self.get_db_root_psswd)
    self.start_db_sizer3.Add(-1, -1, proportion=1)
    self.start_db_cancel_btn = wx.Button(self, label="Cancel", id=wx.ID_CANCEL)
    self.start_db_OK_btn = wx.Button(self, label="OK", id=wx.ID_OK)
    self.start_db_sizer3.Add(self.start_db_cancel_btn)
    self.start_db_sizer3.Add(self.start_db_OK_btn)
    self.vsiz.Add(self.start_db_sizer3, 0, wx.ALL, 60)
    self.main_sizer.Add(self.start_db_sizer, flag=wx.EXPAND | wx.ALL, border=10)
    self.main_sizer.Add(self.start_db_sizer2, flag=wx.EXPAND | wx.ALL, border=10)
    self.main_sizer.Add(self.start_db_sizer3, flag=wx.EXPAND | wx.ALL, border=10)

    self.Fit()
    self.Center()
    self.SetTitle('Start Database')

class LCLSFacilityOptions(BaseDialog):
  ''' Options settings specific to LCLS'''
  def __init__(self, parent, params,
               label_style='bold',
               content_style='normal',
               *args, **kwargs):

    self.params = params
    BaseDialog.__init__(self, parent, label_style=label_style,
                        content_style=content_style, *args, **kwargs)

    self.main_sizer = wx.BoxSizer(wx.VERTICAL)

    self.chk_use_ffb = wx.CheckBox(self,
                                   label='Use ffb (fast feedback) file system. Active experiment only, on hiprio or prio queues')
    self.chk_use_ffb.SetValue(params.facility.lcls.use_ffb)

    self.chk_dump_shots = wx.CheckBox(self,
                                      label='Dump all images to disk. Useful for tuning spotfinding and indexing parameters')
    self.chk_dump_shots.SetValue(params.facility.lcls.dump_shots)

    self.chk_enforce80 = wx.CheckBox(self,
                                     label='Require stream 80 (FEE spectrometer) before processing')
    self.chk_enforce80.SetValue(params.facility.lcls.web.enforce80)


    self.chk_enforce81 = wx.CheckBox(self,
                                     label='Require stream 81 (FEE spectrometer) before processing')
    self.chk_enforce81.SetValue(params.facility.lcls.web.enforce81)

    self.main_sizer.Add(self.chk_use_ffb, flag=wx.ALL, border=10)
    self.main_sizer.Add(self.chk_dump_shots, flag=wx.ALL, border=10)
    self.main_sizer.Add(self.chk_enforce80, flag=wx.ALL, border=10)
    self.main_sizer.Add(self.chk_enforce81, flag=wx.ALL, border=10)
    self.SetSizer(self.main_sizer)

    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    self.SetTitle('LCLS Settings')

    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)

  def onOK(self, e):
    self.params.facility.lcls.use_ffb = bool(self.chk_use_ffb.GetValue())
    self.params.facility.lcls.dump_shots = bool(self.chk_dump_shots.GetValue())
    self.params.facility.lcls.web.enforce80 = bool(self.chk_enforce80.GetValue())
    self.params.facility.lcls.web.enforce81 = bool(self.chk_enforce81.GetValue())
    e.Skip()

class StandaloneOptions(BaseDialog):
  ''' Options settings specific to standalone GUI '''
  def __init__(self, parent, params,
               label_style='bold',
               content_style='normal',
               *args, **kwargs):

    self.params = params
    BaseDialog.__init__(self, parent, label_style=label_style,
                        content_style=content_style, *args, **kwargs)

    self.main_sizer = wx.BoxSizer(wx.VERTICAL)

    # Output folder text control w/ Browse / magnifying glass button
    if self.params.facility.standalone.data_dir is None:
      current_folder = os.path.abspath(os.curdir)
    else:
      current_folder = self.params.facility.standalone.data_dir
    self.data_dir = gctr.TextButtonCtrl(self,
                                        name='data_dir',
                                        label='Folder to monitor',
                                        label_style='bold',
                                        label_size=(300, -1),
                                        big_button=True,
                                        big_button_label='Browse...',
                                        big_button_size=(120, -1),
                                        value=current_folder)
    self.main_sizer.Add(self.data_dir,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    self.SetSizer(self.main_sizer)

    # Raw image options
    self.monitor_for = gctr.RadioCtrl(self,
                                      name='monitor_for',
                                      label='Monitor for',
                                      label_style='bold',
                                      label_size=(-1, -1),
                                      direction='horizontal',
                                      items={'files':'files',
                                            'folders':'folders'})
    getattr(self.monitor_for, self.params.facility.standalone.monitor_for).SetValue(1)

    self.main_sizer.Add(self.monitor_for, flag=wx.EXPAND | wx.ALL, border=10)

    self.folders_options = gctr.RadioCtrl(self,
                                          name='folders_options',
                                          label='Run complete criteria',
                                          label_style='bold',
                                          label_size=(-1, -1),
                                          direction='horizontal',
                                          items={'status_file':'Status file',
                                            'n_files':'Number of files'})
    getattr(self.folders_options, self.params.facility.standalone.folders.method).SetValue(1)

    self.main_sizer.Add(self.folders_options, flag=wx.EXPAND | wx.ALL, border=10)

    self.n_files_needed = gctr.TextButtonCtrl(self,
                                              name='n_files_needed',
                                              label='Number of files per run',
                                              label_style='normal',
                                              label_size=(-1, -1),
                                              value=str(self.params.facility.standalone.folders.n_files_needed))
    self.main_sizer.Add(self.n_files_needed, flag=wx.EXPAND | wx.ALL, border=10)

    self.last_modified = gctr.TextButtonCtrl(self,
                                             name='last_modified',
                                             label='Minimum time since last modified\n(in seconds)',
                                             label_style='normal',
                                             label_size=(-1, -1),
                                             value=str(self.params.facility.standalone.files.last_modified))
    self.main_sizer.Add(self.last_modified, flag=wx.EXPAND | wx.ALL, border=10)

    self.minimum_file_size = gctr.TextButtonCtrl(self,
                                                 name='minimum_file_size',
                                                 label='Minimum file size\n(in bytes)',
                                                 label_style='normal',
                                                 label_size=(-1, -1),
                                                 value=str(self.params.facility.standalone.files.minimum_file_size))
    self.main_sizer.Add(self.minimum_file_size, flag=wx.EXPAND | wx.ALL, border=10)

    # File matching template control
    if self.params.facility.standalone.template is None:
      self.params.facility.standalone.template = ''
    self.template = gctr.TextButtonCtrl(self,
                                        name='template',
                                        label='File matching template\n(example *.h5)',
                                        label_style='bold',
                                        label_size=(300, -1),
                                        value=self.params.facility.standalone.template)
    self.main_sizer.Add(self.template,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    # Composite check
    self.chk_composite = wx.CheckBox(self,
                                   label='Files are composite (like HDF5, files are submitted as individual runs.\nOtherwise, groups of files are submitted as single runs)')
    self.chk_composite.SetValue(params.facility.standalone.composite_files)
    self.main_sizer.Add(self.chk_composite,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    self.SetTitle('Standalone settings')

    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)
    self.Bind(wx.EVT_BUTTON, self.onBrowse, id=self.data_dir.btn_big.GetId())
    self.Bind(wx.EVT_RADIOBUTTON, self.onOptionsChanged, self.monitor_for.files)
    self.Bind(wx.EVT_RADIOBUTTON, self.onOptionsChanged, self.monitor_for.folders)
    self.Bind(wx.EVT_RADIOBUTTON, self.onOptionsChanged, self.folders_options.status_file)
    self.Bind(wx.EVT_RADIOBUTTON, self.onOptionsChanged, self.folders_options.n_files)
    self.Bind(wx.EVT_CHECKBOX, self.onOptionsChanged, self.chk_composite)
    self.update_options()

  def onOK(self, e):
    self.params.facility.standalone.data_dir = self.data_dir.ctr.GetValue()
    if self.monitor_for.files.GetValue():
      self.params.facility.standalone.monitor_for = 'files'
      self.params.facility.standalone.files.last_modified = float(self.last_modified.ctr.GetValue())
      self.params.facility.standalone.files.minimum_file_size = int(self.minimum_file_size.ctr.GetValue())
    else:
      self.params.facility.standalone.monitor_for = 'folders'
    if self.folders_options.status_file.GetValue():
      self.params.facility.standalone.folders.method = 'status_file'
    elif self.folders_options.n_files.GetValue():
      self.params.facility.standalone.folders.method = 'n_files'
      self.params.facility.standalone.folders.n_files_needed = int(self.n_files_needed.ctr.GetValue())
    self.params.facility.standalone.template = self.template.ctr.GetValue()
    self.params.facility.standalone.composite_files = self.chk_composite.GetValue()
    e.Skip()

  def onBrowse(self, e):
    dlg = wx.DirDialog(self, "Choose the input directory:",
                       style=wx.DD_DEFAULT_STYLE)

    if dlg.ShowModal() == wx.ID_OK:
      self.data_dir.ctr.SetValue(dlg.GetPath())
    dlg.Destroy()

  def onOptionsChanged(self, e):
    self.update_options()

  def update_options(self):
    if self.monitor_for.files.GetValue():
      self.folders_options.Disable()
      self.n_files_needed.Disable()
      self.last_modified.Enable()
      self.minimum_file_size.Enable()
    else:
      self.folders_options.Enable()
      if self.folders_options.status_file.GetValue():
        self.n_files_needed.Disable()
      else:
        self.n_files_needed.Enable()
      self.last_modified.Disable()
      self.minimum_file_size.Disable()

class AdvancedSettingsDialog(BaseDialog):
  ''' Advanced settings for the cctbx.xfel front end '''
  def __init__(self, parent, params,
               label_style='bold',
               content_style='normal',
               *args, **kwargs):

    self.params = params
    BaseDialog.__init__(self, parent, label_style=label_style,
                        content_style=content_style, *args, **kwargs)

    # Multiprocessing settings
    mp_box = wx.StaticBox(self, label='Multiprocessing Options')
    self.mp_sizer = wx.StaticBoxSizer(mp_box, wx.VERTICAL)

    choices = [
      'local', 'lsf', 'slurm', 'shifter', 'sfapi', 'sge', 'pbs', 'htcondor',
      'custom'
    ]
    self.mp_option = gctr.ChoiceCtrl(self,
                                     name='mp_option',
                                     label='Multiprocessing:',
                                     label_size=(200, -1),
                                     label_style='bold',
                                     choices=choices)
    self.mp_sizer.Add(self.mp_option, flag=wx.EXPAND | wx.ALL, border=10)
    try:
      self.mp_option.ctr.SetSelection(choices.index(params.mp.method))
    except ValueError:
      pass
    self.Bind(wx.EVT_CHOICE, self.onMultiprocessingChoice, self.mp_option.ctr)

    # Queue
    queues = ['psanaq', 'psanaq', 'psdebugq','psanaidleq', 'psnehhiprioq',
              'psnehprioq', 'psnehq', 'psfehhiprioq', 'psfehprioq', 'psfehq']
    self.queue_choice = gctr.ChoiceCtrl(self,
                                        name='queue',
                                        label='Queue:',
                                        label_size=(200, -1),
                                        label_style='bold',
                                        choices=queues)
    self.Bind(wx.EVT_CHOICE, self.onQueueChoice, self.queue_choice.ctr)
    self.mp_sizer.Add(self.queue_choice, flag=wx.EXPAND | wx.ALL, border=10)
    try:
      self.queue_choice.ctr.SetSelection(queues.index(params.mp.queue))
    except ValueError:
      pass


    self.queue_text = gctr.TextButtonCtrl(self,
                                          name='queue',
                                          label='Queue:',
                                          label_style='bold',
                                          label_size=(200, -1),
                                          value=self.params.mp.queue \
                                                if params.mp.queue is not None else '')
    self.mp_sizer.Add(self.queue_text, flag=wx.EXPAND | wx.ALL, border=10)

    self.nproc = gctr.SpinCtrl(self,
                               name='nproc',
                               label='Total number of processors:',
                               label_size=(240, -1),
                               label_style='normal',
                               ctrl_size=(100, -1),
                               ctrl_value='%d'%params.mp.nproc,
                               ctrl_min=1,
                               ctrl_max=1000)
    self.mp_sizer.Add(self.nproc, flag=wx.EXPAND | wx.ALL, border=10)

    self.nnodes = gctr.SpinCtrl(self,
                                name='nnodes',
                                label='Total number of nodes:',
                                label_size=(240, -1),
                                label_style='normal',
                                ctrl_size=(100, -1),
                                ctrl_value='%d'%params.mp.nnodes,
                                ctrl_min=1,
                                ctrl_max=1000)
    self.mp_sizer.Add(self.nnodes, flag=wx.EXPAND | wx.ALL, border=10)

    self.nppn_box = gctr.CtrlBase(self)
    nppn_txt = wx.StaticText(self.nppn_box, label="Number of processors per node:")
    self.chk_auto_nproc_per_node = wx.CheckBox(self.nppn_box, label='Auto')
    self.chk_auto_nproc_per_node.SetValue(params.mp.nproc_per_node is None)
    self.nproc_per_node = gctr.SpinCtrl(self.nppn_box,
                                        name='nproc_per_node',
                                        ctrl_value='%d'%params.mp.nproc_per_node if params.mp.nproc_per_node else 1,
                                        ctrl_min = 1, ctrl_max = 1000, label_size=(-1,-1))
    if not params.mp.nproc_per_node: self.nproc_per_node.Disable()

    nppn_sizer = wx.FlexGridSizer(1, 3, 0, 10)
    nppn_sizer.Add(nppn_txt, flag=wx.ALL, border=10)
    nppn_sizer.Add(self.chk_auto_nproc_per_node, flag=wx.ALL, border=10)
    nppn_sizer.Add(self.nproc_per_node, flag=wx.EXPAND | wx.ALL, border=10)
    self.nppn_box.SetSizer(nppn_sizer)
    self.mp_sizer.Add(self.nppn_box, flag=wx.EXPAND | wx.RIGHT | wx.TOP | wx.BOTTOM, border=10)

    self.wall_time = gctr.SpinCtrl(self,
                                   name='wall_time',
                                   label='Max Walltime (mins):',
                                   label_size=(240, -1),
                                   label_style='normal',
                                   ctrl_size=(100, -1),
                                   ctrl_value='%d'%params.mp.wall_time if params.mp.wall_time is not None else 1,
                                   ctrl_min=1,
                                   ctrl_max=2880)
    self.mp_sizer.Add(self.wall_time, flag=wx.EXPAND | wx.ALL, border=10)

    self.mpi_command = gctr.TextButtonCtrl(self,
                                           name='mpi_command',
                                           label='MPI command:',
                                           label_style='bold',
                                           label_size=(200, -1),
                                           value=self.params.mp.mpi_command)
    self.mp_sizer.Add(self.mpi_command, flag=wx.EXPAND | wx.ALL, border=10)

    self.env_script = gctr.TextButtonCtrl(self,
                                          name='env_script',
                                          label='Environment setup script:',
                                          label_style='bold',
                                          label_size=(200, -1),
                                          value=params.mp.env_script[0] \
                                                if len(params.mp.env_script) > 0 and \
                                                params.mp.env_script[0] is not None else '')
    self.mp_sizer.Add(self.env_script, flag=wx.EXPAND | wx.ALL, border=10)

    self.phenix_script = gctr.TextButtonCtrl(self,
                                             name='phenix_script',
                                             label='Phenix setup script:',
                                             label_style='bold',
                                             label_size=(200, -1),
                                             value=params.mp.phenix_script[0] \
                                                   if len(params.mp.phenix_script) > 0 and \
                                                   params.mp.phenix_script[0] is not None else '')
    self.mp_sizer.Add(self.phenix_script, flag=wx.EXPAND | wx.ALL, border=10)

    self.htcondor_executable_path = gctr.TextButtonCtrl(self,
                                                        name='htcondor_executable_path',
                                                        label='MPI executable path (mp2script or openmpiscript):',
                                                        label_style='bold',
                                                        label_size=(200, -1),
                                                        value=params.mp.htcondor.executable_path \
                                                        if params.mp.htcondor.executable_path is not None else '')
    self.mp_sizer.Add(self.htcondor_executable_path, flag=wx.EXPAND | wx.ALL, border=10)

    self.htcondor_filesystemdomain = gctr.TextButtonCtrl(self,
                                                        name='htcondor_filesystemdomain',
                                                        label='Shared filesystem domain:',
                                                        label_style='bold',
                                                        label_size=(200, -1),
                                                        value=params.mp.htcondor.filesystemdomain \
                                                        if params.mp.htcondor.filesystemdomain is not None else '')
    self.mp_sizer.Add(self.htcondor_filesystemdomain, flag=wx.EXPAND | wx.ALL, border=10)

    self.main_sizer.Add(self.mp_sizer, flag=wx.EXPAND | wx.ALL, border=10)

    # Different nnodes per job type. Implemented for shifter and slurm

    self.jobtype_nnodes_box = wx.StaticBox(self, label='Nodes per job')
    self.jobtype_nnodes_sizer = wx.StaticBoxSizer(self.jobtype_nnodes_box, wx.HORIZONTAL)

    self.nnodes_index = gctr.SpinCtrl(self,
                                      name='nnodes_index',
                                      label='Indexing:',
                                      label_size=(60, -1),
                                      label_style='normal',
                                      ctrl_size=(60, -1),
                                      ctrl_value='%d'%(params.mp.nnodes_index or 1),
                                      ctrl_min=1,
                                      ctrl_max=1000)
    self.jobtype_nnodes_sizer.Add(self.nnodes_index, flag=wx.EXPAND | wx.ALL, border=10)

    self.nnodes_tder = gctr.SpinCtrl(self,
                                      name='nnodes_tder',
                                      label='TDER:',
                                      label_size=(60, -1),
                                      label_style='normal',
                                      ctrl_size=(60, -1),
                                      ctrl_value='%d'%(params.mp.nnodes_tder or 1),
                                      ctrl_min=1,
                                      ctrl_max=1000)
    self.nnodes_tder.SetToolTip('Time Dependent Ensemble Refinement')
    self.jobtype_nnodes_sizer.Add(self.nnodes_tder, flag=wx.EXPAND | wx.ALL, border=10)

    self.nnodes_scale = gctr.SpinCtrl(self,
                                      name='nnodes_scale',
                                      label='Scaling:',
                                      label_size=(60, -1),
                                      label_style='normal',
                                      ctrl_size=(60, -1),
                                      ctrl_value='%d'%(params.mp.nnodes_scale or 1),
                                      ctrl_min=1,
                                      ctrl_max=1000)
    self.jobtype_nnodes_sizer.Add(self.nnodes_scale, flag=wx.EXPAND | wx.ALL, border=10)

    self.nnodes_merge = gctr.SpinCtrl(self,
                                      name='nnodes_merge',
                                      label='Merging:',
                                      label_size=(60, -1),
                                      label_style='normal',
                                      ctrl_size=(60, -1),
                                      ctrl_value='%d'%(params.mp.nnodes_merge or 1),
                                      ctrl_min=1,
                                      ctrl_max=1000)
    self.jobtype_nnodes_sizer.Add(self.nnodes_merge, flag=wx.EXPAND | wx.ALL, border=10)

    self.mp_sizer.Add(self.jobtype_nnodes_sizer, flag=wx.EXPAND | wx.ALL, border=10)

    self.extra_box = gctr.CtrlBase(self)
    extra_txt = wx.StaticText(self.extra_box, label="Extra submission arguments")
    self.extra_options = gctr.RichTextCtrl(self.extra_box,
                                           name='extra_options',
                                           size=(-1, 60),
                                           style=wx.VSCROLL,
                                           value="\n".join(self.params.mp.extra_options)
                                                 if any(self.params.mp.extra_options) else "")
    extra_sizer = wx.FlexGridSizer(1, 2, 0, 10)
    extra_sizer.Add(extra_txt, flag=wx.ALL, border=10)
    extra_sizer.Add(self.extra_options, flag=wx.EXPAND | wx.ALL, border=10)
    extra_sizer.AddGrowableCol(1)
    self.extra_box.SetSizer(extra_sizer)
    self.mp_sizer.Add(self.extra_box, flag=wx.EXPAND | wx.ALL, border=10)

    # Shifter-specific settings

    self.shifter_image = gctr.TextButtonCtrl(self,
                                             name='shifter_image',
                                             label='Shifter image:',
                                             label_style='bold',
                                             label_size=(200, -1),
                                             value=params.mp.shifter.shifter_image \
                                             if params.mp.shifter.shifter_image is not None else '')
    self.mp_sizer.Add(self.shifter_image, flag=wx.EXPAND | wx.ALL, border=10)

    self.shifter_srun_template = gctr.TextButtonCtrl(self,
                                                     name='shifter_srun_template',
                                                     label='Srun Script Template Path:',
                                                     label_style='bold',
                                                     label_size=(200, -1),
                                                     value=params.mp.shifter.srun_script_template \
                                                     if params.mp.shifter.srun_script_template is not None else '')
    self.mp_sizer.Add(self.shifter_srun_template, flag=wx.EXPAND | wx.ALL, border=10)

    self.shifter_sbatch_template = gctr.TextButtonCtrl(self,
                                                       name='shifter_sbatch_template',
                                                       label='Sbatch Script Template Path:',
                                                       label_style='bold',
                                                       label_size=(200, -1),
                                                       value=params.mp.shifter.sbatch_script_template \
                                                       if params.mp.shifter.sbatch_script_template is not None else '')
    self.mp_sizer.Add(self.shifter_sbatch_template, flag=wx.EXPAND | wx.ALL, border=10)

    self.shifter_jobname = gctr.TextButtonCtrl(self,
                                               name='shifter_jobname',
                                               label='Job Name:',
                                               label_style='bold',
                                               label_size=(200, -1),
                                               value=params.mp.shifter.jobname \
                                               if params.mp.shifter.jobname is not None else '')
    self.mp_sizer.Add(self.shifter_jobname, flag=wx.EXPAND | wx.ALL, border=10)

    self.shifter_project = gctr.TextButtonCtrl(self,
                                               name='shifter_project',
                                               label='NERSC Project (-A):',
                                               label_style='bold',
                                               label_size=(200, -1),
                                               value=params.mp.shifter.project \
                                               if params.mp.shifter.project is not None else '')
    self.mp_sizer.Add(self.shifter_project, flag=wx.EXPAND | wx.ALL, border=10)

    self.shifter_reservation = gctr.TextButtonCtrl(self,
                                                   name='shifter_reservation',
                                                   label='NERSC Reservation:',
                                                   label_style='bold',
                                                   label_size=(200, -1),
                                                   value=params.mp.shifter.reservation \
                                                   if params.mp.shifter.reservation is not None else '')
    self.mp_sizer.Add(self.shifter_reservation, flag=wx.EXPAND | wx.ALL, border=10)

    self.shifter_constraint = gctr.TextButtonCtrl(self,
                                                  name='shifter_constraint',
                                                  label='Job Constraint:',
                                                  label_style='bold',
                                                  label_size=(200, -1),
                                                  value=params.mp.shifter.constraint \
                                                  if params.mp.shifter.constraint is not None else '')
    self.mp_sizer.Add(self.shifter_constraint, flag=wx.EXPAND | wx.ALL, border=10)

    self.staging_methods = ["DataWarp", "None"]
    self.staging_descriptions = [
        'Stage logs to the DataWarp burst buffer. WARNING: Only when writing to Cori cscratch. Otherwise logs will be lost.',
        'Write logs directly to disk.']
    self.log_staging = gctr.ChoiceCtrl(self,
                                       name='staging',
                                       label="Log staging",
                                       label_size=(240, -1),
                                       label_style='bold',
                                       ctrl_size=(-1, -1),
                                       choices=self.staging_methods)
    self.log_staging.ctr.SetSelection(
        self.staging_methods.index(params.mp.shifter.staging))
    self.Bind(wx.EVT_CHOICE, self.onStagingChoice, self.log_staging.ctr)
    self.mp_sizer.Add(self.log_staging, flag=wx.EXPAND | wx.ALL, border=10)
    self.staging_help = wx.StaticText(self, label=self.staging_descriptions[self.log_staging.ctr.GetSelection()], size=(600,30))
    self.staging_help.Wrap(600)
    self.mp_sizer.Add(self.staging_help, flag=wx.EXPAND | wx.ALL, border=10)

    # SFAPI-specific settings

    # Data analysis settings
    analysis_box = wx.StaticBox(self, label='Data Analysis Options')
    self.analysis_sizer = wx.StaticBoxSizer(analysis_box, wx.VERTICAL)

    # Processing back-ends
    self.dispatchers_sizer = wx.BoxSizer(wx.HORIZONTAL)
    self.back_ends = ['cctbx.xfel', 'Small cell', 'custom']
    self.dispatchers = ['cctbx.xfel.process', 'cctbx.xfel.small_cell_process', 'custom']
    # Legacy dispatchers disabled 12/14/22
    #self.back_ends = ['cctbx.xfel (LCLS mode)', 'cctbx.xfel (standalone mode)', 'Ha14', 'Small cell', 'custom']
    #self.dispatchers = ['cctbx.xfel.xtc_process', 'cctbx.xfel.process', 'cxi.xtc_process', 'cctbx.xfel.small_cell_process', 'custom']
    self.dispatcher_descriptions = [
      #'Process the data according to Brewster 2018, using DIALS for indexing, refinement and integration, with stills-specific defaults. Converts XTC into CBF in memory and optionally provides dumping of CBFs.',
      'Process the data according to Brewster 2018, using DIALS for indexing, refinement and integration, with stills-specific defaults. Reads image files directly.',
      #'Process the data according to Hattne 2014, using LABELIT for initial indexing and stills-specific refinement and integration code implemented in the package cctbx.rstbx.',
      'Process the data according to Brewster 2015, using small cell for initial indexing and using DIALS for refinement and integration, with stills-specific defaults.',
      'Provide a custom program. See authors for details.']

    self.back_end = gctr.ChoiceCtrl(self,
                                    name='back_end',
                                    label='Processing back end:',
                                    label_size=(240, -1),
                                    label_style='bold',
                                    ctrl_size=(-1, -1),
                                    choices=self.back_ends)
    self.Bind(wx.EVT_CHOICE, self.onBackendChoice, self.back_end.ctr)
    self.dispatchers_sizer.Add(self.back_end, flag=wx.ALIGN_LEFT)

    self.custom_dispatcher = gctr.PanelTextCtrl(self,
                                                ctrl_size=(300, -1),
                                                value="")
    self.dispatchers_sizer.Add(self.custom_dispatcher, flag=wx.EXPAND | wx.ALL)

    try:
      self.back_end.ctr.SetSelection(self.dispatchers.index(params.dispatcher))
      self.custom_dispatcher.Hide()
    except ValueError:
      self.back_end.ctr.SetSelection(len(self.dispatchers)-1)
      self.custom_dispatcher.ctr.SetValue(params.dispatcher)

    self.analysis_sizer.Add(self.dispatchers_sizer, flag=wx.EXPAND | wx.ALL, border=10)

    self.dispatcher_help = wx.StaticText(self, label=self.dispatcher_descriptions[self.back_end.ctr.GetSelection()], size=(600,80))
    self.dispatcher_help.Wrap(600)
    self.analysis_sizer.Add(self.dispatcher_help, flag=wx.EXPAND | wx.ALL, border=10)

    self.main_sizer.Add(self.analysis_sizer, flag=wx.EXPAND | wx.ALL, border=10)

    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    self.SetTitle('Advanced Settings')

    self.Bind(wx.EVT_CHECKBOX, self.onChkNppnAuto, self.chk_auto_nproc_per_node)
    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)

    self.updateMultiprocessing()

  def onChkNppnAuto(self, e):
    if self.chk_auto_nproc_per_node.GetValue():
      self.nproc_per_node.Disable()
    else:
      self.nproc_per_node.Enable()

  def onMultiprocessingChoice(self, e):
    self.updateMultiprocessing()

  def updateMultiprocessing(self):
    if self.mp_option.ctr.GetStringSelection() == 'local':
      self.queue_choice.Hide()
      self.queue_text.Show()
      self.nnodes.Hide()
      self.nproc.Show()
      self.nppn_box.Hide()
      self.wall_time.Hide()
      self.mpi_command.Hide()
      self.env_script.Hide()
      self.phenix_script.Hide()
      self.htcondor_executable_path.Hide()
      self.htcondor_filesystemdomain.Hide()
      self.jobtype_nnodes_box.Hide()
      self.nnodes_index.Hide()
      self.nnodes_tder.Hide()
      self.nnodes_scale.Hide()
      self.nnodes_merge.Hide()
      self.extra_box.Hide()
      self.shifter_image.Hide()
      self.shifter_srun_template.Hide()
      self.shifter_sbatch_template.Hide()
      self.shifter_jobname.Hide()
      self.shifter_project.Hide()
      self.shifter_reservation.Hide()
      self.shifter_constraint.Hide()
      self.log_staging.Hide()
      self.staging_help.Hide()
    elif self.mp_option.ctr.GetStringSelection() == 'shifter':
      self.queue_choice.Hide()
      self.queue_text.Show()
      self.nproc.Hide()
      self.nnodes.Show()
      self.nppn_box.Show()
      self.wall_time.Show()
      self.mpi_command.Hide()
      self.env_script.Hide()
      self.phenix_script.Hide()
      self.htcondor_executable_path.Hide()
      self.htcondor_filesystemdomain.Hide()
      self.nnodes_index.Show()
      self.nnodes_tder.Show()
      self.nnodes_scale.Show()
      self.nnodes_merge.Show()
      self.extra_box.Show()
      self.jobtype_nnodes_box.Show()
      self.shifter_image.Show()
      self.shifter_srun_template.Show()
      self.shifter_sbatch_template.Show()
      self.shifter_jobname.Show()
      self.shifter_project.Show()
      self.shifter_reservation.Show()
      self.shifter_constraint.Show()
      self.log_staging.Show()
      self.staging_help.Show()
    elif self.mp_option.ctr.GetStringSelection() == 'htcondor':
      self.queue_choice.Hide()
      self.queue_text.Show()
      self.nproc.Show()
      self.nnodes.Hide()
      self.nppn_box.Hide()
      self.wall_time.Hide()
      self.mpi_command.Hide()
      self.env_script.Show()
      self.phenix_script.Show()
      self.htcondor_executable_path.Show()
      self.htcondor_filesystemdomain.Show()
      self.nnodes_index.Hide()
      self.nnodes_tder.Hide()
      self.nnodes_scale.Hide()
      self.nnodes_merge.Hide()
      self.extra_box.Hide()
      self.jobtype_nnodes_box.Hide()
      self.shifter_image.Hide()
      self.shifter_srun_template.Hide()
      self.shifter_sbatch_template.Hide()
      self.shifter_jobname.Hide()
      self.shifter_project.Hide()
      self.shifter_reservation.Hide()
      self.shifter_constraint.Hide()
      self.log_staging.Hide()
      self.staging_help.Hide()
    elif self.mp_option.ctr.GetStringSelection() in ['slurm', 'pbs', 'sfapi' ]:
      self.queue_choice.Hide()
      self.queue_text.Show()
      self.nproc.Hide()
      self.nnodes.Hide()
      self.nppn_box.Show()
      self.wall_time.Hide()
      self.mpi_command.Show()
      self.env_script.Show()
      self.phenix_script.Show()
      self.htcondor_executable_path.Hide()
      self.htcondor_filesystemdomain.Hide()
      self.nnodes_index.Show()
      self.nnodes_tder.Show()
      self.nnodes_scale.Show()
      self.nnodes_merge.Show()
      self.extra_box.Show()
      self.jobtype_nnodes_box.Show()
      self.shifter_image.Hide()
      self.shifter_srun_template.Hide()
      self.shifter_sbatch_template.Hide()
      self.shifter_jobname.Hide()
      self.shifter_project.Hide()
      self.shifter_reservation.Hide()
      self.shifter_constraint.Hide()
      self.log_staging.Hide()
      self.staging_help.Hide()
    else :
      if self.params.facility.name == 'lcls' and self.mp_option.ctr.GetStringSelection() == 'lsf':
        self.queue_choice.Show()
        self.queue_text.Hide()
      else:
        self.queue_choice.Hide()
        self.queue_text.Show()
      self.nproc.Show()
      self.nnodes.Hide()
      self.nppn_box.Hide()
      self.wall_time.Hide()
      self.mpi_command.Show()
      self.env_script.Show()
      self.phenix_script.Show()
      self.htcondor_executable_path.Hide()
      self.htcondor_filesystemdomain.Hide()
      self.nnodes_index.Hide()
      self.nnodes_tder.Hide()
      self.nnodes_scale.Hide()
      self.nnodes_merge.Hide()
      self.extra_box.Show()
      self.jobtype_nnodes_box.Hide()
      self.shifter_image.Hide()
      self.shifter_srun_template.Hide()
      self.shifter_sbatch_template.Hide()
      self.shifter_jobname.Hide()
      self.shifter_project.Hide()
      self.shifter_reservation.Hide()
      self.shifter_constraint.Hide()
      self.log_staging.Hide()
      self.staging_help.Hide()

    self.Fit()

  def onQueueChoice(self, e):
    queue = self.queue_choice.ctr.GetString(self.queue_choice.ctr.GetSelection())
    if 'neh' in queue or 'feh' in queue:
      self.nproc.ctr.SetValue(16)
      self.nproc.ctr.SetIncrement(16)
    elif 'psana' in queue or 'debug' in queue:
      self.nproc.ctr.SetValue(12)
      self.nproc.ctr.SetIncrement(12)
    else:
      self.nproc.ctr.SetValue(1)
      self.nproc.ctr.SetIncrement(1)

  def onStagingChoice(self, e):
    self.params.mp.shifter.staging = self.staging_methods[self.log_staging.ctr.GetSelection()]
    self.staging_help.SetLabel(self.staging_descriptions[self.log_staging.ctr.GetSelection()])
    self.staging_help.Wrap(600)


  def onBackendChoice(self, e):
    self.params.dispatcher = self.dispatchers[self.back_end.ctr.GetSelection()]
    self.dispatcher_help.SetLabel(self.dispatcher_descriptions[self.back_end.ctr.GetSelection()])
    self.dispatcher_help.Wrap(600)
    if self.params.dispatcher == 'custom':
      self.custom_dispatcher.Show()
      self.Layout()
    else:
      self.custom_dispatcher.Hide()
      self.Layout()


  def onOK(self, e):
    self.params.dispatcher = self.dispatchers[self.back_end.ctr.GetSelection()]

    if self.params.dispatcher == 'custom':
      self.params.dispatcher = self.custom_dispatcher.ctr.GetValue()

    self.params.mp.method = self.mp_option.ctr.GetStringSelection()
    self.params.mp.nproc = int(self.nproc.ctr.GetValue())

    if self.params.facility.name == 'lcls' and self.params.mp.method == "lsf":
      self.params.mp.queue = self.queue_choice.ctr.GetStringSelection()
    else:
      if self.chk_auto_nproc_per_node.GetValue():
        self.params.mp.nproc_per_node = None
      else:
        self.params.mp.nproc_per_node = int(self.nproc_per_node.ctr.GetValue())
      self.params.mp.queue = self.queue_text.ctr.GetValue()
      if self.mp_option.ctr.GetStringSelection() in ['shifter', 'slurm', 'pbs']:
        self.params.mp.nnodes_index = int(self.nnodes_index.ctr.GetValue())
        self.params.mp.nnodes_tder = int(self.nnodes_tder.ctr.GetValue())
        self.params.mp.nnodes_scale = int(self.nnodes_scale.ctr.GetValue())
        self.params.mp.nnodes_merge = int(self.nnodes_merge.ctr.GetValue())
      if self.mp_option.ctr.GetStringSelection() == 'shifter':
        self.params.mp.nnodes = int(self.nnodes.ctr.GetValue())
        self.params.mp.wall_time = int(self.wall_time.ctr.GetValue())
      else:
        self.params.mp.env_script = [self.env_script.ctr.GetValue()]
        self.params.mp.phenix_script = [self.phenix_script.ctr.GetValue()]
        self.params.mp.nproc = int(self.nproc.ctr.GetValue())

    self.params.mp.mpi_command = self.mpi_command.ctr.GetValue() \
      if len(self.mpi_command.ctr.GetValue()) > 0 else None
    if len(self.extra_options.GetValue()) > 0:
      self.params.mp.extra_options = self.extra_options.GetValue().split('\n')
    else:
      self.params.mp.extra_options = []

    # Copy htcondor settings into the htcondor phil
    self.params.mp.htcondor.executable_path = self.htcondor_executable_path.ctr.GetValue() \
      if len(self.htcondor_executable_path.ctr.GetValue()) > 0 else None
    self.params.mp.htcondor.filesystemdomain = self.htcondor_filesystemdomain.ctr.GetValue() \
      if len(self.htcondor_filesystemdomain.ctr.GetValue()) > 0 else None

    # Copy shfiter settings into the shifter phil
    self.params.mp.shifter.sbatch_script_template = self.shifter_sbatch_template.ctr.GetValue() \
      if len(self.shifter_sbatch_template.ctr.GetValue()) > 0 else None
    self.params.mp.shifter.shifter_image = self.shifter_image.ctr.GetValue() \
      if len(self.shifter_image.ctr.GetValue()) > 0 else None
    self.params.mp.shifter.srun_script_template = self.shifter_srun_template.ctr.GetValue() \
      if len(self.shifter_srun_template.ctr.GetValue()) > 0 else None
    self.params.mp.shifter.jobname=self.shifter_jobname.ctr.GetValue() \
      if len(self.shifter_jobname.ctr.GetValue()) > 0 else None
    self.params.mp.shifter.project=self.shifter_project.ctr.GetValue() \
      if len(self.shifter_project.ctr.GetValue()) > 0 else None
    self.params.mp.shifter.reservation=self.shifter_reservation.ctr.GetValue() \
      if len(self.shifter_reservation.ctr.GetValue()) > 0 else None
    self.params.mp.shifter.constraint =self.shifter_constraint.ctr.GetValue() \
      if len(self.shifter_constraint.ctr.GetValue()) > 0 else None

    e.Skip()


class CalibrationDialog(BaseDialog):
  def __init__(self, parent,
               label_style='bold',
               content_style='normal',
               db=None,
               *args, **kwargs):

    self.parent = parent
    BaseDialog.__init__(self, parent, label_style=label_style,
                        content_style=content_style, *args, **kwargs)

    self.db = db

    # Metrology version name
    self.version_name = gctr.TextButtonCtrl(self,
                                            label='Metrology Version Name:',
                                            label_size=(180, -1),
                                            label_style='bold',
                                            value='cspad')
    self.main_sizer.Add(self.version_name, flag=wx.EXPAND | wx.ALL, border=10)

    # Reflection type and number of images in subset
    self.top_sizer = wx.FlexGridSizer(1, 2, 0, 20)
    self.top_sizer.AddGrowableCol(1, 1)
    choices = ['indexed', 'reindexed strong', 'integrated']
    self.reflections = gctr.ChoiceCtrl(self,
                                       label='Reflections:',
                                       label_size=(100, -1),
                                       label_style='bold',
                                       choices=choices)
    self.n_subset = gctr.SpinCtrl(self,
                                  label='Images in subset:',
                                  label_size=(120, -1),
                                  label_style='normal',
                                  ctrl_value='1000',
                                  ctrl_max=10000,
                                  ctrl_min=10)
    self.top_sizer.Add(self.reflections)
    self.top_sizer.Add(self.n_subset, wx.ALIGN_RIGHT)

    #Trial & runs
    self.trial_sizer = wx.FlexGridSizer(1, 2, 0, 20)
    self.trial_sizer.AddGrowableCol(1, 1)
    trials = [str(i.trial) for i in self.db.get_all_trials()]
    self.trial_number = gctr.ChoiceCtrl(self,
                                        label='Trial:',
                                        label_size=(40, -1),
                                        label_style='normal',
                                        ctrl_size=(200, -1),
                                        choices=trials)
    self.trial_runs = gctr.CheckListCtrl(self,
                                         label='Runs:',
                                         label_size=(40, -1),
                                         label_style='normal',
                                         ctrl_size=(200, -1),
                                         choices=[])
    self.trial_sizer.Add(self.trial_number)
    self.trial_sizer.Add(self.trial_runs, wx.ALIGN_RIGHT)

    #Phil blob
    self.phil_text = gctr.RichTextCtrl(self, size=(550, 300), style=wx.VSCROLL)
    self.phil_path = gctr.TwoButtonCtrl(self,
                                        label='PHIL path:',
                                        label_size=(80, -1),
                                        label_style='normal',
                                        button1=True,
                                        button1_label='Browse...',
                                        button2=True,
                                        button2_label='Default PHIL')

    self.main_sizer.Add(self.top_sizer, flag=wx.EXPAND | wx.ALL, border=10)
    self.main_sizer.Add(self.trial_sizer, flag=wx.EXPAND | wx.ALL, border=10)
    self.main_sizer.Add(self.phil_text, flag=wx.EXPAND | wx.ALL, border=10)
    self.main_sizer.Add(self.phil_path, flag=wx.EXPAND | wx.ALL, border=10)

    self.chk_split_dataset = wx.CheckBox(self,
                                         label='Split dataset into 2 halves (outputs statistics, double runtime, uses 2x number of images')
    self.chk_split_dataset.SetValue(True)

    self.main_sizer.Add(self.chk_split_dataset, flag=wx.ALL, border=10)

    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)
    # Bindings
    self.Bind(wx.EVT_CHOICE, self.onTrialChoice,
              id=self.trial_number.ctr.GetId())
    self.Bind(wx.EVT_BUTTON, self.onBrowse, self.phil_path.button1)
    self.Bind(wx.EVT_BUTTON, self.onDefault, self.phil_path.button2)
    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)
    self.Bind(wx.EVT_BUTTON, self.onCancel, id=wx.ID_CANCEL)

    self.SetTitle('Calibration Settings')
    self.find_runs()

    self.frames_sentinel = None
    self.start_frames_sentinel()

  def start_frames_sentinel(self):
    if self.frames_sentinel is None:
      from xfel.ui.components.xfel_gui_init import FramesSentinel
      self.frames_sentinel = FramesSentinel(self)
    self.frames_sentinel.active = True
    self.frames_sentinel.start()

  def stop_frames_sentinel(self):
    if self.frames_sentinel is not None:
      self.frames_sentinel.active = False
      self.frames_sentinel.join()

  def onOK(self, e):
    from xfel.ui.db import get_run_path
    from xfel.util.mp import get_lsf_submit_command
    from xfel.ui import settings_dir
    from libtbx import easy_run
    import copy
    params = copy.deepcopy(self.parent.params)
    params.mp.nproc = 1

    version_str = self.version_name.ctr.GetValue()
    working_dir = os.path.join(params.output_folder, "metrology", version_str)
    if os.path.exists(working_dir):
      wx.MessageBox('Version name %s already used'%version_str, 'Warning!',
                    wx.ICON_EXCLAMATION)
      return

    self.stop_frames_sentinel()

    runs = [self.db.get_run(run_number=int(r)) for r in self.trial_runs.ctr.GetCheckedStrings()]
    run_ids = [r.id for r in runs]
    command = "cspad.cbf_metrology reflections=%s tag=%s split_dataset=%s n_subset=%d "% (
      self.reflections.ctr.GetStringSelection(), self.version_name.ctr.GetValue(),
      self.chk_split_dataset.GetValue(), self.n_subset.ctr.GetValue())

    phil_file = os.path.join(settings_dir, "cfgs", "%s.phil"%version_str)
    f = open(phil_file, 'w')
    f.write(self.phil_text.GetValue())
    f.close()
    command += phil_file + " "

    trial = self.db.get_trial(trial_number=int(self.trial_number.ctr.GetStringSelection()))
    runs_found = []
    run_paths = []
    for rungroup in trial.rungroups:
      for run in rungroup.runs:
        if run.id in runs_found:
          continue
        if run.id in run_ids:
          run_ids.pop(run_ids.index(run.id))
          runs_found.append(run.id)
          run_paths.append(os.path.join(get_run_path(params.output_folder, trial, rungroup, run), 'out'))
    assert len(run_ids) == 0

    command += " ".join(run_paths)

    submit_path = os.path.join(settings_dir, "%s.sh"%version_str)
    command = str(get_lsf_submit_command(command, submit_path, working_dir, params.mp)()) # comes back as unicode which throws off easy_run
    cwd = os.getcwd()
    os.makedirs(working_dir)
    os.chdir(working_dir)

    print("Submitting metrology refinement. Command:")
    print(command)
    try:
      results = easy_run.fully_buffered(command=command)
      results.show_stdout()
      results.show_stderr()
      results.raise_if_errors()
    except Exception as exc:
      if not "Warning: job being submitted without an AFS token." in str(exc):
        raise exc
    os.chdir(cwd)
    print("Output will be in", working_dir)

    e.Skip()

  def onTrialChoice(self, e):
    self.find_runs()

  def find_runs(self):
    self.trial_runs.ctr.Clear()
    trial = self.db.get_all_trials()[self.trial_number.ctr.GetSelection()]
    runs = [str(i.run) for i in trial.runs]
    if runs:
      self.trial_runs.ctr.InsertItems(items=runs, pos=0)

  def onBrowse(self, e):
    ''' Open dialog for selecting PHIL file '''
    load_dlg = wx.FileDialog(self,
                             message="Load PHIL file",
                             defaultDir=os.curdir,
                             defaultFile="*.phil",
                             wildcard="*.phil",
                             style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                             )
    if load_dlg.ShowModal() == wx.ID_OK:
      target_file = load_dlg.GetPaths()[0]
      with open(target_file, 'r') as phil_file:
        phil_file_contents = phil_file.read()
      self.phil_text.SetValue(phil_file_contents)
      self.phil_path.ctr.SetValue(target_file)
    load_dlg.Destroy()

  def onDefault(self, e):
    # TODO: Generate default PHIL parameters
    pass

  def onCancel(self, e):
    self.stop_frames_sentinel()
    e.Skip()

class AveragingDialog(BaseDialog):
  def __init__(self, parent, run, params,
               label_style='bold',
               content_style='normal',
               *args, **kwargs):
    self.run = run
    self.params = params

    # Looks for a rungroup to use as a template
    # The averaging job is not assigned a rungroup
    run_rungroups = run.get_rungroups()
    if len(run_rungroups) == 0:
      # If this run has not been included in any rungroups try to use the rungroup
      # with the closest run
      all_rungroups = run.app.get_all_rungroups()
      if len(all_rungroups) == 0:
        self.template_rungroup = None
      else:
        distance = np.zeros(len(all_rungroups))
        for index, rungroup in enumerate(all_rungroups):
          first_run, last_run = rungroup.get_first_and_last_runs()
          distance[index] = min([
            abs(int(first_run.run) - int(run.run)),
            abs(int(last_run.run) - int(run.run))
          ])
        template_rungroup_id = all_rungroups[np.argmin(distance)].rungroup_id
        self.template_rungroup = run.app.get_rungroup(rungroup_id=template_rungroup_id)
    else:
      # If this run has been included in any rungroups - use that
      self.template_rungroup = run_rungroups[-1]

    BaseDialog.__init__(self, parent, label_style=label_style,
                        content_style=content_style, *args, **kwargs)

    # Image Average Options
    self.skip_images = gctr.SpinCtrl(self,
                                   label='Skip Images:',
                                   label_style='bold',
                                   label_size=(150, -1),
                                   ctrl_value=0,
                                   ctrl_min=0,
                                   ctrl_max=None)
    self.num_images_type = gctr.RadioCtrl(self,
                                   name='Use All Images',
                                   label='',
                                   label_style='normal',
                                   label_size=(100, -1),
                                   direction='vertical',
                                   items={'all': 'Use all images',
                                          'specify': 'Specify total images'})
    self.num_images = gctr.SpinCtrl(self,
                                   label='Number Images:',
                                   label_style='bold',
                                   label_size=(150, -1),
                                   ctrl_value=0,
                                   ctrl_min=0,
                                   ctrl_max=None)
    self.main_sizer.Add(self.skip_images, flag=wx.EXPAND | wx.ALL, border=10)
    self.main_sizer.Add(self.num_images_type, flag=wx.EXPAND | wx.ALL, border=10)
    self.main_sizer.Add(self.num_images, flag=wx.EXPAND | wx.ALL, border=10)
    self.skip_images.SetToolTip('Number of images to skip at the start of the dataset')
    self.num_images.SetToolTip('Maximum number of frames to average.')
    self.Bind(wx.EVT_RADIOBUTTON, self.onAllImages, self.num_images_type.all)
    self.Bind(wx.EVT_RADIOBUTTON, self.onSpecifyImages, self.num_images_type.specify)
    self.num_images.Disable()

    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box, flag=wx.EXPAND | wx.ALL, border=10)
    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)

  def onAllImages(self, e):
    self.num_images.Disable()

  def onSpecifyImages(self, e):
    self.num_images.Enable()

  def onOK(self, e):
    if self.template_rungroup is None:
      wx.MessageBox('Add this run to a rungroup in the Trials tab first!', 'Warning', wx.ICON_EXCLAMATION)
      return

    skip_images = self.skip_images.ctr.GetValue()
    if self.num_images_type.all.GetValue():
      num_images = 0
    else:
      num_images = self.num_images.ctr.GetValue()
    if num_images == 1:
      print("Average Aborted.\nNeed more than one image to average.")
      return
    else:
      from xfel.ui.db.job import AveragingJob
      job = AveragingJob(self.run.app)
      job.run = self.run
      job.rungroup = self.template_rungroup
      job.skip_images = skip_images
      job.num_images = num_images
      job.submit()
      self.Destroy()

class TrialTagSelectionDialog(BaseDialog):
  def __init__(self, parent,
               label_style='bold',
               content_style='normal',
               db=None,
               *args, **kwargs):
    BaseDialog.__init__(self, parent, label_style=label_style,
                        content_style=content_style, *args, **kwargs)
    self.db = db
    self.parent = parent
    self.run_paths = []
    self.tags = []
    self.tag_names = []

    self.trials = db.get_all_trials()
    trial_numbers = ["%d"%trial.trial for trial in self.trials]
    self.trials_ctrl = gctr.ChoiceCtrl(self,
                                       label='Trial number:',
                                       label_size=(120, -1),
                                       label_style='bold',
                                       choices=trial_numbers)
    self.main_sizer.Add(self.trials_ctrl, flag=wx.EXPAND | wx.ALL, border=10)


    self.trial_tags = gctr.CheckListCtrl(self,
                                         label='Tags (optional):',
                                         label_size=(40, -1),
                                         label_style='normal',
                                         ctrl_size=(200, -1),
                                         choices=[])
    self.main_sizer.Add(self.trial_tags, flag=wx.EXPAND | wx.ALL, border=10)

    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    self.SetTitle('Pick a trial and optionally some tags from that trial')

    self.Bind(wx.EVT_CHOICE, self.onTrialSelect, id=self.trials_ctrl.ctr.GetId())
    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)
    wx.CallAfter(self.refresh_tags)

  def onTrialSelect(self, e):
    self.refresh_tags()

  def refresh_tags(self):
    self.trial_tags.ctr.Clear()
    trial = self.trials[self.trials_ctrl.ctr.GetSelection()]
    self.tags = []
    self.tag_names = []
    tag_ids = []
    for run in trial.runs:
      for tag in run.tags:
        if tag.id not in tag_ids:
          self.tags.append(tag)
          tag_ids.append(tag.id)
          self.tag_names.append(tag.name)

    if self.tag_names:
      self.trial_tags.ctr.InsertItems(items=self.tag_names, pos=0)

  def onOK(self, e):
    from xfel.ui.db import get_run_path
    self.run_paths = []

    trial = self.trials[self.trials_ctrl.ctr.GetSelection()]
    tags = [self.tags[self.tag_names.index(t)] for t in self.trial_tags.ctr.GetCheckedStrings()]
    tag_ids = [t.id for t in tags]

    run_ids = []
    for rungroup in trial.rungroups:
      for run in rungroup.runs:
        if run.id not in run_ids:
          if len(tags) == 0:
            self.run_paths.append(os.path.join(
              get_run_path(self.parent.main.params.output_folder, trial, rungroup, run),
              'out'))
          else:
            run_tag_ids = [t.id for t in run.tags]
            for tag_id in tag_ids:
              if tag_id in run_tag_ids:
                run_ids.append(run.id)
                self.run_paths.append(os.path.join(
                  get_run_path(self.parent.main.params.output_folder, trial, rungroup, run),
                  'out'))
                break
    e.Skip()

class MultiRunTagDialog(BaseDialog):
  def __init__(self, parent,
               label_style='bold',
               content_style='normal',
               db=None,
               *args, **kwargs):
    BaseDialog.__init__(self, parent, label_style=label_style,
                        content_style=content_style, *args, **kwargs)
    self.parent = parent
    self.db = db
    self.db_tags = self.db.get_all_tags()
    self.db_tag_names = [t.name for t in self.db_tags]
    self.db_tags_selected_sort = self.db_tag_names
    self.db_runs = self.db.get_all_runs()
    self.db_run_numbers = [str(r.run) for r in self.db_runs]
    self.runs_selected = []
    self.buttons_for_runs_selected = []

    self.multiruntag_sizer = wx.BoxSizer(wx.VERTICAL)

    self.select_runs_panel = wx.Panel(self)
    self.select_runs_sizer = wx.BoxSizer(wx.HORIZONTAL)
    self.select_runs =  gctr.CheckListCtrl(self.select_runs_panel,
                                           label='Selected runs:',
                                           label_size=(200, -1),
                                           label_style='normal',
                                           ctrl_size=(240, 300),
                                           direction='vertical',
                                           choices=self.db_run_numbers)
    self.select_runs_sizer.Add(self.select_runs,
                               flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP,
                               border=10)
    self.select_runs_panel.SetSizer(self.select_runs_sizer)

    self.sort_type_panel = wx.Panel(self)
    self.sort_type_sizer = wx.BoxSizer(wx.VERTICAL)
    self.radio_sort = gctr.RadioCtrl(self.sort_type_panel,
                                     label='Sort tags by',
                                     label_style='normal',
                                     label_size=(80, -1),
                                     direction='horizontal',
                                     items={'id':'DB ID',
                                            'alpha':'name'})
    self.sort_type_sizer.Add(self.radio_sort,
                             flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP,
                             border=5)
    self.sort_type_panel.SetSizer(self.sort_type_sizer)

    self.button_panel = wx.Panel(self)
    self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)
    self.btn_add_tags    = wx.Button(self.button_panel, size=(115, -1),
                                     label='Add tags')
    self.btn_remove_tags = wx.Button(self.button_panel, size=(115, -1),
                                     label='Remove tags')
    self.button_sizer.Add(self.btn_add_tags,
                          flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP,
                          border=5)
    self.button_sizer.Add(self.btn_remove_tags,
                          flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP,
                          border=5)
    self.button_panel.SetSizer(self.button_sizer)

    # Add panels to main sizer
    self.multiruntag_sizer.Add(self.select_runs_panel,
                               flag=wx.EXPAND | wx.LEFT | wx.RIGHT)
    self.multiruntag_sizer.Add(self.sort_type_panel,
                               flag=wx.EXPAND | wx.BOTTOM | wx.TOP)
    self.multiruntag_sizer.Add(self.button_panel,
                               flag=wx.EXPAND | wx.BOTTOM | wx.TOP)
    self.main_sizer.Add(self.multiruntag_sizer,
                        flag=wx.EXPAND | wx.RIGHT | wx.LEFT)

    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    self.Layout()
    self.SetTitle('Manage multiple runs')

    self.Bind(wx.EVT_CHECKLISTBOX, self.onRunChoice, self.select_runs.ctr)
    self.Bind(wx.EVT_RADIOBUTTON, self.onSortDefault, self.radio_sort.id)
    self.Bind(wx.EVT_RADIOBUTTON, self.onSortAlphanum, self.radio_sort.alpha)
    self.Bind(wx.EVT_BUTTON, self.onAddTags, self.btn_add_tags)
    self.Bind(wx.EVT_BUTTON, self.onRemoveTags, self.btn_remove_tags)
    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)

  def onRunChoice(self, e):
    run_numbers_selected = self.select_runs.ctr.GetCheckedStrings()
    self.selected = {}
    for r in self.db_runs:
      if r.run in run_numbers_selected:
        self.selected[r.run] = [r]
    for b in self.parent.all_tag_buttons:
      if b.run.run in self.selected:
        self.selected[b.run.run].append(b)

  def onSortDefault(self, e):
    self.db_tags_selected_sort = self.db_tag_names

  def onSortAlphanum(self, e):
    self.db_tags_selected_sort = sorted(self.db_tag_names)

  def onAddTags(self, e):
    tag_dlg = wx.MultiChoiceDialog(self,
                                   message='Add these tags to selected runs',
                                   caption='Add tags to multiple runs',
                                   choices=self.db_tags_selected_sort)
    tag_dlg.Fit()

    if (tag_dlg.ShowModal() == wx.ID_OK):
      tag_indices = tag_dlg.GetSelections()
      add_tags = [t for t in self.db_tags if self.db_tag_names.index(t.name) in tag_indices]

      runs_with_tags_buttons = self.parent.all_tag_buttons

      for tag in add_tags:
        for run_number in self.selected.keys():
          run, button = self.selected[run_number]
          run_tags = run.app.get_run_tags(run.run_id)
          run_tag_ids = [t.tag_id for t in run_tags]
          if tag.tag_id not in run_tag_ids:
            run.add_tag(tag)
            button.tags.append(tag)
            button.update_label()


  def onRemoveTags(self, e):
    tag_dlg = wx.MultiChoiceDialog(self,
                                   message='Remove these tags from selected runs',
                                   caption='Remove tags from multiple runs',
                                   choices=self.db_tags_selected_sort)
    tag_dlg.Fit()

    if (tag_dlg.ShowModal() == wx.ID_OK):
      tag_indices = tag_dlg.GetSelections()
      remove_tags = [t for t in self.db_tags if self.db_tag_names.index(t.name) in tag_indices]

      for tag in remove_tags:
        for run_number in self.selected.keys():
          run, button = self.selected[run_number]
          run_tags = run.app.get_run_tags(run.run_id)
          run_tag_ids = [t.tag_id for t in run_tags]
          if tag.tag_id in run_tag_ids:
            run.remove_tag(tag)
            tag_on_button = [t for t in button.tags if t.tag_id == tag.tag_id][0] # proxy tag =/= other proxy tag
            button.tags.remove(tag_on_button)
            button.update_label()

  def onOK(self, e):
    e.Skip()

class TagDialog(BaseDialog):
  def __init__(self, parent,
               label_style='bold',
               content_style='normal',
               db=None,
               *args, **kwargs):
    BaseDialog.__init__(self, parent, label_style=label_style,
                        content_style=content_style, *args, **kwargs)

    self.db = db
    self.db_tags = self.db.get_all_tags()
    self.deleted_tags = []
    self.new_tags = []
    self.edited_tags =[]
    self.index = 0

    self.top_sizer = wx.BoxSizer(wx.HORIZONTAL)

    self.button_panel = wx.Panel(self)
    self.button_sizer = wx.BoxSizer(wx.VERTICAL)
    self.btn_add = wx.Button(self.button_panel, size=(120, -1),
                             label='Add Tag')
    self.btn_rmv = wx.Button(self.button_panel, size=(120, -1),
                             label='Remove Tags')
    self.btn_clr = wx.Button(self.button_panel, size=(120, -1),
                             label='Clear All')
    self.button_sizer.Add(self.btn_add)
    self.button_sizer.Add(self.btn_rmv)
    self.button_sizer.Add(self.btn_clr)
    self.button_panel.SetSizer(self.button_sizer)

    self.tag_panel = ScrolledPanel(self, size=(500, 400))
    self.tag_list = EdListCtrl(self.tag_panel,
                               style=wx.LC_REPORT | wx.SUNKEN_BORDER)
    self.tag_sizer = wx.BoxSizer(wx.VERTICAL)
    self.tag_panel.SetSizer(self.tag_sizer)

    self.tag_list.InsertColumn(0, 'Sample Tag', width=200)
    self.tag_list.InsertColumn(1, 'Comments', width=300)

    # Populate tags with current values from db
    if len(self.db_tags) > 0:
      for tag in self.db_tags:
        self.tag_list.InsertItem(self.index, str(tag.name))
        self.tag_list.SetItem(self.index, 1, str(tag.comment))
        self.tag_list.SetItemData(self.index, tag.tag_id)
        self.index += 1

    self.tag_sizer.Add(self.tag_list, 1, flag=wx.EXPAND)

    # Add panels to main sizer
    self.top_sizer.Add(self.button_panel,
                       flag=wx.LEFT, border=10)
    self.top_sizer.Add(self.tag_panel,
                       flag=wx.EXPAND | wx.RIGHT | wx.LEFT, border=10)
    self.main_sizer.Add(self.top_sizer,
                        flag=wx.EXPAND| wx.TOP | wx.BOTTOM, border=10)
    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box,
                   flag=wx.EXPAND | wx.ALL,
                   border=10)

    self.Layout()
    self.SetTitle('Manage Tags')

    # Button bindings
    self.Bind(wx.EVT_BUTTON, self.onAdd, self.btn_add)
    self.Bind(wx.EVT_BUTTON, self.onRemove, self.btn_rmv)
    self.Bind(wx.EVT_BUTTON, self.onClear, self.btn_clr)
    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)

  def onAdd(self, e):
    ''' Add a string item to list; focus on item & provide default tag name'''
    new_tag = ('default tag {}'.format(self.index), '', self.index)
    self.new_tags.append(new_tag)
    self.tag_list.InsertItem(self.index, new_tag[0])
    self.tag_list.SetItem(self.index, 1, new_tag[1])
    self.tag_list.SetItemData(self.index, -1)
    #self.tag_list.Select(self.index)
    #self.tag_list.Focus(self.index)
    self.index += 1

  def onRemove(self, e):
    selected_indices = getListCtrlSelection(self.tag_list)
    tag_ids = [self.tag_list.GetItemData(i) for i in selected_indices]
    self.deleted_tags = [i for i in self.db_tags if i.tag_id in tag_ids]

    for i in range(len(selected_indices)):
      to_delete = getListCtrlSelection(self.tag_list)
      self.tag_list.DeleteItem(to_delete[0])

  def onClear(self, e):
    warning = wx.MessageDialog(self,
                               message='Are you sure you want to delete all '
                                       'tags? \n This cannot be reversed!',
                               caption='Warning',
                               style=wx.YES_NO | wx.ICON_EXCLAMATION)
    if (warning.ShowModal() == wx.ID_YES):
      self.deleted_tags = self.db_tags
      self.tag_list.DeleteAllItems()
      self.index = 0
      self.db_tags = []

  def onOK(self, e):
    try:
      # Delete tags from DB
      for tag in self.deleted_tags:
        self.db.delete_tag(tag=tag)

      # Update names for edited tags
      all_items = [(self.tag_list.GetItemData(i),
                    self.tag_list.GetItem(itemIdx=i, col=0),
                    self.tag_list.GetItem(itemIdx=i, col=1))
                    for i in range(self.tag_list.GetItemCount())]

      self.db_tags = self.db.get_all_tags()
      tag_ids = [i.tag_id for i in self.db_tags]
      new_tag_names = [i[1].GetText() for i in all_items]

      if len([i for i in new_tag_names if new_tag_names.count(i) > 1]) != 0:
        wx.MessageBox('Need a unique tag name!', 'Warning',
                       wx.ICON_EXCLAMATION)
      else:
        for item in all_items:
          if item[0] in tag_ids:
            tag = self.db.get_tag(tag_id=item[0])
            tag.name = item[1].GetText()
            tag.comment = item[2].GetText()
          elif item[0] == -1:
            self.db.create_tag(name=item[1].GetText(), comment=item[2].GetText())

    except Exception as exception:
      print(str(exception))

    e.Skip()


class RunBlockDialog(BaseDialog):
  ''' Comes up when individual run block button is clicked; allows for run
  block settings to be manipulated by user '''

  def __init__(self, parent, db,
               block=None, trial=None,
               label_style='bold',
               content_style='normal',
               *args, **kwargs):

    self.parent = parent
    self.block = block
    self.all_blocks = []
    self.db = db
    self.use_ids = db.params.facility.name not in ['lcls']
    self.is_lcls = db.params.facility.name == 'lcls'

    all_runs = db.get_all_runs()
    if self.use_ids:
      runs_available = sorted([i.id for i in all_runs])
    else:
      runs_available = sorted([int(i.run) for i in all_runs])
    self.first_avail = min(runs_available)
    self.last_avail = max(runs_available)

    if block is None:
      runs = self.db.get_all_runs()
      if self.use_ids:
        run_numbers = [r.id for r in runs]
      else:
        run_numbers = list(sorted([int(r.run) for r in runs]))
      assert len(set(run_numbers)) == len(run_numbers)

      if trial is not None:
        self.all_blocks = trial.rungroups

      if len(runs) == 0:
        wx.MessageBox("No runs found", "Error", wx.OK | wx.ICON_EXCLAMATION)
        assert False # Close and destroy dialog properly here

      self.first_run = min(run_numbers)
      self.last_run = None

      class defaults(object):
        def __getattr__(self, item):
          if item == "detector_address":
            return 'CxiDs2.0:Cspad.0'
          elif item == "detz_parameter":
            return 580
          elif item == "format":
            return "cbf"
          elif item == "two_theta_low":
            return 12.5 # Defaults are from kapton tape experiments (this is kapton ring)
          elif item == "two_theta_high":
            return 22.8 # Defaults are from kapton tape experiments (this is water ring)
          elif item in ["extra_phil_str", "calib_dir", "dark_avg_path", "dark_stddev_path",
            "gain_map_path", "beamx", "beamy", "gain_mask_level", "untrusted_pixel_mask_path",
            "binning", "energy", "wavelength_offset", "spectrum_eV_per_pixel", "spectrum_eV_offset",
            "comment", "config_str", "extra_format_str"]:
            return None
          else:
            raise AttributeError(item)
      block = defaults()

    else:
      db = block.app
      self.first_run, self.last_run = block.get_first_and_last_runs()
      if self.first_run is None:
        self.first_run = self.first_avail
      else:
        if self.use_ids:
          if self.first_run is not None: self.first_run = self.first_run.id
          if self.last_run is not None: self.last_run = self.last_run.id
        else:
          if self.first_run is not None: self.first_run = int(self.first_run.run)
          if self.last_run is not None: self.last_run = int(self.last_run.run)

    self.orig_first_run = self.first_run
    self.orig_last_run = self.last_run

    BaseDialog.__init__(self, parent,
                        label_style=label_style,
                        content_style=content_style,
                        size=(600, 600),
                        *args, **kwargs)

    # Run block start / end points (choice widgets)

    if self.is_lcls:
      self.config_panel = wx.Panel(self)
      config_box = wx.StaticBox(self.config_panel, label='Configuration')
      self.config_sizer = wx.StaticBoxSizer(config_box)
      self.config_panel.SetSizer(self.config_sizer)
      self.config_panel.Hide()

    self.phil_panel = wx.Panel(self)
    phil_box = wx.StaticBox(self.phil_panel, label='Extra phil parameters')
    self.phil_sizer = wx.StaticBoxSizer(phil_box)
    self.phil_panel.SetSizer(self.phil_sizer)

    if self.is_lcls:
      self.format_panel = wx.Panel(self)
      format_box = wx.StaticBox(self.format_panel, label='Extra XTC format parameters')
      self.format_sizer = wx.StaticBoxSizer(format_box)
      self.format_panel.SetSizer(self.format_sizer)

    runblock_box = wx.StaticBox(self, label='Options')
    self.runblock_box_sizer = wx.StaticBoxSizer(runblock_box, wx.VERTICAL)
    self.runblock_panel = ScrolledPanel(self, size=(550, 225))
    self.runblock_sizer = wx.BoxSizer(wx.VERTICAL)
    self.runblock_panel.SetSizer(self.runblock_sizer)

    if self.is_lcls:
      # Configuration text ctrl (user can put in anything they want)
      self.config = gctr.PHILBox(self.config_panel,
                                 btn_import=True,
                                 btn_import_label='Import Config',
                                 btn_export=False,
                                 btn_default=True,
                                 btn_default_label='Default Config',
                                 ctr_size=(-1, 100),
                                 ctr_value=str(block.config_str))
      self.config_sizer.Add(self.config, 1, flag=wx.EXPAND | wx.ALL, border=10)

    # Extra phil
    self.phil = gctr.PHILBox(self.phil_panel,
                             btn_import=True,
                             btn_import_label='Import PHIL',
                             btn_export=False,
                             btn_default=True,
                             btn_default_label='Default PHIL',
                             ctr_size=(-1, 100),
                             ctr_value=str(block.extra_phil_str))
    self.phil_sizer.Add(self.phil, 1, flag=wx.EXPAND | wx.ALL, border=10)

    if self.is_lcls:
      # Extra format options text ctrl (user can put in anything they want)
      self.format = gctr.PHILBox(self.format_panel,
                                 btn_import=True,
                                 btn_import_label='Import PHIL',
                                 btn_export=False,
                                 btn_default=False,
                                 ctr_size=(-1, 100),
                                 ctr_value=str(block.extra_format_str))
      self.format_sizer.Add(self.format, 1, flag=wx.EXPAND | wx.ALL, border=10)

    self.start_stop_sizer = wx.FlexGridSizer(1, 3, 60, 20)

    self.runblocks_start = gctr.SpinCtrl(self.runblock_panel,
                                   label='Start run:',
                                   label_style='bold',
                                   label_size=(100, -1),
                                   ctrl_value=(self.first_run or self.last_avail),
                                   ctrl_min=self.first_avail,
                                   ctrl_max=self.last_avail)
    self.runblocks_end = gctr.SpinCtrl(self.runblock_panel,
                                   label='End run:',
                                   label_style='bold',
                                   label_size=(100, -1),
                                   ctrl_value=(self.last_run or self.last_avail),
                                   ctrl_min=self.first_avail,
                                   ctrl_max=self.last_avail)
    self.end_type = gctr.RadioCtrl(self.runblock_panel,
                                   name='rg_end_type',
                                   label='',
                                   label_style='normal',
                                   label_size=(100, -1),
                                   direction='vertical',
                                   items={'auto':'Auto add runs',
                                          'specify':'Specify end run'})
    self.start_stop_sizer.AddMany([(self.runblocks_start),
                                   (self.runblocks_end),
                                   (self.end_type)])
    self.runblock_sizer.Add(self.start_stop_sizer, flag=wx.EXPAND | wx.ALL, border=10)

    if self.is_lcls:
      # Detector address
      self.address = gctr.TextButtonCtrl(self.runblock_panel,
                                         name='rg_address',
                                         label='Detector Address:',
                                         label_style='bold',
                                         label_size=(100, -1),
                                         value=block.detector_address)
      self.runblock_sizer.Add(self.address, flag=wx.EXPAND | wx.ALL, border=10)


      # Beam XYZ (X, Y - pickle only)
      self.beam_xyz = gctr.OptionCtrl(self.runblock_panel,
                                      name='rg_beam_xyz',
                                      label='Beam:',
                                      label_style='bold',
                                      label_size=(100, -1),
                                      ctrl_size=(60, -1),
                                      items=[('X', block.beamx),
                                             ('Y', block.beamy),
                                             ('DetZ', block.detz_parameter)])
      self.runblock_sizer.Add(self.beam_xyz, flag=wx.EXPAND | wx.ALL, border=10)

    # Binning, energy, gain mask level
    if self.is_lcls:
      items = [('binning', block.binning),
               ('energy', block.energy)]
      self.bin_nrg_gain = gctr.OptionCtrl(self.runblock_panel,
                                          name='rg_bin_nrg_gain',
                                          ctrl_size=(80, -1),
                                          items=items)
      self.runblock_sizer.Add(self.bin_nrg_gain, flag=wx.EXPAND | wx.ALL, border=10)
      self.wavelength_offset = gctr.OptionCtrl(self.runblock_panel,
                                               name='rg_wavelength_offset',
                                               ctrl_size=(80, -1),
                                               items=[('wavelength_offset', block.wavelength_offset)])
      self.runblock_sizer.Add(self.wavelength_offset, flag=wx.EXPAND | wx.ALL, border=10)
      self.spectrum_calibration = gctr.OptionCtrl(self.runblock_panel,
                                                  name='rg_spectrum_calibration',
                                                  ctrl_size=(80, -1),
                                                  items=[('spectrum_eV_per_pixel', block.spectrum_eV_per_pixel),
                                                         ('spectrum_eV_offset', block.spectrum_eV_offset)])
      self.runblock_sizer.Add(self.spectrum_calibration, flag=wx.EXPAND | wx.ALL, border=10)
    else:
      self.energy = gctr.TextButtonCtrl(self.runblock_panel,
                                        name='rg_energy',
                                        label='Energy override',
                                        label_size=(150, -1))
      self.energy.ctr.SetValue(str(block.energy))
      self.runblock_sizer.Add(self.energy, flag=wx.EXPAND | wx.ALL, border=10)

    # Two theta values for droplet hit finding
    self.two_thetas = gctr.OptionCtrl(self.runblock_panel,
                                      name='rg_two_thetas',
                                      ctrl_size=(80, -1),
                                      items=[('two_theta_low', block.two_theta_low),
                                             ('two_theta_high', block.two_theta_high)])
    self.runblock_sizer.Add(self.two_thetas, flag=wx.EXPAND | wx.ALL, border=10)


    # Untrusted pixel mask path
    self.untrusted_path = gctr.TextButtonCtrl(self.runblock_panel,
                                              label='Untrusted Pixel Mask:',
                                              label_style='normal',
                                              label_size=(180, -1),
                                              big_button=True,
                                              value=str(block.untrusted_pixel_mask_path))
    self.runblock_sizer.Add(self.untrusted_path, flag=wx.EXPAND | wx.ALL,
                            border=10)

    # Comment
    self.comment = gctr.TextButtonCtrl(self.runblock_panel,
                                       label='Comment:',
                                       label_style='normal',
                                       label_size=(100, -1))
    self.comment.ctr.SetValue(str(block.comment))
    self.runblock_sizer.Add(self.comment, flag=wx.EXPAND | wx.ALL,
                            border=10)

    self.main_sizer.Add(self.phil_panel, flag=wx.EXPAND | wx.ALL, border=10)
    if self.is_lcls:
      self.main_sizer.Add(self.config_panel, flag=wx.EXPAND | wx.ALL, border=10)
      self.main_sizer.Add(self.format_panel, flag=wx.EXPAND | wx.ALL, border=10)
    self.runblock_box_sizer.Add(self.runblock_panel)
    self.main_sizer.Add(self.runblock_box_sizer, flag=wx.EXPAND | wx.ALL,
                        border=10)

    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    self.Bind(wx.EVT_RADIOBUTTON, self.onAutoEnd, self.end_type.auto)
    self.Bind(wx.EVT_RADIOBUTTON, self.onSpecifyEnd, self.end_type.specify)
    self.Bind(wx.EVT_BUTTON, self.onImportPhil, self.phil.btn_import)
    if self.is_lcls:
      self.Bind(wx.EVT_BUTTON, self.onImportFormat, self.format.btn_import)
    self.Bind(wx.EVT_BUTTON, self.onUntrustedBrowse,
              id=self.untrusted_path.btn_big.GetId())
    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)


    self.fill_in_fields()
    self.configure_controls()
    self.Layout()
    self.runblock_panel.SetupScrolling()
    self.SetTitle('Run Block Settings')

  def onAutoEnd(self, e):
    self.last_run = None
    self.runblocks_end.Disable()

  def onSpecifyEnd(self, e):
    self.runblocks_end.Enable()

  def onImportConfig(self, e):
    cfg_dlg = wx.FileDialog(self,
                            message="Load configuration file",
                            defaultDir=os.curdir,
                            defaultFile="*",
                            wildcard="*",
                            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                            )
    if cfg_dlg.ShowModal() == wx.ID_OK:
      config_file = cfg_dlg.GetPaths()[0]
      with open(config_file, 'r') as cfg:
        cfg_contents = cfg.read()
      self.config.ctr.SetValue(cfg_contents)
    cfg_dlg.Destroy()

  def onImportPhil(self, e):
    phil_dlg = wx.FileDialog(self,
                             message="Load phil file",
                             defaultDir=os.curdir,
                             defaultFile="*",
                             wildcard="*.phil",
                             style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                             )
    if phil_dlg.ShowModal() == wx.ID_OK:
      phil_file = phil_dlg.GetPaths()[0]
      with open(phil_file, 'r') as phil:
        phil_contents = phil.read()
      self.phil.ctr.SetValue(phil_contents)
    phil_dlg.Destroy()

  def onImportFormat(self, e):
    phil_dlg = wx.FileDialog(self,
                             message="Load phil file",
                             defaultDir=os.curdir,
                             defaultFile="*",
                             wildcard="*.phil",
                             style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                             )
    if phil_dlg.ShowModal() == wx.ID_OK:
      phil_file = phil_dlg.GetPaths()[0]
      with open(phil_file, 'r') as phil:
        phil_contents = phil.read()
      self.format.ctr.SetValue(phil_contents)
    phil_dlg.Destroy()

  def onDefaultConfig(self, e):
    # TODO: Generate default config parameters (re-do based on pickle / CBF)
    pass

  def onOK(self, e):
    try:
      first = int(self.runblocks_start.ctr.GetValue())
      assert first > 0 and first >= self.first_avail
      self.first_run = first
    except (ValueError, AssertionError) as e:
      wx.MessageBox("Please select a contiguous runs between %d and %d." % (self.first_avail, self.last_avail),
                    'Warning!', wx.ICON_EXCLAMATION)
      return
    if self.end_type.specify.GetValue() == 1:
      try:
        last = int(self.runblocks_end.ctr.GetValue())
        assert last > 0 and last <= self.last_avail and last >= first
        self.last_run = last
      except (ValueError, AssertionError) as e:
        wx.MessageBox("Please select contiguous runs between %d and %d." % (self.first_avail, self.last_avail),
                      'Warning!', wx.ICON_EXCLAMATION)
        return
    elif self.end_type.specify.GetValue() == 0:
      self.last_run = None
    else:
      assert False
    rg_open = self.last_run is None

    if self.is_lcls:
      # Validation
      if 'rayonix' in self.address.ctr.GetValue().lower():
        def is_none(string):
          return string is None or string in ['', 'None']
        if is_none(self.beam_xyz.X.GetValue()) or is_none(self.beam_xyz.Y.GetValue()) or \
             is_none(self.beam_xyz.DetZ.GetValue()):
          wx.MessageBox("For Rayonix, beam x, y, and DetZ are required, even if reference_geometry is specified. reference_geometry will take precedence.",
                        'Warning!', wx.ICON_EXCLAMATION)
          return

    rg_dict = dict(active=True,
                   open=rg_open,
                   extra_phil_str=self.phil.ctr.GetValue(),
                   untrusted_pixel_mask_path=self.untrusted_path.ctr.GetValue().strip(),
                   two_theta_low=self.two_thetas.two_theta_low.GetValue(),
                   two_theta_high=self.two_thetas.two_theta_high.GetValue(),
                   comment=self.comment.ctr.GetValue())

    if self.is_lcls:
      rg_dict['detz_parameter']=self.beam_xyz.DetZ.GetValue()
      rg_dict['beamx']=self.beam_xyz.X.GetValue()
      rg_dict['beamy']=self.beam_xyz.Y.GetValue()
      rg_dict['energy']=self.bin_nrg_gain.energy.GetValue()
      rg_dict['wavelength_offset']=self.wavelength_offset.wavelength_offset.GetValue()
      rg_dict['binning']=self.bin_nrg_gain.binning.GetValue()
      rg_dict['detector_address']=self.address.ctr.GetValue()
      rg_dict['config_str']=self.config.ctr.GetValue()
      rg_dict['extra_format_str']=self.format.ctr.GetValue()
      rg_dict['spectrum_eV_per_pixel']=self.spectrum_calibration.spectrum_eV_per_pixel.GetValue()
      rg_dict['spectrum_eV_offset']=self.spectrum_calibration.spectrum_eV_offset.GetValue()
    else:
      rg_dict['energy']=self.energy.ctr.GetValue()

    for key, value in six.iteritems(rg_dict):
      if str(value) == 'None' or str(value) == '':
        rg_dict[key] = None
      elif type(value) == bool:
        rg_dict[key] = int(value)

    if self.block is None:
      self.block = self.db.create_rungroup(**rg_dict)
      self.block.sync_runs(self.first_run, self.last_run, self.use_ids)
      self.parent.trial.add_rungroup(self.block)
    else:
      # if all the parameters are unchanged, do nothing
      all_the_same = [str(rg_dict[key]).strip() == str(getattr(self.block, key)).strip() for key in rg_dict].count(False) == 0
      all_the_same &= self.first_run == self.orig_first_run and self.last_run == self.orig_last_run
      if not all_the_same:
        # if all the parameters except open and comment are the same,
        # only update those fields
        keep_old_run_group = [str(rg_dict[key]).strip() == str(getattr(self.block, key)).strip() for key in rg_dict \
                              if key not in ['open', 'comment']].count(False) == 0
        if keep_old_run_group:
          main = self.parent.parent.GetParent().main
          running = main.job_sentinel is not None and main.job_sentinel.active
          if running:
            main.stop_job_sentinel()

          self.block.open = rg_open
          self.block.comment = rg_dict['comment']
          self.block.sync_runs(self.first_run, self.last_run, self.use_ids)

          if running:
            main.start_job_sentinel()

        else:
          # enough parameters have changed to warrant creating a new run group
          self.block.active = False
          self.block = self.db.create_rungroup(**rg_dict)
          self.block.sync_runs(self.first_run, self.last_run, self.use_ids)
          self.parent.trial.add_rungroup(self.block)

    e.Skip()

  def fill_in_fields(self):
    ''' If previous rungroups exist in trial, fill in fields in nascent block '''
    if len(self.all_blocks) > 0:
      last = self.all_blocks[-1]
      self.phil.ctr.SetValue(str(last.extra_phil_str))
      if self.is_lcls:
        self.address.ctr.SetValue(str(last.detector_address))
        self.config.ctr.SetValue(str(last.config_str))
        self.format.ctr.SetValue(str(last.extra_format_str))
        self.beam_xyz.DetZ.SetValue(str(last.detz_parameter))
        self.beam_xyz.X.SetValue(str(last.beamx))
        self.beam_xyz.Y.SetValue(str(last.beamy))
        self.bin_nrg_gain.binning.SetValue(str(last.binning))
        self.bin_nrg_gain.energy.SetValue(str(last.energy))
        self.wavelength_offset.wavelength_offset.SetValue(str(last.wavelength_offset))
        self.spectrum_calibration.spectrum_eV_per_pixel.SetValue(str(last.spectrum_eV_per_pixel))
        self.spectrum_calibration.spectrum_eV_offset.SetValue(str(last.spectrum_eV_offset))
      self.two_thetas.two_theta_low.SetValue(str(last.two_theta_low))
      self.two_thetas.two_theta_high.SetValue(str(last.two_theta_high))
      self.untrusted_path.ctr.SetValue(str(last.untrusted_pixel_mask_path))
      self.comment.ctr.SetValue(str(last.comment))

  def configure_controls(self):
    if self.last_run is None:
      self.runblocks_end.Disable()
      self.end_type.auto.SetValue(1)
    else:
      self.runblocks_end.Enable()
      self.end_type.specify.SetValue(1)

  def onDarkAvgBrowse(self, e):
    dark_dlg = wx.FileDialog(self,
                             message="Load dark average file",
                             defaultDir=os.curdir,
                             defaultFile="*.cbf",
                             wildcard="*.cbf",
                             style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                             )

    if dark_dlg.ShowModal() == wx.ID_OK:
      self.dark_avg_path.ctr.SetValue(dark_dlg.GetPaths()[0])
    dark_dlg.Destroy()

  def onDarkMapBrowse(self, e):
    dark_dlg = wx.FileDialog(self,
                             message="Load dark map file",
                             defaultDir=os.curdir,
                             defaultFile="*.cbf",
                             wildcard="*.cbf",
                             style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                             )

    if dark_dlg.ShowModal() == wx.ID_OK:
      self.gain_map_path.ctr.SetValue(dark_dlg.GetPaths()[0])
    dark_dlg.Destroy()

  def onDarkStdBrowse(self, e):
    dark_dlg = wx.FileDialog(self,
                             message="Load dark stddev file",
                             defaultDir=os.curdir,
                             defaultFile="*.cbf",
                             wildcard="*.cbf",
                             style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                             )

    if dark_dlg.ShowModal() == wx.ID_OK:
      self.dark_stddev_path.ctr.SetValue(dark_dlg.GetPaths()[0])
    dark_dlg.Destroy()

  def onUntrustedBrowse(self, e):
    dlg = wx.FileDialog(self,
                             message="Load untrusted pixel mask",
                             defaultDir=os.curdir,
                             defaultFile="*.mask",
                             wildcard="*.mask",
                             style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                             )

    if dlg.ShowModal() == wx.ID_OK:
      self.untrusted_path.ctr.SetValue(dlg.GetPaths()[0])
    dlg.Destroy()

  def onCalibDirBrowse(self, e):
    dlg = wx.DirDialog(self, "Choose calibration directory:",
                       style=wx.DD_DEFAULT_STYLE)

    if dlg.ShowModal() == wx.ID_OK:
      self.calib_dir.ctr.SetValue(dlg.GetPath())
    dlg.Destroy()
    e.Skip()

class SelectRunBlocksDialog(BaseDialog):
  def __init__(self, parent, trial,
               label_style='bold',
               content_style='normal',
               db=None,
               *args, **kwargs):
    BaseDialog.__init__(self, parent, label_style=label_style,
                        content_style=content_style, *args, **kwargs)

    self.db = db
    self.trial = trial

    self.top_sizer = wx.BoxSizer(wx.HORIZONTAL)

    self.button_panel = wx.Panel(self)
    self.button_sizer = wx.BoxSizer(wx.VERTICAL)
    self.button_panel.SetSizer(self.button_sizer)

    self.runblocks_panel = ScrolledPanel(self, size=(500, 400))

    # Populate rungroups with current values from db
    self.trial_rungroups = [t.id for t in trial.rungroups]
    self.all_rungroups = self.db.get_all_rungroups()
    choices = []
    selected = []
    use_ids = db.params.facility.name not in ['lcls']
    for rungroup in self.all_rungroups:
      selected.append(rungroup.id in self.trial_rungroups)
      first_run, last_run = rungroup.get_first_and_last_runs()
      if last_run is None:
        if first_run is None:
          desc = "[%d]"%(rungroup.id)
        else:
          desc = "[%d] %d+"%(rungroup.id, int(first_run.id) if use_ids else int(first_run.run))
      else:
        desc = "[%d] %d-%d"%(rungroup.id, int(first_run.id) if use_ids else int(first_run.run), \
                                          int(last_run.id) if use_ids else int(last_run.run))
      if rungroup.comment is not None:
        desc += " " + rungroup.comment

      choices.append(desc)

    self.runblocks_list = gctr.CheckListCtrl(self.runblocks_panel,
                                             label='Select runblocks',
                                             label_size=(40, -1),
                                             label_style='normal',
                                             ctrl_size=(450, 350),
                                             direction='vertical',
                                             choices=choices)
    for i in range(len(selected)):
      self.runblocks_list.ctr.Check(i, selected[i])

    self.runblocks_sizer = wx.BoxSizer(wx.VERTICAL)
    self.runblocks_panel.SetSizer(self.runblocks_sizer)

    self.runblocks_sizer.Add(self.runblocks_list, 1, flag=wx.EXPAND)

    # Add panels to main sizer
    self.top_sizer.Add(self.button_panel,
                       flag=wx.LEFT, border=10)
    self.top_sizer.Add(self.runblocks_panel,
                       flag=wx.EXPAND | wx.RIGHT | wx.LEFT, border=10)
    self.main_sizer.Add(self.top_sizer,
                        flag=wx.EXPAND| wx.TOP | wx.BOTTOM, border=10)
    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box,
                   flag=wx.EXPAND | wx.ALL,
                   border=10)

    self.Layout()
    self.SetTitle('Select run blocks')

    # Button bindings
    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)

  def onOK(self, e):
    trial_rungroups = [t.id for t in self.db.get_trial_rungroups(self.trial.id, only_active=False)]
    trials = self.db.get_all_trials()

    for i, rungroup in enumerate(self.all_rungroups):
      if self.runblocks_list.ctr.IsChecked(i):
        if not rungroup.id in trial_rungroups:
          self.trial.add_rungroup(rungroup)
        rungroup.active = True
      else:
        if rungroup.id in trial_rungroups:
          self.trial.remove_rungroup(rungroup)
        if not any([rungroup.id in [rg.id for rg in t.rungroups] for t in trials]):
          rungroup.active = False

    e.Skip()

class TrialDialog(BaseDialog):
  def __init__(self, parent, db,
               new=True,
               trial=None,
               label_style='bold',
               content_style='normal',
               *args, **kwargs):

    self.db = db
    self.new = new
    self.trial = trial
    self.all_trials = db.get_all_trials()

    BaseDialog.__init__(self, parent,
                        label_style=label_style,
                        content_style=content_style,
                        size=(600, 600),
                        *args, **kwargs)

    if trial is None:
      trials = [t.trial for t in self.all_trials]
      if len(trials) == 0:
        trial_number = 0
      else:
        trial_number = max(trials) + 1
      d_min = 1.5
    else:
      trial_number = trial.trial
      d_min = trial.d_min if trial.d_min is not None else 1.5

    self.trial_info = gctr.TwoButtonCtrl(self,
                                         label='Trial number:',
                                         label_size=(100, -1),
                                         label_style='bold',
                                         button1=True,
                                         button1_label='Import PHIL',
                                         button1_size=(120, -1),
                                         button2=True,
                                         button2_label='Default PHIL',
                                         button2_size=(120, -1),
                                         value="{}".format(trial_number))
    self.trial_comment = gctr.TextButtonCtrl(self,
                                             label='Comment:',
                                             label_size=(100, -1),
                                             label_style='bold',
                                             ghost_button=False)

    self.phil_box = gctr.RichTextCtrl(self, style=wx.VSCROLL, size=(-1, 400))

    choices = [('None', None)] + \
              [('Trial {}'.format(t.trial), t.trial) for t in self.all_trials]
    self.copy_runblocks = gctr.ChoiceCtrl(self,
                                          label='Copy runblocks from',
                                          label_style='normal',
                                          label_size=(180, -1),
                                          ctrl_size=(80, -1),
                                          choices=choices)
    self.copy_runblocks.ctr.SetSelection(0)
    self.throttle = gctr.SpinCtrl(self,
                                  name='trial_throttle',
                                  label='Percent events processed:',
                                  label_size=(180, -1),
                                  label_style='bold',
                                  ctrl_size=(80, -1),
                                  ctrl_value='100',
                                  ctrl_min=1,
                                  ctrl_max=100)
    self.num_bins = gctr.SpinCtrl(self,
                                  name='trial_num_bins',
                                  label='Number of bins:',
                                  label_size=(180, -1),
                                  label_style='bold',
                                  ctrl_size=(80, -1),
                                  ctrl_value='20',
                                  ctrl_min=1,
                                  ctrl_max=100,
                                  ctrl_step=1)
    self.d_min = gctr.SpinCtrl(self,
                               name='trial_d_min',
                               label='High res. limit ({}):'
                               ''.format(u'\N{ANGSTROM SIGN}'.encode('utf-8')),
                               label_size=(180, -1),
                               label_style='bold',
                               ctrl_size=(80, -1),
                               ctrl_value=str(d_min),
                               ctrl_min=0.1,
                               ctrl_max=100,
                               ctrl_step=0.1,
                               ctrl_digits=1)

    self.option_sizer = wx.FlexGridSizer(3, 2, 10, 20)
    self.option_sizer.AddMany([(self.copy_runblocks),
                               (0, 0),
                               (self.throttle),
                               (0, 0),
                               (self.num_bins),
                               (self.d_min)])

    self.main_sizer.Add(self.trial_info,
                        flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
                        border=10)
    self.main_sizer.Add(self.trial_comment,
                        flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
                        border=10)
    self.main_sizer.Add(self.phil_box, 1,
                        flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
                        border=10)
    self.main_sizer.Add(self.option_sizer, flag=wx.EXPAND | wx.ALL, border=10)


    # Dialog control
    if self.new:
      dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    else:
      dialog_box = self.CreateSeparatedButtonSizer(wx.OK)
    self.main_sizer.Add(dialog_box,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    self.Layout()

    if self.new:
      self.SetTitle('New Trial Settings')

      # If previous trials exist, propagate previous settings from them
      if len(self.all_trials) > 0:
        self.last_trial = self.all_trials[-1]
        target_phil_str = self.last_trial.target_phil_str
        self.trial_comment.ctr.SetValue(str(self.last_trial.comment))
        process_percent = self.last_trial.process_percent

      else:
        target_phil_str = ''
        process_percent = 100

    else:
      self.SetTitle('Trial {}'.format(self.trial.trial))
      target_phil_str = self.trial.target_phil_str
      self.trial_comment.ctr.SetValue(str(self.trial.comment))
      process_percent = self.trial.process_percent

      # Disable controls for viewing
      self.trial_info.button1.Disable()
      self.trial_info.button2.Disable()
      self.trial_info.ctr.SetEditable(False)
      self.copy_runblocks.Hide()
      self.phil_box.SetEditable(False)
      self.throttle.ctr.Disable()
      self.num_bins.ctr.Disable()
      self.d_min.ctr.Disable()

    if target_phil_str is None:
      target_phil_str = ""
    if process_percent is None:
      process_percent = 100
    self.phil_box.SetValue(target_phil_str)
    self.throttle.ctr.SetValue(process_percent)

    # Bindings
    self.Bind(wx.EVT_BUTTON, self.onBrowse, self.trial_info.button1)
    self.Bind(wx.EVT_BUTTON, self.onDefault, self.trial_info.button2)
    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)

  def onBrowse(self, e):
    ''' Open dialog for selecting PHIL file '''

    load_dlg = wx.FileDialog(self,
                             message="Load PHIL file",
                             defaultDir=os.curdir,
                             defaultFile="*.phil",
                             wildcard="*.phil",
                             style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                             )
    if load_dlg.ShowModal() == wx.ID_OK:
      target_file = load_dlg.GetPaths()[0]
      with open(target_file, 'r') as phil_file:
        phil_file_contents = phil_file.read()
      self.phil_box.SetValue(phil_file_contents)
    load_dlg.Destroy()

  def onDefault(self, e):
    # TODO: Generate default PHIL parameters
    pass

  def onOK(self, e):
    if self.new:
      target_phil_str = self.phil_box.GetValue()

      # Parameter validation
      dispatcher = self.db.params.dispatcher
      from xfel.ui import load_phil_scope_from_dispatcher
      phil_scope = load_phil_scope_from_dispatcher(dispatcher)

      from iotbx.phil import parse
      msg = None
      try:
        trial_params, unused = phil_scope.fetch(parse(target_phil_str), track_unused_definitions = True)
      except Exception as e:
        msg = '\nParameters incompatible with %s dispatcher:\n%s\n' % (dispatcher, str(e))
      else:
        if len(unused) > 0:
          msg = [str(item) for item in unused]
          msg = '\n'.join(['  %s' % line for line in msg])
          msg = 'The following definitions were not recognized:\n%s\n' % msg

        try:
          params = trial_params.extract()
        except Exception as e:
          if msg is None: msg = ""
          msg += '\nOne or more values could not be parsed:\n%s\n' % str(e)

      if msg is not None:
        msg += '\nFix the parameters and press OK again'
        msgdlg = wx.MessageDialog(self,
                                  message=msg,
                                  caption='Warning',
                                  style=wx.OK |  wx.ICON_EXCLAMATION)
        msgdlg.ShowModal()
        return

      comment = self.trial_comment.ctr.GetValue()
      process_percent = int(self.throttle.ctr.GetValue())
      d_min = float(self.d_min.ctr.GetValue())
      n_bins = int(self.num_bins.ctr.GetValue())
      if process_percent == 100:
        process_percent = None

      if self.trial is None:
        self.db.create_trial(
          trial = int(self.trial_info.ctr.GetValue()),
          active = False,
          target_phil_str = target_phil_str,
          comment = comment,
          process_percent = process_percent,
          d_min = d_min,
          n_bins = n_bins)
        self.trial = self.db.get_trial(trial_number=int(self.trial_info.ctr.GetValue()))

        template_trial_number = self.copy_runblocks.ctr.GetClientData(
          self.copy_runblocks.ctr.GetSelection())

        if template_trial_number is not None:
          template_trial = self.db.get_trial(trial_number=template_trial_number)
          for block in template_trial.rungroups:
            self.trial.add_rungroup(block)

      else:
        self.trial.target_phil_str = target_phil_str
        self.trial.comment = comment
    else:
      if self.trial.comment != self.trial_comment.ctr.GetValue():
        self.trial.comment = self.trial_comment.ctr.GetValue()
    e.Skip()

class DatasetDialog(BaseDialog):
  def __init__(self, parent, db,
               new=True,
               dataset=None,
               label_style='bold',
               content_style='normal',
               *args, **kwargs):

    self.db = db
    self.new = new
    self.dataset = dataset

    BaseDialog.__init__(self, parent,
                        label_style=label_style,
                        content_style=content_style,
                        size=(600, 600),
                        *args, **kwargs)

    self.name = gctr.TextButtonCtrl(self,
                                    label='Name:',
                                    label_size=(100, -1),
                                    label_style='bold',
                                    ghost_button=False)

    self.comment = gctr.TextButtonCtrl(self,
                                       label='Comment:',
                                       label_size=(100, -1),
                                       label_style='bold',
                                       ghost_button=False)

    self.tag_checklist = gctr.CheckListCtrl(self,
                                        label='Tags:',
                                        label_size=(200, -1),
                                        label_style='normal',
                                        ctrl_size=(150, 100),
                                        direction='vertical',
                                        choices=[])

    self.selection_type_radio = gctr.RadioCtrl(self,
                                           label='',
                                           label_style='normal',
                                           label_size=(-1, -1),
                                           direction='horizontal',
                                           items={'inter':'intersection',
                                                  'union':'union'})

    self.main_sizer.Add(self.name,
                        flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
                        border=10)
    self.main_sizer.Add(self.comment,
                        flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
                        border=10)
    self.main_sizer.Add(self.tag_checklist,
                        flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
                        border=10)
    self.main_sizer.Add(self.selection_type_radio,
                        flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
                        border=10)

    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    self.Layout()

    if self.new:
      self.SetTitle('New Dataset Settings')
      self.selection_type_radio.inter.SetValue(1)
    else:
      self.SetTitle('Dataset Settings')
      self.name.ctr.SetValue(str(self.dataset.name))
      self.comment.ctr.SetValue(str(self.dataset.comment))
      if self.dataset.tag_operator == "union":
        self.selection_type_radio.union.SetValue(1)
      elif self.dataset.tag_operator == "intersection":
        self.selection_type_radio.inter.SetValue(1)
      else:
        assert False

    # Bindings
    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)

    # Initialize tag list
    self.all_tags = db.get_all_tags()
    tag_names = [t.name for t in self.all_tags]
    self.dataset_tagnames = [t.name for t in self.dataset.tags] if self.dataset is not None else []
    if tag_names:
      self.tag_checklist.ctr.InsertItems(items=tag_names, pos=0)
      checked = [tag_idx for tag_idx, tag_name in enumerate(tag_names) if tag_name in self.dataset_tagnames]
      self.tag_checklist.ctr.SetCheckedItems(checked)

  def onOK(self, e):
    name = self.name.ctr.GetValue()
    comment = self.comment.ctr.GetValue()
    if self.selection_type_radio.union.GetValue() == 1:
      mode = 'union'
    else:
      mode = 'intersection'

    if self.new:
      self.dataset = self.db.create_dataset(name = name,
                                            comment = comment,
                                            active = False,
                                            tag_operator = mode)
    else:
      self.dataset.name = name
      self.dataset.comment = comment
      self.dataset.tag_operator = mode

    checked = self.tag_checklist.ctr.GetCheckedItems()
    for tag_idx, tag in enumerate(self.all_tags):
      if tag_idx in checked:
        if tag.name not in self.dataset_tagnames:
          self.dataset.add_tag(tag)
      else:
        if tag.name in self.dataset_tagnames:
          self.dataset.remove_tag(tag)

    e.Skip()

class TaskDialog(BaseDialog):
  def __init__(self, parent, db,
               dataset=None,
               task=None,
               label_style='bold',
               content_style='normal',
               *args, **kwargs):

    self.db = db
    self.dataset = dataset
    self.task = task
    self.all_trials = db.get_all_trials()
    self.all_trial_numbers = [str(t.trial) for t in self.all_trials]

    BaseDialog.__init__(self, parent,
                        label_style=label_style,
                        content_style=content_style,
                        size=(600, 600),
                        *args, **kwargs)

    self.type = gctr.ChoiceCtrl(self,
                                label='Task type:',
                                label_size=(100, -1),
                                ctrl_size=(150, -1),
                                choices=task_types)

    self.trial = gctr.ChoiceCtrl(self,
                                 label='Trial:',
                                 label_size=(100, -1),
                                 ctrl_size=(150, -1),
                                 choices=self.all_trial_numbers)

    self.phil_box = gctr.RichTextCtrl(self, style=wx.VSCROLL, size=(550, 400))

    self.main_sizer.Add(self.type,
                        flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
                        border=10)
    self.main_sizer.Add(self.trial,
                        flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
                        border=10)
    self.main_sizer.Add(self.phil_box, 1,
                        flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
                        border=10)

    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box,
                        flag=wx.EXPAND | wx.ALL,
                        border=10)

    self.Layout()

    if task is None:
      self.SetTitle('New Task Settings')

      # If previous tasks exist, propagate previous settings from them
      # XXX
      self.type.ctr.SetSelection(0)
      self.trial.ctr.SetSelection(0)
    else:
      self.SetTitle('Task Settings')
      self.type.ctr.SetSelection(task_types.index(self.task.type))
      self.trial.ctr.SetSelection(self.all_trial_numbers.index(str(self.task.trial.trial)))
      if self.task.parameters is not None:
        self.phil_box.SetValue(self.task.parameters)

    # Bindings
    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)

  def onOK(self, e):
    task_type = self.type.ctr.GetStringSelection()
    # Remember, a trial number doesn't necessarily match its id and they are not guaranteed to be consecutive
    trial = self.all_trials[self.all_trial_numbers.index(self.trial.ctr.GetStringSelection())]
    parameters = self.phil_box.GetValue()

    # Parameter validation
    from xfel.ui.db.task import Task
    from iotbx.phil import parse
    dispatcher, phil_scope = Task.get_phil_scope(self.db, task_type)

    if phil_scope is not None:
      msg = None
      try:
        task_params, unused = phil_scope.fetch(parse(parameters), track_unused_definitions = True)
      except Exception as e:
        msg = '\nParameters incompatible with %s dispatcher:\n%s\n' % (dispatcher, str(e))
      else:
        if len(unused) > 0:
          msg = [str(item) for item in unused]
          msg = '\n'.join(['  %s' % line for line in msg])
          msg = 'The following definitions were not recognized:\n%s\n' % msg

        try:
          params = task_params.extract()
        except Exception as e:
          if msg is None: msg = ""
          msg += '\nOne or more values could not be parsed:\n%s\n' % str(e)

      if msg is not None:
        msg += '\nFix the parameters and press OK again'
        msgdlg = wx.MessageDialog(self,
                                  message=msg,
                                  caption='Warning',
                                  style=wx.OK |  wx.ICON_EXCLAMATION)
        msgdlg.ShowModal()
        return

    if self.task is None:
      task = self.db.create_task(type = task_type,
                                 trial_id = trial.id,
                                 parameters = parameters)
      self.dataset.add_task(task)
    else:
      self.task.type = task_type
      self.task.trial_id = trial.id
      self.task.parameters = parameters

    e.Skip()

class SelectTasksDialog(BaseDialog):
  def __init__(self, parent, dataset,
               label_style='bold',
               content_style='normal',
               db=None,
               *args, **kwargs):
    BaseDialog.__init__(self, parent, label_style=label_style,
                        content_style=content_style, *args, **kwargs)

    self.db = db
    self.dataset = dataset

    self.top_sizer = wx.BoxSizer(wx.HORIZONTAL)

    self.button_panel = wx.Panel(self)
    self.button_sizer = wx.BoxSizer(wx.VERTICAL)
    self.button_panel.SetSizer(self.button_sizer)

    self.tasks_panel = ScrolledPanel(self, size=(500, 400))

    # Populate tasks with current values from db
    self.dataset_tasks = [t for t in dataset.tasks]
    self.dataset_task_ids = [t.id for t in self.dataset_tasks]
    self.all_tasks = self.db.get_all_tasks()
    choices = []
    selected = []
    for task in self.all_tasks:
      selected.append(task.id in self.dataset_task_ids)
      desc = "[%d] %s"%(task.id, task.type)
      if task.trial is not None:
        desc += " (trial %d)"%task.trial.trial

      choices.append(desc)

    self.tasks_list = gctr.CheckListCtrl(self.tasks_panel,
                                         label='Select tasks',
                                         label_size=(40, -1),
                                         label_style='normal',
                                         ctrl_size=(450, 350),
                                         direction='vertical',
                                         choices=choices)
    for i in range(len(selected)):
      self.tasks_list.ctr.Check(i, selected[i])

    self.tasks_sizer = wx.BoxSizer(wx.VERTICAL)
    self.tasks_panel.SetSizer(self.tasks_sizer)

    self.tasks_sizer.Add(self.tasks_list, 1, flag=wx.EXPAND)

    # Add panels to main sizer
    self.top_sizer.Add(self.button_panel,
                       flag=wx.LEFT, border=10)
    self.top_sizer.Add(self.tasks_panel,
                       flag=wx.EXPAND | wx.RIGHT | wx.LEFT, border=10)
    self.main_sizer.Add(self.top_sizer,
                        flag=wx.EXPAND| wx.TOP | wx.BOTTOM, border=10)
    # Dialog control
    dialog_box = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
    self.main_sizer.Add(dialog_box,
                   flag=wx.EXPAND | wx.ALL,
                   border=10)

    self.Layout()
    self.SetTitle('Select tasks')

    # Button bindings
    self.Bind(wx.EVT_BUTTON, self.onOK, id=wx.ID_OK)

  def onOK(self, e):
    for task in self.dataset_tasks:
      self.dataset.remove_task(task)

    for i, task in enumerate(self.all_tasks):
      if self.tasks_list.ctr.IsChecked(i):
        self.dataset.add_task(task)

    e.Skip()
