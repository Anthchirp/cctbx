"""
The state object is a container of references. 
The origin of any signal related to changes of state.
"""
from pathlib import Path

from PySide2.QtCore import QObject, QTimer, Signal, Slot
from iotbx.data_manager import DataManager

from .ref import Ref,ModelRef,MapRef,SelectionRef, RestraintsRef, RestraintRef, ResultsRef
from ...last.python_utils import DotDict
from .data import MolecularModelData, RealSpaceMapData



# class StateSignalEmitter(QObject):
#   """
#   These are signals emitted from the state object (the Model in MVC). 
#   It is the responsibility of the viewer to listen and implement these signals.
#   """
#   signal_dm_changed = Signal()
#   #signal_model_change = Signal()
#   signal_map_change = Signal()
#   #signal_selection_change = Signal()
#   #signal_references_change = Signal() # the list of references changed, not just change in active
#   signal_load_active_model = Signal()
#   signal_load_active_map = Signal()

#   # Style properties
#   signal_iso_change = Signal(str,float) # (ref_id, iso value)
 
#   signal_repr_change = Signal(str,list) # (ref_id, list of desired reprs)
#   signal_viz_change = Signal(str,bool) # change visibility (ref_id, on/off)

class StateSignals(QObject):
  style_change = Signal(str)# json
  tab_change = Signal(str) # tab name TOOD: Move? Not really a state thing...
  color_change = Signal(Ref)
  model_change = Signal(object) # model ref
  map_change = Signal(object) # map ref
  data_change = Signal() # The data manager has changed
  restraints_change = Signal(object) # restraits ref
  selection_change = Signal(object) # selection  ref
  references_change = Signal() # generic, TODO: refactor out
  results_change = Signal(object)
  repr_change = Signal(str,list) # (ref_id, list of desired reprs)
  viz_change = Signal(str,bool) # change visibility (ref_id, on/off)
  picking_level = Signal(int) # one of:  1 for "residue" or  0 for "element"
  sync = Signal()
  clear = Signal(str) # reload all active objects, send the messagebox message
  select = Signal(object) # select a ref object

class State:

  @classmethod
  def from_empty(cls):
    dm = DataManager()
    return cls(dm)

  def __init__(self, data_manager):
    # Props are more complex data structures
    self._active_model_ref = None
    self._active_map_ref = None
    self._active_selection_ref = None
    self._data_manager = data_manager
    self._model = None
    self._map_manager = None
    self._iso = 0.5
    #self.associations = {} # model: map associations
    self.references = {} # dictionary of all 'objects' tracked by the State
    self.params = DotDict()
    self.params.default_format = 'pdb'

    self.signals = StateSignals()
    self.signals.data_change.connect(self._data_manager_changed)

    # Initialize references

    # models
    for name in self.data_manager.get_model_names():
      print("Model name: ",name)
      model = self.data_manager.get_model(filename=name)
      self.add_ref_from_mmtbx_model(model,filename=name)

    # maps
    for name in self.data_manager.get_real_map_names():
      print("Map name: ",name)
      map_manager = self.data_manager.get_real_map(filename=name)
      self.add_ref_from_mmtbx_map(map_manager,filename=name)


    # TODO: remove this
    #self._guess_associations() # TODO: Make associations explicit

  def _sync(self):
    # Run this after all controllers are initialized
    self.signals.model_change.emit(self.active_model_ref)
    self.signals.map_change.emit(self.active_map_ref)


  # def last_ref(self):
  #   # dictionaries maintain insertion order in >3.7
  #   if len(self.references)==0:
  #     return None
  #   else:
  #     last_key, last_ref = list(self.references.items())[-1]
  #     return last_ref

  # def last_ref_from_filename(self,filename):
  #   # Return the last inserted reference matching a given filename
  #   for key,ref in reversed(list(self.references.items())):
  #     if hasattr(ref.data,"filename"):
  #       if filename == ref.data.filename or filename == ref.data.filepath:
  #         return ref
  #   return None


  
  # def get_ref_from_mmtbx_model(self,model):
  #   if model in list(self.data_manager._models.values()):
  #     rev_d = {value:key for key,value in self.data_manager._models}
  #     filename = rev_d[model]
  #     ref = self.last_ref_from_filename(filename)
  #     assert isinstance(ref,ModelRef),f"Expected model ref, got: {type(ref)}"
  #     return ref
  #   else:
  #     pass

  def add_ref(self,ref):
    if ref in self.references:
      return

    # Add the new reference
    print(f"Adding ref of type {ref.__class__.__name__}: {ref.id}")

    assert isinstance(ref,Ref), "Must add instance of Ref or subclass"
    self.references[ref.id] = ref
    ref.state = self

    if isinstance(ref,ModelRef):
      #self.active_model_ref = ref
      #self.signals.model_change.emit(self.active_model)
      pass
    elif isinstance(ref,MapRef):
      #self.active_map_ref = ref
      #self.signals.model_change.emit(self.active_model)
      pass
    elif isinstance(ref,SelectionRef):
      self.active_selection_ref = ref
      self.signals.selection_change.emit(self.active_selection_ref)
      pass
    elif isinstance(ref,(RestraintRef,RestraintsRef)):
      pass # access through model
      #self.signals.selection_change.emit(self.active_selection_ref)

    elif isinstance(ref,(ResultsRef)):
      self.signals.results_change.emit(ref)
    else:
      raise ValueError(f"ref provided not among those expected: {ref}")

    

  def _remove_ref(self,ref):
    assert isinstance(ref,Ref), "Must add instance of Ref or subclass"
    del self.references[ref.id]
  
  def add_ref_from_mmtbx_model(self,model,filename=None):
    # filepath = None
    # name = filename
    # if name is None:
    #   input = model.get_model_input()
    #   if input is not None:
    #     if hasattr(input,"_source_info"):
    #       filename = input._source_info.split("file")[1]
    #       if filename not in ["",None]:
    #         filename = Path(filename)
    #         if filename.exists():
    #           name = filename
    if filename is not None:
      filepath = str(Path(filename).absolute())
    
    data = MolecularModelData(filepath=filepath,model=model)
    ref = ModelRef(data=data)
    self.add_ref(ref)
    return ref

  def add_ref_from_mmtbx_map(self,map_manager,filename=None):
    # filepath = None
    # name = filename
    # if name is None:
    #   if map_manager.file_name not in ["",None]:
    #     filename = Path(map_manager.file_name)
    #     if filename.exists():
    #       name = filename
    if filename is not None:
      filepath = str(Path(filename).absolute())
    data = RealSpaceMapData(filepath=filepath,map_manager=map_manager)
    ref = MapRef(data=data,model_ref=None)
    self.add_ref(ref)
    return ref


  @property
  def references_model(self):
    return [value for key,value in self.references.items() if isinstance(value,ModelRef)]
  @property
  def references_map(self):
    return [value for key,value in self.references.items() if isinstance(value,MapRef)]

  @property
  def references_selection(self):
    return [value for key,value in self.references.items() if isinstance(value,SelectionRef)]

  @property
  def state(self):
    return self

  @property
  def data_manager(self):
    return self._data_manager

  @data_manager.setter
  def data_manager(self,value):
    self._data_manager = value
    self._data_manager_changed()

  @property
  def dm(self):
    # alias
    return self.data_manager

  def _data_manager_changed(self):
    # Call this to emit a signal the data_manager changed.
    # 
    # Unless putting signals in data manager, this must
    # be called explicitly/manually if the data manager changes.
    #self.signals.data_manager_changed.emit()
    print("_data_manager_changed")
    model_refs = [ref for ref in self.references_model]
    model_keys = [ref.data.filepath for ref in model_refs]
    print("model keys")
    print(model_keys)
    for filename in self.data_manager.get_model_names():
      print("filename",filename)
      if filename not in model_keys:
        print(f"New file found in data manager: {filename} and not found in references: {model_keys}")
        model = self.data_manager.get_model(filename=filename)
        ref = self.add_ref_from_mmtbx_model(model,filename=filename)
        
        self.signals.model_change.emit(self._active_model_ref) # No change, just trigger update
        

    map_refs = [ref for ref in self.references_map]
    map_keys = [ref.data.filepath for ref in map_refs]
    for filename in self.data_manager.get_real_map_names():
      if filename not in map_keys:
        print(f"New file found in data manager: {filename} and not found in references: {map_keys}")
        map_manager = self.data_manager.get_real_map(filename=filename)
        self.add_ref_from_mmtbx_map(map_manager,filename=filename)
        self.signals.map_change.emit(self.active_map_ref) # No change, just trigger update
  #####################################
  # Models / Mols
  #####################################

  @property
  def active_model_ref(self):
    return self._active_model_ref

  @active_model_ref.setter
  def active_model_ref(self,value):
    if value is None:
      self._active_model_ref = None
    else:
      assert isinstance(value,Ref), "Set active_model_ref with instance of Ref or subclass"
      assert value in self.references.values(), "Cannot set active ref before adding to state"
      self._active_model_ref = value
      self.signals.model_change.emit(self.active_model_ref)
    self.signals.references_change.emit()

  @property
  def active_model(self):
    if self.active_model_ref is not None:
      return self.active_model_ref.model

  @property
  def model(self):
    # alias
    return self.active_model

  @property
  def active_mol(self):
    # the active mol
    if self.active_model_ref is not None:
      return self.active_model_ref.mol

  @property
  def mol(self):
    # alias
    return self.active_mol


  #####################################
  # Maps
  #####################################
  @property
  def active_map_ref(self):
    return self._active_map_ref

  @active_map_ref.setter
  def active_map_ref(self,value):
    if value is None:
      self._active_map_ref = None
    else:
      assert isinstance(value,Ref), "Set active_model_ref with instance of Ref or subclass"
      assert value in self.references.values(), "Cannot set active ref before adding to state"
      self._active_map_ref = value
      self.signals.map_change.emit(self.active_map_ref)
    self.signals.references_change.emit()


  @property
  def active_map(self):
    if self.active_map_ref is not None:
      return self.active_map_ref.map_manager

  # def _guess_associations(self):
  #   # Try to guess map/model associations if not specified.
  #   # TODO: This is horrible
  #   for model_ref,map_ref in zip(self.references_model,self.references_map):
  #     self.associations[model_ref] = map_ref
  #     self.associations[map_ref] = model_ref
    
  #   # TODO: remove this, load maps no matter what
  #   for map_ref in self.references_map:
  #     if map_ref not in self.associations:
  #       model_ref = self.references_model[0]
  #       self.associations[model_ref] = map_ref
  #       self.associations[map_ref] = model_ref

  #####################################
  # Selections
  #####################################


  @property
  def active_selection_ref(self):
    return self._active_selection_ref

  @active_selection_ref.setter
  def active_selection_ref(self,value):
    if value is None:
      self._active_selection_ref =None
    else:
      assert isinstance(value,Ref), "Set active_model_ref with instance of Ref or subclass"
      assert value in self.references.values(), "Cannot set active ref before adding to state"
      self._active_selection_ref = value
    
    self.signals.selection_change.emit(value)
    self.signals.references_change.emit()


  #####################################
  # Restraints
  #####################################
  
  @property 
  def active_restraint_ref(self):
    return self.active_model_ref.restraints

