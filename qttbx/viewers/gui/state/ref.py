"""
A Ref is a top level container for data. It provides:
 1. unification of indentifiers across various programs. 
 2. 'Instances' of data, where identical data appears in multiple objects
 3. An additional unique identifier to pass around in signals
"""


from pathlib import Path
import uuid
import hashlib
import json
import copy
from dataclasses import fields, replace
import pandas as pd

from .style import Style
from ...last.mol import MolDF
from ...last.selection_utils import SelectionQuery
from .data import MolecularModelData, RealSpaceMapData
from .results import Result, QscoreResult
from .base import DataClassBase
from .restraints import Restraint, Restraints
from typing import Optional




class Ref:
  """
  The fundamental object in the app.
  """



  def __init__(self,data: DataClassBase,style: Optional[Style] = None, show: bool =True):
    self._data = data
    self._id = self._generate_uuid()
    self._external_ids = {}

    self._state = None
    self._label = None
    self._show_in_list = show
    self.entry = None # set later the entry controller object
    self.results = {} # program_name: result_ref


    if style is None:
      style = Style.from_default(self.id)
    else:
      style = replace(style,ref_id=self.id,query=self.query)
    self._style = style
    self.style_history = []


  @property
  def id(self):
    return self._id

  @property
  def data(self):
    return self._data 

  @property
  def external_ids(self):
    return self._external_ids



  @property
  def state(self):
    assert self._state is not None, "State was never set for this ref"
    return self._state
    
  @state.setter
  def state(self,value):
    self._state = value

  @property
  def style(self):
    return self._style

  @style.setter
  def style(self,value):
    assert isinstance(value,Style), "Set with Style class"
    self.state.signals.style_change.emit(value.to_json())
    self.style_history.append(self.style)
    self._style = value
    
  @property
  def last_visible_style(self):
    for style in reversed(self.style_history):
      if style.visible:
        return style

  @property
  def show_in_list(self):
    return self._show_in_list

  @ show_in_list.setter
  def show_in_list(self,value):
    self._show_in_list = value

  @property 
  def label(self):
    if self._label is None:
      self._label = self._truncate_string(f"{self.__class__.__name__} {self.id}")
    return self._label

  @label.setter
  def label(self,value):
    self._label = value

  @property
  def query(self):
    raise NotImplementedError




  def _truncate_string(self,path, max_len=20):
    if len(path) > max_len:
      return path[:max_len // 2] + "..." + path[-max_len // 2:]
    else:
      return path

  @staticmethod
  def _generate_uuid(length: int=24):
    # Generate a UUID
    full_uuid = str(uuid.uuid4())

    # Hash the UUID
    hashed_uuid = hashlib.sha1(full_uuid.encode()).hexdigest()

    # Truncate to the desired length
    short_uuid = hashed_uuid[:length]
    return short_uuid

class ModelRef(Ref):
  def __init__(self,data: MolecularModelData, style: Optional[Style] = None):
    super().__init__(data=data,style=style)
    self._mol = None
    self._restraints = None
    
    

  @classmethod
  def from_filename(cls,filename: str):
    model_data = MolecularModelData(filename=filename)
    return cls(model_data)

  @property
  def model(self):
    return self.data.model

  @property
  def restraints(self):
    return self._restraints

  @restraints.setter
  def restraints(self,value):
    assert value.__class__.__name__ == "RestraintsRef", "Set restraints with a RestraintsRef"
    self._restraints = value
    self.state.signals.restraints_change.emit(value)

  @property
  def has_restraints(self):
    return self.restraints is not None

  @property
  def mol(self):
    if self._mol is None:
      mol = MolDF.from_mmtbx_model(self.model)
      self._mol = mol
    return self._mol

  @property
  def label(self):
    if self._label is None:
      if self.data.filename is not None:
        try:
          # if key is a path, get the stem
          self._label = self._truncate_string(Path(self.data.filename).name)
        except:
          self._label = self._truncate_string(self.data.filename)
      else:
        self._label = super().label

    return self._label

  @property
  def model_ref(self):
    return self
  @property
  def query(self):    
    return SelectionQuery.from_model_ref(self)
  

class MapRef(Ref):
  def __init__(self,data: RealSpaceMapData, model_ref: Optional[ModelRef] = None,style: Optional[Style] = None):
    super().__init__(data=data,style=style)
    self._model_ref = model_ref


  @property
  def map_manager(self):
    return self.data.map_manager

  @property
  def model_ref(self):
    return self._model_ref

  
  @model_ref.setter
  def model_ref(self,value):
    self._model_ref = value

  @property
  def label(self):
    if self._label is None:
      if self.data.filename is not None:
        try:
          # if key is a path, get the stem
          self._label = self._truncate_string(Path(self.data.filename).name)
        except:
          self._label = self._truncate_string(self.data.filename)
      else:
        self._label = super().label

    return self._label

  @property
  def query(self):
    return None

class SelectionRef(Ref):
  def __init__(self,data: SelectionQuery,model_ref: ModelRef,  show: Optional[bool] = True):
    assert show is not None, "Be explicit about whether to show selection ref in list of selections or not"
    assert ModelRef is not None, "Selection Ref cannot exist without reference model"
    assert data is not None, "Provide a Selection query as the data"
    self._show_in_list = show
    

    self._model_ref = model_ref
    self._query = data
    self._query.params.refId=model_ref.id
    super().__init__(data=data, style=model_ref.style)

  @property
  def show_in_list(self):
    return self._show_in_list
  @show_in_list.setter
  def show_in_list(self,value):
    self._show_in_list = value
    
  @property
  def query(self):
    query = SelectionQuery.from_model_ref(self.model_ref)
    query.selections = self._query.selections
    return query

  @property
  def model_ref(self):
    return self._model_ref

  @property
  def phenix_string(self):
    raise NotImplementedError

  def to_json(self,indent=2):
    d = {"phenix_string":self.query.phenix_string,
         "pandas_string":self.query.pandas_query,
         "query":self.query.to_json()
         }
    return json.dumps(d,indent=indent)


class RestraintRef(Ref):
  def __init__(self,data: Restraint, model_ref: ModelRef):
    self._selection_ref = None
    self.model_ref = model_ref
    super().__init__(data=data, style=model_ref.style)

  @property
  def selection_ref(self):
    if self._selection_ref is None:
      query = SelectionQuery.from_i_seqs(self.model_ref.mol.atom_sites,self.data.i_seqs)
      self._selection_ref = SelectionRef(data=query,model_ref=self.model_ref,show=False)
    return self._selection_ref


class RestraintsRef(Ref):
  def __init__(self,data: Restraints, model_ref: ModelRef):
    super().__init__(data=data,style=model_ref.style)
    self.model_ref = model_ref
    self.supported_restraints = ['bond','angle']
    for field in fields(self.data.__class__):
      field_name = field.name
      field_value = getattr(self.data, field_name)
      if field_name in self.supported_restraints and field_value is not None:
          field_value = [RestraintRef(data=data,model_ref=model_ref) for data in field_value]
      setattr(self,field_name,field_value)

  @property
  def dfs(self):
    dfs = {}
    for field in fields(self.data.__class__):
      field_name = field.name
      field_value = getattr(self, field_name)
      if field_name in self.supported_restraints and field_value is not None:
        df = pd.DataFrame([dataclass_instance.data.__dict__ for dataclass_instance in field_value])
        df = df.round(2)
        # Drop some columns for now
        columns_to_drop = ['labels', 'slack', 'rt_mx',"sym_op_i"]  
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df.drop(columns=columns_to_drop, inplace=True)
        # Sort columns
        sort_order = ['residual',"delta",'ideal','model','i_seq']
        matched_columns = []
        for string in sort_order:
            matched_columns += [col for col in df.columns if string in col]

        # append the remaining columns
        remaining_columns = [col for col in df.columns if col not in matched_columns]
        new_column_order = matched_columns + remaining_columns
        df = df[new_column_order]

        # sort rows
        df = df.sort_values(by='residual',ascending=False)

        dfs[field_name] = df
    return dfs


  @classmethod
  def from_model_ref(cls,model_ref):
    data = Restraints.from_model_ref(model_ref)
    return cls(data=data,model_ref=model_ref)



class ResultsRef(Ref):
  def __init__(self,data: Result):
    super().__init__(data=data)

class ModelResultsRef(ResultsRef):
  def __init__(self,data: Result, model_ref: ModelRef, selection_ref: Optional[SelectionRef] = None):
    super().__init__(data=data)
    self.model_ref = model_ref
    self.selection_ref = selection_ref
    

class QscoreRef(ModelResultsRef):
    def __init__(self,data: QscoreResult,model_ref: ModelRef, selection_ref: Optional[SelectionRef] = None):
      super().__init__(data=data,model_ref=model_ref,selection_ref=selection_ref)
      self.model_ref.results["qscore"] = self



