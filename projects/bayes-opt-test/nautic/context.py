import os
import re
import yaml
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, create_model
from colorama import Fore, Style
from .engine import Engine
from .flowx import flowx_cfg
from .taskx import taskx_cfg

# Simple wrapper for scalar alias values
class AliasRef:
    def __init__(self, src, target):
        self.src = src
        self.target = target
        self._container = None
        self._field = None

    def set_model(self, model):
         self._container, self._field = self.__set_model(model, self.target)

    def get(self):
        return self._container[self._field] \
            if isinstance(self._container, list) else getattr(self._container, self._field)


    def set(self, value):
        if isinstance(self._container, list):
            self._container[self._field] = value
        else:
            setattr(self._container, self._field, value)

    def __set_model(self, model, target):
        """
        Resolves a dotted/indexed path like 'layer_config[2].dropout' into:
        - parent container (object or list)
        - final attribute name or list index

        Returns:
            (container, attr_or_index)
        """
        parts = target.split('.')
        obj = model

        for part in parts[:-1]:
            try:
                if '[' in part:
                    name, idx = part[:-1].split('[')
                    obj = getattr(obj, name)[int(idx)]
                else:
                    obj = getattr(obj, part)
            except AttributeError as e:
                raise AttributeError(f"Failed to resolve path '{target}' in config spec: "
                                     f"'{self.src}' => '{self.target}")

        last = parts[-1]
        if '[' in last:
            name, idx = last[:-1].split('[')
            return getattr(obj, name), int(idx)
        else:
            return obj, last
    def __repr__(self):
        return f"AliasRef(src={self.src} => target={self.target}"




# Main Config wrapper
class Context:
    # Type mapping for explicit string declarations
    __TYPE_MAP = {
        '<int>': int,
        '<float>': float,
        '<str>': str,
        '<dict>': dict,
        '<list>': list,
        '<obj>': object,
    }

    ALIAS_PATTERN = re.compile(r"^\(\(\s*([a-zA-Z0-9_.]+)\s*\)\)$")

    @staticmethod
    def create(cfg_yaml: str,
               disable_nautic:bool=True,
               log_level: str = None,
               **kwargs) -> BaseModel:

        with open(cfg_yaml, "r") as f:
            raw_yaml = yaml.safe_load(f)

        if disable_nautic:
            from .logger_simple import Logger
        else:
            from .logger_prefect import Logger

        taskx_cfg.disable_nautic = disable_nautic
        flowx_cfg.disable_nautic = disable_nautic

        keywords = {'log': Logger,
                    'engine': Engine }

        for kw in keywords:
            if kw in raw_yaml:
                raise ValueError(f"Configuration YAML must not contain '{kw}' keyword.")
            raw_yaml[kw] = '<obj>'

        model_builder = Context.__build_model("Context", raw_yaml)
        # aliases contains
        (cfg, refs) =  Context.__get_values(raw_yaml)

        model = model_builder(**cfg)
        # resolve alias references
        for ref in refs:
            ref.set_model(model)

        for kw in keywords:
            setattr(model, kw, keywords[kw](ctx=model,
                                            log_level=log_level,
                                            disable_nautic=disable_nautic,
                                            **kwargs))

        # Resolve alias references
        return model



    @staticmethod
    def __build_model(name: str, content: Dict[str, Any]) -> BaseModel:
        """
        Build a nested BaseModel schema from content, automatically detecting
        alias targets and typing them as Optional[Any].
        """

        # Decide field type and default value
        def __resolve_type(value):
            if value is None:
                return (Optional[Any], None)
            elif isinstance(value, str) and value in Context.__TYPE_MAP:
                return (Optional[Context.__TYPE_MAP[value]], None)
            else:
                return (type(value), value)

        def build_fields(content, path=""):
            fields = {}
            for key, val in content.items():
                full_path = f"{path}.{key}" if path else key

                if isinstance(val, dict):
                    # Recursively build a submodel
                    submodel_fields = build_fields(val, path=full_path)
                    sub_model = create_model(
                        f"{name}_{key}".title().replace("_", ""), **submodel_fields)
                    fields[key] = (sub_model, ...)
                elif isinstance(val, list):
                    elem_type = type(val[0]) if val else Any
                    fields[key] = (List[elem_type], val)
                else:
                    is_alias_reference = isinstance(val, str) and Context.ALIAS_PATTERN.match(val)
                    if is_alias_reference:
                        fields[key] = (Optional[Any], None)
                    else:
                        fields[key] = __resolve_type(val)
            return fields

        top_fields = build_fields(content)
        return create_model(name, **top_fields)


    @staticmethod
    def __get_values(data, path="", refs=None):
        if refs is None:
            refs = set()

        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                sub_path = f"{path}.{k}" if path else k
                val, _ = Context.__get_values(v, sub_path, refs)
                result[k] = val
            return result, refs

        elif isinstance(data, list):
            result = []
            for i, v in enumerate(data):
                sub_path = f"{path}[{i}]"
                val, _ = Context.__get_values(v, sub_path, refs)
                result.append(val)
            return result, refs

        if isinstance(data, str):
            alias_match = Context.ALIAS_PATTERN.match(data)
            if alias_match:
                ref_path = alias_match.group(1).strip()
                ref = AliasRef(src=path, target=ref_path)
                refs.add(ref)
                return ref, refs
            elif data in Context.__TYPE_MAP:
                return None, refs
            else:
                return data, refs
        else:
            return data, refs





    def __getattr__(self, name):
        return getattr(self._data, name)

    def __repr__(self):
        return repr(self._data)
