import os
import os.path as osp

import importlib.util
import inspect
from types import SimpleNamespace

class Engine:
    def __init__(self, ctx, path=None, **kwargs):
        self.__tasks = {}
        self.context = ctx
        self.__load(path)
        if len(self.__tasks) == 0:
            raise ValueError("No taskx methods found in the specified path.")

    def __load(self, path):
        root_path = os.path.abspath(path or osp.join(os.getcwd(), 'tasks'))

        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                if not filename.endswith(".py") or filename.startswith("__"):
                    continue

                file_path = os.path.join(dirpath, filename)

                # Compute the module name and namespace path
                rel_path = os.path.relpath(file_path, root_path)
                module_parts = rel_path.replace(".py", "").split(os.sep)
                module_name = "_".join(module_parts)  # unique ID for importlib
                namespace_parts = module_parts[:-1]  # all folders before the .py file


                # Load module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if not spec or not spec.loader:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find taskx methods in all classes
                for _, cls in inspect.getmembers(module, inspect.isclass):
                    for method_name in dir(cls):
                        raw_attr = cls.__dict__.get(method_name)
                        if not isinstance(raw_attr, staticmethod):
                            continue

                        func = raw_attr.__func__

                        if not getattr(func, "_is_taskx", False):
                            continue

                        task_func = getattr(cls, method_name)

                        def make_bound_task(task_func, task_name):
                            def bound_wrapper(*args, **kwargs):
                                if args:
                                    raise TypeError(
                                        f"All arguments to task '{task_name}' must be passed by name (no positional arguments allowed)."
                                    )
                                return task_func(self.context, **kwargs)
                            return bound_wrapper

                        bound = make_bound_task(task_func, method_name)

                        # Attach method to correct namespace
                        target = self._ensure_namespace(namespace_parts)
                        setattr(target, method_name, bound)
                        self.__tasks["/".join(namespace_parts + [method_name])] = bound

                        # Also bind to top-level for flat access if no namespace
                        if not namespace_parts:
                            setattr(self, method_name, bound)

    def _ensure_namespace(self, parts):
        current = self
        for part in parts:
            if not hasattr(current, part):
                ns = SimpleNamespace()
                setattr(current, part, ns)
                current = ns
            else:
                current = getattr(current, part)
        return current

    @property
    def tasks(self):
        return list(self.__tasks.keys())