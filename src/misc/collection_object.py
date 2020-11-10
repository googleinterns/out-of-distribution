from typing import Any


class DictObject(dict):
    def __getattr__(self, key: Any) -> Any:
        if key not in self:
            raise AttributeError(key)

        if isinstance(self[key], list):
            return ListObject(self[key])
        elif isinstance(self[key], dict):
            return DictObject(self[key])
        else:
            return self[key]


class ListObject(list):
    def __getitem__(self, index: int) -> Any:
        if not 0 <= index < len(self):
            raise IndexError(index)

        if isinstance(self[index], list):
            return ListObject(self[index])
        elif isinstance(self[index], dict):
            return DictObject(self[index])
        else:
            return self[index]
