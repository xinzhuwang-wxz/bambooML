class Registry:
    """注册器类，用于注册和获取对象"""
    
    def __init__(self):
        self._items = {}

    def register(self, name: str, obj):
        self._items[name] = obj

    def get(self, name: str):
        return self._items[name]

    def has(self, name: str) -> bool:
        return name in self._items

model_registry = Registry()
task_registry = Registry()
datasource_registry = Registry()
