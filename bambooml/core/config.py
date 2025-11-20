import yaml
import json
from pathlib import Path

def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)  # 安全加载

def dump_yaml(obj: dict, path: str | Path) -> None:  # str|Path means 可以是字符串或路径对象
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False)  # 安全转储, 不排序键；

def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)

def dump_json(obj: dict, path: str | Path) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
