"""Shared action and camera layout for rollouts, cached features, critics and runtime."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name('defaults.json')


@dataclass(frozen=True)
class QConfig:
    max_steps: int
    horizon: int
    action_dim: int
    camera_keys: tuple[str, ...]
    platform_camera_keys: tuple[str, ...]
    policy_camera_keys: tuple[str, ...]

    def __post_init__(self):
        for field in ('max_steps','horizon','action_dim'):
            value = getattr(self,field)
            if type(value) is not int or value < 1:
                raise ValueError(f'{field} must be a positive integer')
        for field in ('camera_keys','platform_camera_keys','policy_camera_keys'):
            keys = getattr(self,field)
            if (not isinstance(keys,(list,tuple)) or not keys
                    or any(not isinstance(key,str) or not key.strip() for key in keys)
                    or len(set(keys)) != len(keys)):
                raise ValueError(f'{field} must be a nonempty sequence of distinct camera names')
            object.__setattr__(self,field,tuple(keys))
        if not len(self.camera_keys) == len(self.platform_camera_keys) == len(self.policy_camera_keys):
            raise ValueError('archive, platform and policy camera lists must align by view')

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def to_dict(self):
        values = asdict(self)
        for field in ('camera_keys','platform_camera_keys','policy_camera_keys'):
            values[field] = list(values[field])
        return values


def load_config(path=None):
    defaults = json.loads(DEFAULT_PATH.read_text())
    if path is not None:
        defaults.update(json.loads(Path(path).read_text()))
    return QConfig.from_dict(defaults)


DEFAULT_CONFIG = load_config()


def resolve_model_config(checkpoint, requested=None):
    """Legacy state dicts accept an explicit layout, otherwise use the package defaults."""
    if 'q_config' not in checkpoint:
        return requested or DEFAULT_CONFIG
    saved = QConfig.from_dict(checkpoint['q_config'])
    if requested is not None and saved != requested:
        raise ValueError('requested Q configuration differs from the saved checkpoint')
    return saved
