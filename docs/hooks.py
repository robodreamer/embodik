import tomllib
import pathlib


def on_config(config):
    pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    config.extra["version"] = data["project"]["version"]
    return config
