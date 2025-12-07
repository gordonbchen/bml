import json
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path


class CLIParams:
    def cli_override(self):
        """Override params from CLI args."""
        parser = ArgumentParser()
        for k, v in asdict(self).items():
            parser.add_argument(f"--{k}", type=type(v), default=v)
        args = parser.parse_args()

        for k, v in vars(args).items():
            setattr(self, k, v)
        return self

    def to_cli_args(self) -> list[str]:
        """Convert to params to CLI args."""
        args = []
        for k, v in asdict(self).items():
            args.append(f"--{k}={v}")
        return args

    def to_json_file(self, filename: str | Path):
        with open(filename, "w") as f:
            json.dump(asdict(self), f, indent=4)

    def from_json_file(self, filename: str | Path):
        with open(filename, "r") as f:
            d = json.load(f)
        for k, v in d.items():
            setattr(self, k, v)
        return self