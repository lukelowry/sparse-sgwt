from importlib_resources import files, as_file
from json import load
from dataclasses import dataclass


@dataclass(frozen=True)
class KernID:
    name: str

    def get(self):
        N = self.name

        with as_file(files("sgwt").joinpath(f"data/KERNELS/{N}.json")) as kern_path:
            with open(kern_path) as f:
                return load(f)


#class LapLib(Enum):
MEXICAN_HAT     = KernID("MEXICAN_HAT")
GAUSSIAN_WAV    = KernID("GAUSSIAN_WAV")
MODIFIED_MORLET = KernID("MODIFIED_MORLET")
SHANNON         = KernID("SHANNON")


