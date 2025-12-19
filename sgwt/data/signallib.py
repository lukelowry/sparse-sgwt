from importlib_resources import files, as_file

from numpy import load

from dataclasses import dataclass


@dataclass(frozen=True)
class LapID:
    kind: str
    region: str

    def get(self):
        
        # Weight Type
        B = self.kind

        # Name/Region
        N = self.region

        with as_file(files("sgwt").joinpath(f"data/SIGNALS/{N}_{B}.npz")) as sig_path:
            return load(sig_path)


#class LapLib(Enum):
COORD_EASTWEST = LapID("COORDS", "EASTWEST")
COORD_HAWAII = LapID("COORDS", "HAWAII")
COORD_TEXAS = LapID("COORDS", "TEXAS")
COORD_USA = LapID("COORDS", "USA")

