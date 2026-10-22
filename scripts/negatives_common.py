"""
Shared helpers for building negative examples (fetch, generate, verify).
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SERPAPI_RAW_DIR = DATA_DIR / "serpapi_raw"
REAL_HEADLINES_PATH = DATA_DIR / "real_headlines.csv"
SYNTHETIC_HEADLINES_PATH = DATA_DIR / "synthetic_headlines.csv"

# Definition of a positive, shared by the generator and the judge.
# Mirrors the Pacific Institute Water Conflict Chronology labels in positives.csv.
WATER_CONFLICT_DEFINITION = """\
A headline is a WATER CONFLICT if it reports violence, threats of violence, or \
violent/disruptive confrontation where water plays one of these roles:
- Trigger: water (access, shortage, allocation, pricing, pollution, dams, wells) is a cause \
of violence or violent protest, e.g. "Two injured in protest over water outages in eMalahleni".
- Weapon: water or water systems are used as a weapon or threat, e.g. flooding land to repel \
troops, poisoning wells, threatening to contaminate a water supply.
- Casualty: water resources or infrastructure (dams, pipes, wells, treatment plants, pumping \
stations, canals) are targets or collateral damage of violence, sabotage or vandalism, \
e.g. "Shelling damages water pipes in Belgorod".
Protests about water, attacks that damage water infrastructure, disputes over water that \
involve force, violence against people over their water work or activism (e.g. killing a dam \
opponent, attacking a well driller), and conflicts named after a water body all count as \
water conflict."""

WATER_TERMS = re.compile(
    r"\b("
    r"water\w*|wastewater|dams?|river\w*|lakes?|drought\w*|flood\w*|irrigat\w*|"
    r"canals?|reservoirs?|aquifers?|groundwater|desalinat\w*|hydro\w*|"
    r"rain\w*|monsoon\w*|sewage|sanitation|wells|boreholes?|glaciers?|"
    r"fish\w*|nile|euphrates|tigris|indus|mekong|ganges|brahmaputra|zambezi|"
    r"thirst\w*|cholera|tsunami|cyclone|hurricane|typhoon|storm\w*"
    r")\b",
    re.IGNORECASE,
)


def load_env_key(name: str) -> str:
    """Read a key from the environment, falling back to the project .env file."""
    key = os.environ.get(name)
    env_path = PROJECT_ROOT / ".env"
    if not key and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit(f"{name} not found in environment or .env")
    return key


def normalize(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower())
