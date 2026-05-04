import subprocess
import sys
from pathlib import Path

def run():
    snakefile = Path(__file__).parent / "Snakefile"
    cmd = [
        "snakemake",
        "--snakefile", str(snakefile),
        "--cores", "1",
    ] + sys.argv[1:]
    subprocess.run(cmd, check=True)