import subprocess
import sys

procs = []

for i in range(3):
    proc = subprocess.Popen([sys.executable,'Vesicle_Analysis_Execution.py'])
    procs.append(proc)


for proc in procs:
    proc.wait()
