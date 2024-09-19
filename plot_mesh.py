import viscid
from viscid.plot import vpyplot as vlt
import os

cwd = os.getcwd()
reader = viscid.load_file(cwd+'/outputs/flat_solutions_coupled.xdmf')
