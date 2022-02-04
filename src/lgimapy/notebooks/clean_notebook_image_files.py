import os
from lgimapy.utils import root

# %%

path = root('src/lgimapy/notebooks')
for fid in path.glob('**/*.png'):
    os.remove(fid)
