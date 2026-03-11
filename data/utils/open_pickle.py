import pickle
from pprint import pprint
from itertools import islice
with open('data/childes_by_stage/mature/mature.derivations.p', 'rb') as f:
    data = pickle.load(f)

der = data['91-0699']
detail_rule = der[2]
pprint(der)
#pprint(data, width=80)