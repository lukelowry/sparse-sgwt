
import sgwt 
from sgwt.laplib import IMPEDANCE_TEXAS as graph

L = graph.laplacian()

chol = sgwt.CholWrapper(L)
chol.sym_factor()
chol.num_factor(2)

b = sgwt.impulse(L,n=100)
x = chol.solve(b)

print(x)