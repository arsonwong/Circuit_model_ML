from Circuit_Model_ML.circuit_network import *

grid_circuit = GridCircuit(rows=4,cols=4,is_linear=True,has_current_source=False) # makes a random grid circuit each time
_, RMS = grid_circuit.solve(convergence_RMS=1e-8)
grid_circuit.draw()
print(RMS)
    