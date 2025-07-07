'''
Ideas:
- General Circuit Model, so with network connections, not just parallel or series
- X - Recast models from PV circuit model into this format [Sat]
- Share PV Circuit Model Elements [Sat]
- Be able to generate LT Spice Netlist or read from it
- X - CSR format [Sat]
- X - Use pyG aggregation and pytorch autograd to construct jacobian [Sat]

- X - Steal the newton solver from other repo [Sat]
- X - cast into graph NN, dictate the continuity [Sun]
- X - set up training program and stuff [Mon]
- X - put up onto the cloud [Tue]

- 
- Next, how to learn the inverse?  Say the fitting of the tandem cell let's say
- That's a problem where element parameters are adjusted until fit
- What does training look like?
- Randomly assign element parameters within the plausible space
- Simulate the outcome experimental results
- Learn inverse pattern, that's certainly one way
- But what kind of structure learns this out-input?  Same kind of GNN?
- Forward is (Node,Edge features) --> Encoder --> GNN processor --> Decoder --> (Node voltages, maybe edge current?)
- Processing inverse involves (Experimental results that don't tell you anything about node voltages or edge currents) --> (Node,Edge features)
- How does info flow backwards like this?
- With images that didn't seem too hard as the input output formats are the same
- We must express experiment data as known (Node, Edge featrues) that's incomplete (not all known)
- this way we can again use the GNN structure

- Next, 

'''

