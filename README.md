## Network Efficiency Coordinates

This repository independently manages/stages a python implementation of the network efficiency coordinates as analytically
developed by Massima Marchiori and used in [*Efficient Behavior of Small-World Networks*][1]. 

The present implementation allows for weighted networks to be considered, though this may result in distortions when normalized. 

```
import numpy as np
import efficiency_coordinates as ec

A = np.array([
[0, 1, 0, 1],
[0, 0, 1, 0.5],
[1, 0, 0, 0],
[0, 0, 0.5, 0],
])

print('Non-normalized:   E_diff: {0}, E_rout: {1}'.format(*ec.network_efficiencies(A, normalize=False)))
print('Normalized:       E_diff: {0}, E_rout: {1}'.format(*ec.network_efficiencies(A, normalize=True)))

>>> Non-normalized:   E_diff: 1.972067414836056, E_rout: 3.3766233766233764
>>> Normalized:       E_diff: 0.0003585764919229447, E_rout: 1.719204113570311
```

<!---- Links: ---->
[1]: https://www.researchgate.net/publication/308881997_Efficient_Behavior_of_Small-World_Networks
