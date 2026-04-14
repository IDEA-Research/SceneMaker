import numpy as np

a = np.load('00a1a602456f4eb188b522d7ef19e81b.npz')

np.savez('00a1a602456f4eb188b522d7ef19e81b_.npz',
            surface=a['surface'],
            sharp_surface=a['sharp_surface'],
            bounds=a['bounds']
        )