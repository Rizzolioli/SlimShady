from data_creation import create_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time



for data_function in [
    # 'rastrigin', 'sphere', 'rosenbrock', 'ackley',
    #              'alpine1',
                 'alpine2' ,
                 'michalewicz' ]:

    for (input_scale, output_scale) in [((0,1), (0,1)),
                                        ((0,1), (0,10)), ((0,1), (0,100)), ((0,1), (0,1000)),
                                        ((0, 10), (0, 1)), ((0, 100), (0, 1)),((0, 1000), (0, 1)),
                                        ((0,10), (0,10)), ((0,100), (0,100)), ((0,1000), (0,1000))
                                        ]:


        X, z = create_dataset(10000, 2, scale_inputs = input_scale, scale_output = output_scale,
                              function = data_function, seed = 1)

        x = X[:,0]
        y = X[:, 1]

        # plt.close()

        # Create 2D grid coordinates for surface plot
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate to get z values on grid
        from scipy.interpolate import griddata

        zi = griddata((x, y), z, (xi, yi), method='cubic')

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xi, yi, zi, cmap=cm.viridis_r)
        ax.view_init(elev=45, azim=30)

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')

        plt.title(f'{data_function}  INP:{input_scale}  OUT:{output_scale}')


        plt.savefig(f'fitness_landscapes/{data_function}_INP{input_scale[1]}_OUT{output_scale[1]}.png')
        plt.show(block = False)
        # time.sleep(3)
        plt.close()