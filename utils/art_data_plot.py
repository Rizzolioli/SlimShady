from data_creation import create_dataset
import matplotlib.pyplot as plt



for data_function in ['rastrigin', 'sphere', 'rosenbrock' ]:

    for (input_scale, output_scale) in [((0,1), (0,1)),
                                        # ((0,1), (0,10)), ((0,1), (0,100)), ((0,1), (0,1000)),
                                        # ((0, 10), (0, 1)), ((0, 100), (0, 1)),((0, 1000), (0, 1)),
                                        # ((0,10), (0,10)), ((0,100), (0,100)), ((0,1000), (0,1000))
                                        ]:


        X, y = create_dataset(10000, 2, scale_inputs = input_scale, scale_output = output_scale,
                              function = data_function, seed = 1)

        # plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')



        ax.scatter(X[:, 0], X[:, 1], y)
        # plt.savefig()
        plt.show()