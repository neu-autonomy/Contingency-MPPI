import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pydecomp as pdc

class OccupGrid:
    def __init__(self, boundary, resolution):
        self.resolution = resolution
        self.boundary = boundary
        self.occup_grid = None

    def find_occupied(self, pt):
        # x_range = self.boundary[0][1] - self.boundary[0][0]
        # y_range = self.boundary[1][1] - self.boundary[1][0]

        x_ind = int(np.round((pt[0] - self.boundary[0][0])/self.resolution))
        y_ind = int((np.round(pt[1] - self.boundary[1][0])/self.resolution))
        if x_ind < 0 or x_ind >= self.occup_grid.shape[0] or y_ind < 0 or y_ind >= self.occup_grid.shape[1]:
            return True
        if self.occup_grid[x_ind, y_ind]:
            return True

        return False

    def find_all_occupied(self, obstacles):
        '''
            boundary: [[x_min, x_max], [y_min, y_max]]
        '''
        x = np.arange(self.boundary[0][0], self.boundary[0][1], self.resolution)
        y = np.arange(self.boundary[1][0], self.boundary[1][1], self.resolution)
        xx, yy = np.meshgrid(x,y)
        coords = np.hstack([xx.reshape(-1,1),yy.reshape(-1,1)])

        occupied = []
        # np.any(np.linalg.norm(obstacles[:,:2]-point, axis=1) < np.sqrt(obstacles[:,2]))
        # breakpoint()
        for point in coords:
            if np.any(np.linalg.norm(obstacles[:,:2]-point, axis=1) < np.sqrt(obstacles[:,2])):
                # breakpoint()
                occupied.append(point)
        return np.array(occupied)

    def find_occupancy_grid(self, obstacles, buffer=0.5):
        xs = np.arange(self.boundary[0][0], self.boundary[0][1], self.resolution)
        ys = np.arange(self.boundary[1][0], self.boundary[1][1], self.resolution)
        x_range = self.boundary[0][1] - self.boundary[0][0]
        y_range = self.boundary[1][1] - self.boundary[1][0]
        occup_grid = np.zeros((int(x_range/self.resolution), int(y_range/self.resolution)))
        for i,x in  enumerate(xs):
            for j,y in enumerate(ys):
                if np.any(np.linalg.norm(obstacles[:,:2]-np.array([x,y]), axis=1) < np.sqrt(obstacles[:,2])+buffer):
                    occup_grid[i,j] = 100
        self.occup_grid = occup_grid
        return occup_grid

def main():
    obs = jnp.array([[-12.5, -5,2.5],
                    [-15,0,1],
                    [-5, 2, 2],
                    [-20,5, 4]
                    ])
    boundary = [[-30,0], [-10, 10]]
    grid = OccupGrid(boundary, 0.5)
    occupied = grid.find_all_occupied(obs)
    path = np.array([[-5.0,4.0],
                    [-10.0,2.0],
                    [-15.0,-2.0],
                    [-20.0,0.0]])
    box = np.array([[2,2]])
    A, b = pdc.convex_decomposition_2D(occupied, path, box)
    ax = pdc.visualize_environment(Al=A, bl=b, p =path, planar=True)
    ax.plot(path[:,0], path[:,1], "k-o")

    for circ in obs:
        circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        plt.gca().add_artist(circle)
    plt.plot(occupied[:,0], occupied[:,1], '*')

    plt.figure()
    occupancy = grid.find_occupancy_grid(obs)
    plt.imshow(occupancy.T)
    # plt.xlim(boundary[0])
    # plt.ylim(boundary[1])
    plt.show()


if __name__ == "__main__":
    main()
