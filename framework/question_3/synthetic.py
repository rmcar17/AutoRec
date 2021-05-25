import numpy as np
from scipy.stats import multinomial, uniform
import matplotlib.pyplot as plt

def rotate(points, angle):
    rot_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]]
        )
    
    return points @ rot_matrix


def transform_higher(dataset, seed=1337):
    
    np.random.seed(seed)
    
    inv_weights = np.random.randn(2, 5)
    
    noise = np.random.randn(dataset.shape[0], 5) * 0.3
    
    bias = np.random.randn(1, 2)
    
    inv_sigmoid = lambda x: np.log(x / (1-x))
    
    new_dataset = (inv_sigmoid(dataset) - bias) @ inv_weights + noise

    return new_dataset


def generate_synthetic(npoints, seed=1337):
    point_split = multinomial.rvs(npoints, [0.25]*4, random_state=seed)
    
    ex1 = uniform.rvs(loc=-0.7, scale=2*0.7, size=(point_split[0], 2), random_state=seed+1)
    ex1[:, 0] *= 0.05
    ex1 = rotate(ex1, np.pi/4)

    ex2 = uniform.rvs(loc=-0.7, scale=2*0.7, size=(point_split[1], 2), random_state=seed+2)
    ex2[:, 0] *= 0.05
    ex2 = rotate(ex2, -np.pi/4)

    de1 = uniform.rvs(loc=-0.7, scale=2*0.7, size=(point_split[2], 2), random_state=seed+3)
    de1[:, 0] *= 0.05

    de2 = np.zeros((point_split[3], 2))
    de2_y = uniform.rvs(loc=-0.7, scale=2*0.7, size=(point_split[3]), random_state=seed+4)
    de2_r = uniform.rvs(loc=0.8, scale=0.2, size =(point_split[3]), random_state=seed+5)

    de2[:, 0] = np.sqrt(de2_r - de2_y**2) - 0.5
    de2[:, 1] = de2_y

    ex1 = rotate(ex1, np.pi/2)
    ex2 = rotate(ex2, np.pi/2)
    de1 = rotate(de1, np.pi/2)
    de2 = rotate(de2, np.pi/2)

    de1[:, 1] += 0.7
    de2[:, 1] += 0.7

    ex1[:, 1] += 1.3
    ex2[:, 1] += 1.3
    
    ex1[:, 0] += 1.0
    ex2[:, 0] += 1.0
    
    de1[:, 0] += 1.0
    de2[:, 0] += 1.0
    
    ex1[:, 1] /= 2 
    ex2[:, 1] /= 2

    de1[:, 1] /= 2
    de2[:, 1] /= 2
    
    ex1[:, 0] /= 2 
    ex2[:, 0] /= 2
    
    de1[:, 0] /= 2
    de2[:, 0] /= 2
    
    
    raw_synthetic_dataset = np.vstack([ex1, ex2, de1, de2])
    
    categories = []
    for i in range(len(point_split)):
        v = 0 if i < 2 else 1
        categories += point_split[i] * [v]
    categories = np.array(categories)

    plt.scatter(raw_synthetic_dataset[:, 0], raw_synthetic_dataset[:, 1])
    plt.show()
    
    return transform_higher(raw_synthetic_dataset, seed=seed), categories