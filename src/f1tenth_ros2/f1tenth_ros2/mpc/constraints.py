import numpy as np
from track_params import TrackParams

def Boundary(x0, track_width, map, eps=0.05):
    """
    Compute linear boundary constraints for MPC.
    
    Args:
        x0 (array-like): current position (2,)
        track_width (float): total width of the track
        map (str): track identifier to load with TrackParams
        eps (float): offset used to estimate tangent direction
        
    Returns:
        A (2,2 ndarray): constraint matrix
        b (2,1 ndarray): constraint vector
    """
    track = TrackParams(map)
    
    theta = track.xy2theta(x0[0], x0[1])
    
    # Get two nearby points to approximate tangent
    x_forward, y_forward = track.theta2xy(theta + eps)
    x_backward, y_backward = track.theta2xy(theta - eps)
    x_center, y_center = track.theta2xy(theta)
    
    # Tangent vector
    tangent = np.array([x_forward - x_backward, y_forward - y_backward])
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-8:
        raise ValueError("Tangent vector norm too small.")
    tangent /= tangent_norm
    normal = np.array([-tangent[1], tangent[0]])

    half_width = track_width / 2.0

    inner_point = np.array([x_center, y_center]) - half_width * normal
    outer_point = np.array([x_center, y_center]) + half_width * normal

    A = np.vstack([
        normal,      # Outer constraint
        -normal      # Inner constraint (negate normal to get the half-plane inside)
    ])
    b = np.array([
        np.dot(normal, outer_point),
        -np.dot(normal, inner_point)
    ]).reshape(-1, 1)
    
    return A, b

if __name__ == '__main__':
	x0  = [-0.44708959788569586, -0.09416882273686002]
	map = "Shanghai"
	track_width = 2.6
	A, b = Boundary(x0, track_width, map)
	# print(np.dot(A,x0))
