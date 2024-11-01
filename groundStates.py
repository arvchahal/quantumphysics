import jax.numpy as np
from jax import grad, vmap
from jax.scipy.linalg import cho_factor, h

# Define parameters for the Gaussian function
alpha1 = 1.0
alpha2 = 1.0
alpha = 0.5
planks = h
alp_arr = [alpha,alpha1,alpha2]
dim = 3
# Construct the A matrix (2x2 for 2D)
A = np.array([[alpha1 + alpha, -alpha],
              [-alpha, alpha2 + alpha]])

# Define the s vector (mean position terms) for Φ_i and Φ_j
r1 = np.array([1.0, 2.0])   #place holders for 2d space to trouble shoot
r2 = np.array([4.0, 5.0])  
r = np.array([[r1],[r2]])
# Define 2D mean vectors s1 and s2 for each Gaussian
s1 = np.array([0.5, 0.5]) 
s2 = np.array([1.5, 1.5])  
#s_arr =[] place holder
"""
Need to have laser field wavelength to get k1 and k2
k1 = 2*pi/wavelength

"""
#temp k1 and k2 values
lambda1 = 1.0  # Wavelength along r1 direction
lambda2 = 2.0  # Wavelength along r2 direction

# Define wave vectors based on wavelengths
k1 = 2 * np.pi / lambda1
k2 = 2 * np.pi / lambda2

s = np.array([[alpha1*s1+1j*k1],[alpha2*s2+1j*k2]])
# Define a constant term c if needed for the exponential
c = 0.0

def gaussian_wavefunction(r, s, alpha1, alpha2, alpha, k1, k2):
    r1, r2 = r
    s1, s2 = s

    real_part = -alpha1 * (r1 - s1)**2 - alpha2 * (r2 - s2)**2 - alpha * (r1 - r2)**2
   
    imag_part = k1 * r1 + k2 * r2

    return np.exp(real_part + 1j * imag_part)

def overlap(B, s, s_prime, c, c_prime, d=3):
    v = s+s_prime
    term1 = ((2*np.pi)**2 /( np.linalg.det(B)))**(d/2)
    term2 = np.exp(-0.5 * v.T @ B @ v + c + c_prime)
    return term1 * term2

def construct_A_matrix(dim):
    pass

def mv_gaussian(A,s,c,r):
    return np.exp(-.5 * (r.T @A@r) + r.T@s + c)

def kinetic(s,s_prime,d,A,B,O, A_prime):
    term1 = planks**2 *.5
    y = A_prime@np.linalg.inv(B)@s - A @ np.linalg.inv(B) @s_prime
    term2 = d * np.trace(A@np.linalg.inv(B)@A_prime) -(y.T@y)
    return term1*term2*O


def potential():
    pass

def Hamiltonian(s,s_prime,d,A,B,O, A_prime):
    # Evaluate the wavefunction
    return kinetic(s,s_prime,d,A,B,O, A_prime) + potential()

# Define matrix element computation for ⟨Φᵢ | Ĥ | Φⱼ⟩
def matrixElement_Phi_i_H_Phi_j():
    pass

# Example calculation
print("Matrix Element ⟨Φᵢ | Ĥ | Φⱼ⟩ = ", matrixElement_Phi_i_H_Phi_j())

