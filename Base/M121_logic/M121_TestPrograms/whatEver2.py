import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def main():
    tstart = 0.0
    tend = 0.5
    N = 500 # number of time steps
    t = np.linspace(tstart , tend, N)
    J = 0.0607
    Ra = 0.65
    La = 0.0008
    K = 1.25
    b = 0.02

    # System definition
    A = np.array([[-Ra/La, -K/La], [K/J, -b/J]])
    B = np.array([[1/La], [0.]])
    C = np.array([[0., 1.]])
    D = np.array([[0.]])
    G = signal.StateSpace(A, B, C, D)

    # Perform integration
    u = np.concatenate((50*np.ones(100), 90*np.ones(200)))
    u = np.concatenate((u, -30*np.ones(200)))
    tout, y, x = signal.lsim(G, u, t)


    # Plot the results
    plt.subplot(211)
    plt.plot(t,u,'b')
    plt.ylabel('u [V]')
    plt.grid()
    plt.subplot(212)
    plt.plot(t,y,'r')
    plt.xlabel('t [s]')
    plt.ylabel('y [rad/s]')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()