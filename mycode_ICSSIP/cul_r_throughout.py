import numpy as np

# -----------------------------
# 1.  steering vector of UPA
# -----------------------------
def cul_through_out(input ):
    [theta_true, phi_true, theta_bf, phi_bf] = input
    def steering_vector(theta, phi, M=32, N=32, d=0.5, wavelength=1.0):
        """
        theta: elevation angle in radians
        phi:   azimuth angle in radians
        M, N:  UPA size (MxN)
        d:     element spacing (normalized by wavelength)
        """
        k = 2 * np.pi / wavelength

        # index grid
        m_idx = np.arange(M)
        n_idx = np.arange(N)
        m, n = np.meshgrid(m_idx, n_idx, indexing='ij')

        # phase term
        phase = k * d * (m * np.sin(theta) * np.cos(phi) + n * np.sin(theta) * np.sin(phi))
        a = np.exp(1j * phase)
        return a.reshape(-1) / np.sqrt(M * N)

    # -----------------------------
    # 2.  compute effective gain
    # -----------------------------
    def effective_gain(theta_true, phi_true, theta_bf, phi_bf, M=32, N=32):
        a_true = steering_vector(theta_true, phi_true, M, N)
        a_bf = steering_vector(theta_bf, phi_bf, M, N)
        G = np.abs(np.vdot(a_true, a_bf)) ** 2  # |a_true^H a_bf|^2
        return G

    # -----------------------------
    # 3.  throughput
    # -----------------------------
    def throughput(G, B=1e9, SNR_dB=10):
        """
        G: gain
        B: bandwidth (Hz)
        SNR_dB: transmit SNR
        """
        gamma = 10 ** (SNR_dB / 10)
        R = B * np.log2(1 + gamma * G)
        return R

    # -----------------------------
    # Example: fill angles here
    # -----------------------------
    # angles in radians
    #theta_true = np.deg2rad(10)  # true elevation angle
    #phi_true = np.deg2rad(20)  # true azimuth angle

    #theta_bf = np.deg2rad(12)  # used in beamforming
    #phi_bf = np.deg2rad(22)

    # compute gain and throughput
    G = effective_gain(theta_true, phi_true, theta_bf, phi_bf)
    R = throughput(G, B=1, SNR_dB=10)

    print("Effective gain G =", G)
    print("Throughput R =", R, "bps/Hz")
    return R

cul_through_out([0.6861300468444824,
 0.12870284914970398,
 0.7062091827392578,
 0.20396874845027924])