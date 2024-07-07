
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fftpack import fft
from scipy.stats import skew, kurtosis, norm, kstest
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.linalg import expm
import warnings

class QuantumRTNSimulation:
    def __init__(self, tau_high, tau_low, total_time, dt, noise_level_gaussian, noise_level_flicker, hamiltonian, states=2):
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.total_time = total_time
        self.dt = dt
        self.noise_level_gaussian = noise_level_gaussian
        self.noise_level_flicker = noise_level_flicker
        self.hamiltonian = hamiltonian
        self.states = states
        self.time = np.arange(0, total_time, dt)
        self.signal = np.zeros_like(self.time)
        self.noisy_signal = np.zeros_like(self.time)
        self.state_vector = np.array([1, 0])  # Start in the first state
        
    def generate_rtn_signal(self):
        state = 1  # Start in high state
        t_next = np.random.exponential(self.tau_high)
        
        for i, t in enumerate(self.time):
            if t >= t_next:
                state = (state + 1) % self.states  # Switch state
                t_next += np.random.exponential(self.tau_high if state == 1 else self.tau_low)
            self.signal[i] = state
        
    def add_noise(self):
        # Add Gaussian noise
        gaussian_noise = np.random.normal(0, self.noise_level_gaussian, size=self.signal.shape)
        # Add 1/f noise
        flicker_noise = self.noise_level_flicker * np.cumsum(np.random.normal(size=self.signal.shape))
        self.noisy_signal = self.signal + gaussian_noise + flicker_noise
    
    def quantum_time_evolution(self):
        U = expm(-1j * self.hamiltonian * self.dt)  # Time evolution operator
        states_over_time = []

        for t in self.time:
            self.state_vector = U @ self.state_vector
            states_over_time.append(np.abs(self.state_vector[0])**2)  # Probability of being in the first state
        
        self.quantum_signal = np.array(states_over_time)
    
    def plot_signals(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.time, self.noisy_signal, label='Noisy RTN Signal')
        plt.plot(self.time, self.quantum_signal, label='Quantum State Probability')
        plt.xlabel('Time')
        plt.ylabel('Signal / Probability')
        plt.title('RTN Signal and Quantum State Probability')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_psd(self):
        frequencies, psd = welch(self.noisy_signal, fs=1/self.dt, nperseg=1024)
        plt.figure(figsize=(12, 6))
        plt.loglog(frequencies, psd, label='PSD')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V^2/Hz]')
        plt.title('Power Spectral Density of RTN')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_fft(self):
        fft_values = fft(self.noisy_signal)
        fft_freq = np.fft.fftfreq(len(fft_values), self.dt)
        plt.figure(figsize=(12, 6))
        plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_values)[:len(fft_values)//2], label='FFT')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title('Fourier Transform of RTN Signal')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_autocorrelation(self):
        plt.figure(figsize=(12, 6))
        plot_acf(self.noisy_signal, lags=50, ax=plt.gca())
        plt.title('Autocorrelation of RTN Signal')
        plt.xlabel('Lags')
        plt.ylabel('Autocorrelation')
        plt.grid(True)
        plt.show()

    def plot_partial_autocorrelation(self):
        plt.figure(figsize=(12, 6))
        plot_pacf(self.noisy_signal, lags=50, method='ywm', ax=plt.gca())
        plt.title('Partial Autocorrelation of RTN Signal')
        plt.xlabel('Lags')
        plt.ylabel('Partial Autocorrelation')
        plt.grid(True)
        plt.show()
    
    def plot_histogram(self):
        plt.figure(figsize=(12, 6))
        plt.hist(self.noisy_signal, bins=50, density=True, alpha=0.6, color='g', label='Signal Histogram')
        mu, std = norm.fit(self.noisy_signal)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, label='Normal fit')
        plt.xlabel('Signal Value')
        plt.ylabel('Probability Density')
        plt.title('Histogram of RTN Signal')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def locate_charge_traps(self):
        transitions = np.where(np.diff(self.signal) != 0)[0]
        trap_times = self.time[transitions]

        if len(trap_times) > 1:
            try:
                # Manual clustering implementation
                mid_point = (trap_times.min() + trap_times.max()) / 2
                cluster_1 = trap_times[trap_times < mid_point]
                cluster_2 = trap_times[trap_times >= mid_point]
                trap_clusters = np.array([cluster_1.mean(), cluster_2.mean()])
            except AttributeError:
                warnings.warn("Threadpoolctl issue encountered. Returning trap times without clustering.")
                trap_clusters = trap_times.reshape(-1, 1)
        else:
            trap_clusters = trap_times.reshape(-1, 1)

        return trap_times, trap_clusters

    def plot_traps(self, trap_times, trap_clusters):
        plt.figure(figsize=(12, 6))
        plt.plot(self.time, self.noisy_signal, label='Noisy RTN Signal')
        plt.scatter(trap_times, self.noisy_signal[np.searchsorted(self.time, trap_times)], color='red', label='Charge Traps', zorder=5)
        plt.scatter(trap_clusters, np.zeros_like(trap_clusters), color='blue', label='Trap Clusters', zorder=5)
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.title('RTN Signal with Charge Traps')
        plt.legend()
        plt.grid(True)
        plt.show()

    def statistical_analysis(self):
        mean_value = np.mean(self.noisy_signal)
        variance_value = np.var(self.noisy_signal)
        skewness_value = skew(self.noisy_signal)
        kurtosis_value = kurtosis(self.noisy_signal)
        print(f'Mean of the signal: {mean_value}')
        print(f'Variance of the signal: {variance_value}')
        print(f'Skewness of the signal: {skewness_value}')
        print(f'Kurtosis of the signal: {kurtosis_value}')

        d_value, p_value = kstest(self.noisy_signal, 'norm', args=(mean_value, np.sqrt(variance_value)))
        print(f'Kolmogorov-Smirnov test D-value: {d_value}, p-value: {p_value}')

        return {
            'mean': mean_value,
            'variance': variance_value,
            'skewness': skewness_value,
            'kurtosis': kurtosis_value,
            'ks_test_d': d_value,
            'ks_test_p': p_value
        }

def main():
    tau_high = 10
    tau_low = 5
    total_time = 1000
    dt = 0.1
    noise_level_gaussian = 0.5
    noise_level_flicker = 0.1
    hamiltonian = np.array([[0, -1j], [1j, 0]])
    
    quantum_rtn_sim = QuantumRTNSimulation(tau_high, tau_low, total_time, dt, noise_level_gaussian, noise_level_flicker, hamiltonian)
    
    quantum_rtn_sim.generate_rtn_signal()
    quantum_rtn_sim.add_noise()
    quantum_rtn_sim.quantum_time_evolution()
    quantum_rtn_sim.plot_signals()
    quantum_rtn_sim.plot_psd()
    quantum_rtn_sim.plot_fft()
    quantum_rtn_sim.plot_autocorrelation()
    quantum_rtn_sim.plot_partial_autocorrelation()
    quantum_rtn_sim.plot_histogram()
    trap_times, trap_clusters = quantum_rtn_sim.locate_charge_traps()
    quantum_rtn_sim.plot_traps(trap_times, trap_clusters)
    stats = quantum_rtn_sim.statistical_analysis()
    print(f'Statistical Analysis: {stats}')

if __name__ == "__main__":
    main()
