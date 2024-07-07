from quantum_rtn_simulation import QuantumRTNSimulation
import numpy as np

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
