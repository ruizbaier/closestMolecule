import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
import matplotlib
from scipy.optimize import curve_fit
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def error_func(x, a, b, c):
    return a*np.square(x) + b*x + c


def flux_func(x, b):
    return b*x


def required_rate(sigma, conc, gamma):
    return 4*np.pi*2*sigma*(conc/(conc + gamma))


def adjustment_sigma(sigma, rate_change, a, b):
    return (1/(2*a))*(2*sigma*a + b - np.sqrt(np.square(2*sigma*a + b) - 4*a*rate_change))


plt.rcParams.update({"text.usetex":True})
rc('text', usetex=True)
rc('font', family='serif', weight='bold')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
size = 24
params = {'axes.labelsize': size, 'axes.titlesize': size, 'legend.fontsize': size,
          'xtick.labelsize': size, 'ytick.labelsize': size}
matplotlib.rcParams.update(params)
data = np.load('concentration_test/concentration_test.npy')
#adjusted_data = np.load('boundary_condition_test/flat_test_first_order_boundary.npy')
#data = data[data[:, 1].argsort()]
#data2 = np.load('gamma_test2.npy')
#data = np.concatenate([data, data2])
# Each result contains: [sigma, gamma, concentration, total_r1_flux, total_r2_flux, total_flux, total_approx_flux].
# Plot the error as a function of sigma
r1_vals = np.array([1,2,4,8,16])
sigmas = data[:, 0]
gammas = data[:, 1]
concentrations = data[:, 2]
total_flux = data[:, 5]
total_flux_error = (data[:, 5] - data[:, 6])
r1_flux_error = (data[:, 3] - data[:, 6])
total_approx = data[:, 6]

'''
adjusted_total_flux_error = adjusted_data[:, 5] - adjusted_data[:, 6]
figr = plt.figure(dpi=80)
rate_ax = figr.subplots()
# Fit function to increase in flux
popt, pcov = curve_fit(flux_func, sigmas, total_flux)
print(popt)
print(pcov)
rate_ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
sigmas_range = np.arange(0, max(sigmas)+0.01, 0.01)
expected_rate = required_rate(sigmas, concentrations[0], gammas[0])
rate_ax.scatter(r1_vals, total_flux_error[:-1]/total_approx[:-1], s=125, label='Outer solution.')
print(total_flux_error/total_approx)
rate_ax.scatter(r1_vals, adjusted_total_flux_error[:-1]/total_approx[:-1], label='Leading order solution.', s=125)
print(adjusted_total_flux_error/total_approx)
rate_ax.set_ylabel(r'\bf{Relative error in flux over} $\boldsymbol{r_1 = 0.1}$')
rate_ax.set_xlabel(r'$\boldsymbol{r_1^{\text{\bf{max}}}}$')
rate_ax.set_xticks([0, 1, 2,4,8,16])
rate_ax.legend()
'''

fig, error_ax = plt.subplots()
fig2, r1_flux_ax = plt.subplots()
fig3, r2_flux_ax = plt.subplots()

#*********************** sigma plotting ****************************
'''
error_ax.plot(sigmas, total_flux_error, label=r'Total correction', linewidth=2, marker='o', markersize=10,
              color='black', linestyle='--')
error_ax.plot(sigmas, r1_flux_error, label=r'$r_1$ component', linewidth=2, marker='o', markersize=10,
              color='r', linestyle='--')
error_ax.plot(sigmas, data[:, 4], label=r'$r_2$ component', linewidth=2, marker='o', markersize=10,
              color='b', linestyle='--')
error_ax.ticklabel_format(axis='y')
error_ax.set_ylabel(r'$\boldsymbol{O(\sigma_{\text{\bf{max}}}^2) correction}$')
error_ax.set_xlabel(r'$\boldsymbol{\sigma}$')
#error_ax.set_ylim([0, 0.017])
error_ax.legend()
#ax.plot(gammas, error_func(gammas, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
r1_flux_ax.plot(sigmas, data[:, 3], label=r'$r_1$ flux', linewidth=2, marker='o', markersize=10, color='black',
                linestyle='--')
r1_flux_ax.ticklabel_format(axis='y')
r1_flux_ax.set_ylabel(r'$\boldsymbol{r_1}$ \bf{flux}')
r1_flux_ax.set_xlabel(r'$\boldsymbol{\sigma}$')
#r1_flux_ax.set_ylim([0, 1])

r2_flux_ax.plot(sigmas, data[:, 4], label=r'$r_2$ flux', linewidth=2, marker='o', markersize=10, color='b',
                linestyle='--')
r2_flux_ax.ticklabel_format(axis='y')
r2_flux_ax.set_ylabel(r'$\boldsymbol{r_2}$ \bf{flux}')
r2_flux_ax.set_xlabel(r'$\boldsymbol{\sigma}$')
'''
'''
# ************* gamma plotting **************************
error_ax.plot(gammas, total_flux_error, label='Total correction', linewidth=2, marker='o', markersize=10,
              color='black', linestyle='--')
error_ax.plot(gammas, r1_flux_error, label=r'$r_1$ component', linewidth=2, marker='o', markersize=10,
              color='r', linestyle='--')
error_ax.plot(gammas, data[:, 4], label=r'$r_2$ component', linewidth=2, marker='o', markersize=10,
              color='b', linestyle='--')
error_ax.ticklabel_format(axis='y')
error_ax.set_ylabel(r'$\boldsymbol{O(\sigma_{\text{\bf{max}}}^2)}$ \bf{correction}')
error_ax.set_xlabel(r'$\boldsymbol{\Gamma}$')
error_ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3])
#error_ax.set_ylim([0, 0.017])
error_ax.legend()
#ax.plot(gammas, error_func(gammas, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
r1_flux_ax.plot(gammas, data[:, 3], label=r'$r_1$ flux', linewidth=2, marker='o', markersize=10, color='black',
                linestyle='--')
r1_flux_ax.ticklabel_format(axis='y')
r1_flux_ax.set_ylabel(r'$\boldsymbol{r_1}$ \bf{flux}')
r1_flux_ax.set_xlabel(r'$\boldsymbol{\Gamma}$')
r1_flux_ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3])
#r1_flux_ax.set_ylim([0, 1])

r2_flux_ax.plot(gammas, data[:, 4], label=r'$r_2$ flux error', linewidth=2, marker='o', markersize=10, color='b',
                linestyle='--')
r2_flux_ax.ticklabel_format(axis='y')
r2_flux_ax.set_ylabel(r'$\boldsymbol{r_2}$ \bf{flux}')
r2_flux_ax.set_xlabel(r'$\boldsymbol{\Gamma}$')
r2_flux_ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3])
'''
# *************************** concentration plotting *************************************
error_ax.plot(concentrations, total_flux_error, label='Total correction', linewidth=2, marker='o', markersize=10,
              color='black', linestyle='--')
error_ax.plot(concentrations, r1_flux_error, label=r'$r_1$ component', linewidth=2, marker='o', markersize=10,
              color='r', linestyle='--')
error_ax.plot(concentrations, data[:, 4], label=r'$r_2$ component', linewidth=2, marker='o', markersize=10,
              color='b', linestyle='--')
error_ax.ticklabel_format(axis='y')
error_ax.set_ylabel(r'$\boldsymbol{O(\sigma_{\text{\bf{max}}}^2)}$ \bf{correction}')
error_ax.set_xlabel(r'$\boldsymbol{c}$')
error_ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16])
#error_ax.set_ylim([0, 0.017])
error_ax.legend()
#ax.plot(gammas, error_func(gammas, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
r1_flux_ax.plot(concentrations, data[:, 3], label=r'$r_1$ flux', linewidth=2, marker='o', markersize=10, color='black',
                linestyle='--')
r1_flux_ax.ticklabel_format(axis='y')
r1_flux_ax.set_ylabel(r'$\boldsymbol{r_1}$ \bf{flux}')
r1_flux_ax.set_xlabel(r'$\boldsymbol{c}$')
r1_flux_ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16])
#r1_flux_ax.set_ylim([0, 1])

r2_flux_ax.plot(concentrations, data[:, 4], label=r'$r_2$ flux error', linewidth=2, marker='o', markersize=10, color='b',
                linestyle='--')
r2_flux_ax.ticklabel_format(axis='y')
r2_flux_ax.set_ylabel(r'$\boldsymbol{r_2}$ \bf{flux}')
r2_flux_ax.set_xlabel(r'$\boldsymbol{c}$')
r2_flux_ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16])
plt.show()
