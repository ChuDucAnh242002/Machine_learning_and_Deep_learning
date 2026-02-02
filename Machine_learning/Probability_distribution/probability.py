import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

np.random.seed(66) # to always get the same points
N_h = 1000
N_e = 200
mu_h_gt = 1.1
mu_e_gt = 1.9
sigma_h_gt = 0.3
sigma_e_gt = 0.4
x_h = np.random.normal(mu_h_gt,sigma_h_gt,N_h)
x_e = np.random.normal(mu_e_gt,sigma_e_gt,N_e)
plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobit")
plt.plot(x_e,np.zeros([N_e,1]),'mo', label="elf")
plt.title(f'{N_h} hobit height')
plt.legend()
plt.xlabel('height [m]')
plt.axis([0.0,3.0,-1.1,+1.1])
plt.show()

mu_h_est = np.mean(x_h)
mu_e_est = np.mean(x_e)
sigma_h_est = np.std(x_h)
sigma_e_est = np.std(x_e)
print(f'Avg height hobits {mu_h_est:0.2f} (GT: {mu_h_gt:0.2f})')
print(f'Avg height elves {mu_e_est:0.2f} (GT: {mu_e_gt:0.2f})')
print(f'Height st. deviation hobits {sigma_h_est:0.3f} (GT: {sigma_h_gt:0.3f})')
print(f'Height st. deviation elves {sigma_e_est:0.3f} (GT: {sigma_e_gt:0.3f})')

def gaussian(x, mu, sigma, priori):
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        z[i] = priori*1/np.sqrt(2*np.pi)*1/sigma*np.exp((-1/2*(x[i]-mu)**2)/(2*sigma**2))
    return z

[x, step_size] = np.linspace(0,3.0,70,retstep=True)
lhood_h_est = gaussian(x, mu_h_est, sigma_h_est, 1)
lhood_e_est = gaussian(x, mu_e_est, sigma_e_est, 1)
lhood_h_gt = gaussian(x, mu_h_gt, sigma_h_gt, 1)
lhood_e_gt = gaussian(x, mu_e_gt, sigma_e_gt, 1)
plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobit")
plt.plot(x_e,np.zeros([N_e,1]),'mo', label="elf")
plt.plot(x,lhood_h_gt,'c-', label="hobit (GT)")
plt.plot(x,lhood_e_gt,'m-', label="elf (GT)")
plt.plot(x,lhood_h_est,'c--', label="hobit (est)")
plt.plot(x,lhood_e_est,'m--', label="elf (est)")
plt.legend()
plt.xlabel('pituus [m]')
#plt.axis([0.0,3.0,-1.1,+5])
plt.show()

kern_width = 0.2
[x, step_size] = np.linspace(0, 3.0, 70, retstep=True)
plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobit")
plt.plot(x_e,np.zeros([N_e,1]),'mo', label="elf")
[x_kern, step_size_kern] = np.linspace(0,3.0,11,retstep=True)
x_kern_plot = np.linspace(-1.0,+4.0,201)
for foo_ind, foo_val in enumerate(x_kern):
    foo_kern = gaussian(x_kern_plot, foo_val, kern_width, 1)
    plt.plot(x_kern_plot, foo_kern,'y--',label='kernel')
    break
plt.legend()
plt.xlabel('height [m]')
#plt.axis([0.0,3.0,-1.1,+5])
plt.show()

lhood_h_est_kern = np.zeros(len(x))
for xind, xval in enumerate(x):
    lhood_h_est_kern[xind] = sum(gaussian(x_h,xval,kern_width,1))/len(x_h)
    #lhood_h_est_kern[xind] = sum(stats.norm.pdf(x_h, xval, kernel_width))
plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobit")
plt.plot(x_e,np.zeros([N_e,1]),'mo', label="elf")
plt.plot(x,lhood_h_gt,'c-', label="hobit (GT)")
plt.plot(x,lhood_h_est_kern,'c--', label="hobit (est)")
#plt.plot(x,lhood_e_est,'m--', label="haltija (est)")
plt.legend()
plt.xlabel('height [m]')
#plt.axis([0.0,3.0,-1.1,+5])
plt.show()

kern_width = 0.02
[x, step_size] = np.linspace(0,3.0,70,retstep=True)
plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobit")
plt.plot(x_e,np.zeros([N_e,1]),'mo', label="elf")
[x_kern, step_size_kern] = np.linspace(0,3.0,11,retstep=True)
x_kern_plot = np.linspace(-1.0,+4.0,201)
for foo_ind, foo_val in enumerate(x_kern):
    foo_kern = gaussian(x_kern_plot, foo_val, kern_width, 1)
    plt.plot(x_kern_plot, foo_kern,'y--',label='kerneli')
    break
plt.legend()
plt.xlabel('height [m]')
#plt.axis([0.0,3.0,-1.1,+5])
plt.show()

lhood_h_est_kern = np.zeros(len(x))
for xind, xval in enumerate(x):
    lhood_h_est_kern[xind] = sum(gaussian(x_h,xval,kern_width,1))/len(x_h)
    #lhood_h_est_kern[xind] = sum(stats.norm.pdf(x_h, xval, kernel_width))

plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobit")
plt.plot(x_e,np.zeros([N_e,1]),'mo', label="elf")
plt.plot(x,lhood_h_gt,'c-', label="hobit (GT)")
plt.plot(x,lhood_h_est_kern,'c--', label="hobit (est)")
#plt.plot(x,lhood_e_est,'m--', label="haltija (est)")
plt.legend()
plt.xlabel('height [m]')
#plt.axis([0.0,3.0,-1.1,+5])
plt.show()

kern_width = 0.8
[x, step_size] = np.linspace(0,3.0,70,retstep=True)
plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobit")
plt.plot(x_e,np.zeros([N_e,1]),'mo', label="elf")
[x_kern, step_size_kern] = np.linspace(0,3.0,11,retstep=True)
x_kern_plot = np.linspace(-1.0,+4.0,201)
for foo_ind, foo_val in enumerate(x_kern):
    foo_kern = gaussian(x_kern_plot, foo_val, kern_width, 1)
    plt.plot(x_kern_plot, foo_kern,'y--',label='kernel')
    break
plt.legend()
plt.xlabel('height [m]')
#plt.axis([0.0,3.0,-1.1,+5])
plt.show()

lhood_h_est_kern = np.zeros(len(x))
for xind, xval in enumerate(x):
    lhood_h_est_kern[xind] = sum(gaussian(x_h,xval,kern_width,1))/len(x_h)
    #lhood_h_est_kern[xind] = sum(stats.norm.pdf(x_h, xval, kernel_width))

plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobit")
plt.plot(x_e,np.zeros([N_e,1]),'mo', label="elf")
plt.plot(x,lhood_h_gt,'c-', label="hobit (GT)")
plt.plot(x,lhood_h_est_kern,'c--', label="hobit (est)")
#plt.plot(x,lhood_e_est,'m--', label="haltija (est)")
plt.legend()
plt.xlabel('height [m]')
#plt.axis([0.0,3.0,-1.1,+5])
plt.show()