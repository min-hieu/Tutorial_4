import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from PIL import Image

def step(x, dt=0.001, mu=1.7, sigma=1.2):
    z = np.random.randn(*x.shape)
    dx = -mu*x*dt + sigma*np.sqrt(dt)*z
    return x + dx

def step_img(x, dt=0.001, mu=120, sigma=1):
    z = np.random.randn(*x.shape)
    dx = -mu*x*dt + sigma*np.sqrt(dt)*z
    return x + dx


def sample_two_gaus(N=10000):
    N1 = N // 2
    N2 = N - N1
    samples1 = np.random.randn(N1) + 2
    samples2 = np.random.randn(N2) - 2
    samples = np.hstack((samples1, samples2))
    return samples

def p(xt, N=100, mi=-5, ma=5):
    return np.histogram(xt, bins=N, range=(mi,ma))[0]


n_steps = 500
bins = 100
path = np.zeros((bins, n_steps))
single_path = np.zeros((n_steps))
xt = sample_two_gaus()
single_xt = np.array([-2.8])

for i in range(n_steps):
    path[:,i] = p(xt, N=bins)
    xt = step(xt, dt=1/n_steps)

    single_xt = step(single_xt, dt=1/n_steps)
    single_path[i] = single_xt[0]

smooth_path = gaussian_filter(path, sigma=3)

thanos_img = np.array(Image.open('./thanos.png')) / 255. - 0.5

fig = plt.Figure(figsize=(15,5))
fig.tight_layout()

ax = fig.add_subplot(3,9,(4,27))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.imshow(smooth_path, interpolation='nearest', aspect='auto')

ax2 = fig.add_subplot(3,9,(4,27))
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.set_ylim(-5, 5)
ax2.set_xlim(0, n_steps)
ax2.patch.set_alpha(0.)
point = ax2.plot(-6*np.ones(n_steps), 'ro')[0]
line = ax2.plot(single_path, 'r-')[0]

ax_img = fig.add_subplot(3,9,(1,21))
ax_img.get_xaxis().set_visible(False)
ax_img.get_yaxis().set_visible(False)
ax_img.margins(0,0)
ax_img.imshow(thanos_img)

n_animate = 50
pbar = tqdm(total=n_animate)

def update(i):
    point_y = -6*np.ones(n_steps)
    point_y[1000//n_animate*i] = single_path[1000//n_animate*i]
    point.set_ydata(point_y)

    line_y = single_path[:1000//n_animate*i]
    line.set_data(range(line_y.shape[0]), line_y)

    ax_img.cla()
    global thanos_img
    ax_img.imshow(thanos_img + 0.5, interpolation='nearest', aspect='auto')

    thanos_img = step_img(thanos_img)
    print(thanos_img.min(), thanos_img.max())

    pbar.update(1)

ani = animation.FuncAnimation(fig, update, range(n_animate))
ani.save('diff_traj.gif', writer='imagemagick', fps=25, savefig_kwargs=dict(pad_inches=0, bbox_inches='tight'));

# fig.savefig('diff_traj.png')
