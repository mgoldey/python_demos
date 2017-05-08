import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

class ParticleBox:
	"""    
	init_state is an [N x 6] array, where N is the number of particles:
	   [[x1, y1, vx1, vy1, ax1, ay1],
		[x2, y2, vx2, vy2, ax2, ay2],
		...               ]

	bounds is the dimensions of the box: [xmin, xmax, ymin, ymax]
	"""
	def __init__(self,
				 init_state = np.identity(6)[:4],
				 bounds = [-2,2,-2,2],
				 radii=0.05*np.ones(4),
				 temp=300,
				 M = 1.0):
		self.init_state = np.asarray(init_state, dtype=float)
		self.radii=radii
		self.M = M * np.ones(self.init_state.shape[0])
		self.state = self.init_state.copy()
		self.time_elapsed = 0
		self.bounds = bounds
		self.num_particles=self.init_state.shape[0]
		self.T=temp # TEMP IN kB

	def reset_vcm(self):
		""" Resets the velocity of the center of mass """
		M=np.sum(self.M)
		v_cm=np.dot(self.M,self.state[:,2:4])/M
		self.state[:,2:4]-=v_cm

	def KE(self):
		v2=np.linalg.norm(self.state[:,2:4],axis=1)**2
		KE=0.5*np.sum(self.M.dot(v2))
		return KE

	def thermostat(self,dt=1.0,tau=1.0):
		""" Rescales velocities to match temperature """
		effective_temp=self.KE()/(1.0*len(self.radii)) # 2/2 for 2 dimensions

		# STANDARD VELOCITY RESCALING
		# l=np.sqrt(self.T/effective_temp)  

		# BERENDSEN THERMOSTAT
		l=np.sqrt(1.0+(dt/tau)*(self.T/effective_temp-1)) 

		# RESCALE VELOCITIES
		self.state[:,2:4]*=l
		# print(self.KE())

	def step(self,dt):
		"""step once by dt seconds"""    
		self.time_elapsed += dt
		if (self.time_elapsed//dt)%10==0:	        
			self.reset_vcm()
			self.thermostat(dt,tau=dt*100)

		dt2=dt*dt

		# UPDATE positions
		self.state[:, :2] += dt * self.state[:, 2:4] +0.5*self.state[:,4:]*dt2

		# UPDATE accelerations
		D = squareform(pdist(self.state[:, :2]))
		ind1, ind2 = np.where(D < 3 * self.radii.max())
		unique = (ind1 < ind2)
		ind1 = ind1[unique]
		ind2 = ind2[unique]
		old_acc=1.0*self.state[:,4:]
		self.state[:,4:]=0.
		for i1, i2 in zip(ind1, ind2):
			# mass
			m1 = self.M[i1]
			m2 = self.M[i2]
			# location vectors
			r1 = self.state[i1, :2]
			r2 = self.state[i2, :2]
			r_rel=r1-r2

			s=self.radii[i1]+self.radii[i2]
			eps=4.0*(1.+int(self.radii[i1]==self.radii[i2]))
			r=(s/np.linalg.norm(r_rel))**2
			r8=r**4
			r6=r**3
			#f1=4*eps*(r12-r6)*r_rel/np.linalg.norm(r_rel)
			f1=48*eps*r_rel*r8*(r6-.5)
			self.state[i2,4:]+=-f1/m2
			self.state[i1,4:]+=f1/m1

		# UPDATE velocities
		self.state[:, 2:4] += 0.5*dt * (self.state[:, 4:] + old_acc)

		# check for crossing boundary
		leftx=(self.state[:,0]-self.radii)<self.bounds[0]
		rightx=(self.state[:,0]+self.radii)>self.bounds[1]
		bottomy=(self.state[:,1]-self.radii)<self.bounds[2]
		topy=(self.state[:,1]+self.radii)>self.bounds[3]
		
		# REVERSE DIRECTIONS
		self.state[:,2][leftx+rightx]*=-1
		self.state[:,3][bottomy+topy]*=-1


#------------------------------------------------------------
# set up initial state
nump=250
init_state=np.zeros((nump,6))
lbx=lby=-2.
ubx=uby=2.
size=0.075

if False:
	# RANDOM SIZES
	sizes=size*(np.round(np.random.random(nump))+1)
else:
	# SET HALF TO be bigger than the others
	sizes=np.ones(nump)*size
	sizes[len(sizes)//2:]*=2

x,y=np.ogrid[lbx+size*2:ubx-size*2:size*3,lby+size*2:uby-size*2:size*3]
a,b=np.meshgrid(x,y)
a=a.flatten()
b=b.flatten()
for i in range(nump):
	init_state[i,:2]=a[i],b[i]
init_state[:,2:4]=np.random.random((nump,2))*1.0

box = ParticleBox(init_state,radii=sizes,temp=300)
dt = 1e-4

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
					 xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))

# particles holds the locations of the particles
particles1, = ax.plot([], [], 'bo', ms=6)
particles2, = ax.plot([], [], 'bo', ms=6)
#particles, = ax.scatter([], [])

# rect is the box edge
rect = plt.Rectangle(box.bounds[::2],
					 box.bounds[1] - box.bounds[0],
					 box.bounds[3] - box.bounds[2],
					 ec='none', lw=2, fc='none')
ax.add_patch(rect)
title=ax.text(.5,.95,'centered title',
        horizontalalignment='center',
        transform=ax.transAxes,
        fontsize=20)

def init():
	"""initialize animation"""
	global box, rect,title
	particles1.set_data([], [])
	particles2.set_data([], [])
	rect.set_edgecolor('k')
	title.set_text("T= %.4f" % 0.0)
	return particles1,particles2, rect,title

def animate(i):
	"""perform animation step"""
	global box, rect, dt, ax, fig,title
	for i in range(5):
		box.step(dt)

	title.set_text("T= %.4f" % box.time_elapsed)
	particles1.set_color('b')
	p1=box.radii==box.radii.min()
	particles1.set_data(box.state[:, 0][p1], box.state[:, 1][p1])
	ms1 = 0.6*float((fig.dpi * 2 * box.radii.min() * fig.get_figwidth()
			 / np.diff(ax.get_xbound())[0]))
	particles1.set_markersize(ms1)

	if len(np.unique(box.radii))>1:
		particles2.set_color('r')
		p2=box.radii==box.radii.max()
		particles2.set_data(box.state[:, 0][p2], box.state[:, 1][p2])
		ms2 = .6*float((fig.dpi * 2 * box.radii.max() * fig.get_figwidth()
				 / np.diff(ax.get_xbound())[0]))
		particles2.set_markersize(ms2)

	return particles1, particles2, rect,title

# HACK TO UPDATE TITLE DYNAMICALLY
def _blit_draw(self, artists, bg_cache):
	# Handles blitted drawing, which renders only the artists given instead
	# of the entire figure.
	updated_ax = []
	for a in artists:
		# If we haven't cached the background for this axes object, do
		# so now. This might not always be reliable, but it's an attempt
		# to automate the process.
		if a.axes not in bg_cache:
			# bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
			# change here
			bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
		a.axes.draw_artist(a)
		updated_ax.append(a.axes)

	# After rendering all the needed artists, blit each axes individually.
	for ax in set(updated_ax):
		# and here
		# ax.figure.canvas.blit(ax.bbox)
		ax.figure.canvas.blit(ax.figure.bbox)

animation.Animation._blit_draw = _blit_draw


ani = animation.FuncAnimation(fig, animate, frames=2000,
			  interval=0, blit=True, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# plt.show()
