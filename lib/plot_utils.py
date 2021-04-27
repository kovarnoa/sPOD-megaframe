import matplotlib.pyplot as plt
import numpy as np


from matplotlib import rc
rc('font',**{'family':'serif','serif':['Helvetica'],'size'   : 12})
#
rc('text', usetex=True)

pic_dir = "../imgs"
plt.close("all")

def show_animation(q, Xgrid=None, cycles=1, frequency = 1, figure_number = None):
    
    ndim = np.size(Xgrid)
    ntime = q.shape[-1]
    if figure_number == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(num=figure_number)    
        
    if Xgrid is not None: 
    
        h = ax.pcolormesh(Xgrid[0], Xgrid[1], q[..., 0])
        h.set_clim(np.min(q),np.max(q))
        ax.axis("image")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        for t in range(0, cycles * ntime, frequency):
            h.set_array(q[:-1,:-1,t % ntime].ravel())
            #fig.savefig("../imgs/FTR_6modes_%3.3d.png" % t)
            plt.draw()
            plt.pause(0.05)
            
    else:
        x = np.arange(0,np.size(q,0))
        h, = ax.plot(x,q[:, 0])
        ax.set_ylim(np.min(q),np.max(q))
        ax.set_xlabel(r"$x$")
        for t in range(0, cycles * ntime, frequency):
            h.set_data(x,q[:,t % ntime])
            #fig.savefig("../imgs/FTR_6modes_%3.3d.png" % t)
            plt.draw()
            plt.pause(0.05)

