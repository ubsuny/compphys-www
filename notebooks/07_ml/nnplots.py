import numpy as np
import matplotlib.pyplot as plt

def apply_layer(y_in,w,b,activation):
    """
    Go from one layer to the next, given a
    weight matrix w (shape [n_neurons_in,n_neurons_out])
    a bias vector b (length n_neurons_out)
    and the values of input neurons y_in
    (shape [batchsize,n_neurons_in])

    returns the values of the output neurons in the next layer
    (shape [batchsize, n_neurons_out])
    """
    # to understand the following line, watch the beginning of lecture 2
    z=np.dot(y_in,w)+b # batch processing: y_in is of shape [batchsize,num_neurons_in]
    activation = activation.lower()
    if activation=='sigmoid':
        return 1/(1+np.exp(-z*2)), z
    elif activation=='jump':
        return np.array(z>0,dtype='float'), z
    elif activation=='linear':
        return z, z
    elif activation=='relu':
        return (z>0)*z, z

def apply_net(y_in, weights, biases, activations):
    """
    Apply a whole network of multiple layers
    """
    y = y_in
    y_all = [y]
    z_all = [[]]
    for j in range(len(biases)):
        y, z = apply_layer(y, weights[j], biases[j], activations[j])
        y_all.append(y)
        z_all.append(z)
    return y_all, z_all

# some internal routines for plotting the network:
def default_connection_style(weight, y, vmax):
    style = {"linewidth": 3}
    if weight > 0:
        style["col"] = [0., 0.4, 0.8, abs(weight) / vmax]
    else:
        style["col"] = [1., 0.3, 0., abs(weight) / vmax]
    return style
    
def plot_connection_line(ax, X, Y, W, y, vmax=1.0, stylefunc=default_connection_style):
    t=np.linspace(0,1,20)
    style = stylefunc(W, y, vmax)
    col = style["col"]
    linewidth = style["linewidth"]
    #if W>0:
    #    col=[0,0.4,0.8]
    #else:
    #    col=[1,0.3,0]
    ax.plot(X[0]+(3*t**2-2*t**3)*(X[1]-X[0]),Y[0]+t*(Y[1]-Y[0]),
           color=col, linewidth=linewidth)

def plot_neuron_alpha(ax, X, Y, B, size=100.0, vmax=1.0):
    if B>0:
        col=[0,0.4,0.8]
    else:
        col=[1,0.3,0]
    ax.scatter([X],[Y],marker='o',c=col,alpha=abs(B)/vmax,s=size,zorder=10)
    
def default_neuron_style(bias, value, vmax):
    style = {"size": 100.}
    if bias > 0:
        style["col"] = [0., 0.4, 0.8]
    else:
        style["col"] = [1., 0.3, 0.]
    return style
    
def plot_neuron(ax, X, Y, B, val, vmax=1.0, stylefunc=default_neuron_style):
    style = stylefunc(B, val, vmax)
    col = style["col"]
    size = style["size"]
    ax.scatter([X], [Y], marker='o', c=[col], s=size, zorder=10)

def visualize_network_2D(weights,biases,activations,
                      M=100,y0range=[-1,1],y1range=[-1,1],
                     size=400.0, linewidth=5.0):
    """
    Visualize a neural network with 2 input
    neurons and 1 output neuron (plot output vs input in a 2D plot)

    weights is a list of the weight matrices for the
    layers, where weights[j] is the matrix for the connections
    from layer j to layer j+1 (where j==0 is the input)

    weights[j][m,k] is the weight for input neuron k going to output neuron m
    (note: internally, m and k are swapped, see the explanation of
    batch processing in lecture 2)

    biases[j] is the vector of bias values for obtaining the neurons in layer j+1
    biases[j][k] is the bias for neuron k in layer j+1

    activations is a list of the activation functions for
    the different layers: choose 'linear','sigmoid',
    'jump' (i.e. step-function), and 'reLU'

    M is the resolution (MxM grid)

    y0range is the range of y0 neuron values (horizontal axis)
    y1range is the range of y1 neuron values (vertical axis)
    """
    swapped_weights=[]
    for j in range(len(weights)):
        swapped_weights.append(np.transpose(weights[j]))

    y0,y1=np.meshgrid(np.linspace(y0range[0],y0range[1],M),np.linspace(y1range[0],y1range[1],M))
    y_in=np.zeros([M*M,2])
    y_in[:,0]=y0.flatten()
    y_in[:,1]=y1.flatten()
    y_all, z_all = apply_net(y_in,swapped_weights,biases,activations)
    y_out = y_all[-1]

    fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(8,4))

    # plot the network itself:

    # positions of neurons on plot:
    posX=[[-0.5,+0.5]]; posY=[[0,0]]
    vmax=0.0 # for finding the maximum weight
    vmaxB=0.0 # for maximum bias
    for j in range(len(biases)):
        n_neurons=len(biases[j])
        posX.append(np.array(range(n_neurons))-0.5*(n_neurons-1))
        posY.append(np.full(n_neurons,j+1))
        vmax=np.maximum(vmax,np.max(np.abs(weights[j])))
        vmaxB=np.maximum(vmaxB,np.max(np.abs(biases[j])))

    # plot connections
    for j in range(len(biases)):
        for k in range(len(posX[j])):
            for m in range(len(posX[j+1])):
                plot_connection_line(ax[0],[posX[j][k],posX[j+1][m]],
                                     [posY[j][k],posY[j+1][m]],
                                     swapped_weights[j][k,m],vmax=vmax,
                                    linewidth=linewidth)

    # plot neurons
    for k in range(len(posX[0])): # input neurons (have no bias!)
        plot_neuron(ax[0], posX[0][k], posY[0][k], 
                   vmaxB, vmax=vmaxB, size=size)
    for j in range(len(biases)): # all other neurons
        for k in range(len(posX[j+1])):
            plot_neuron(ax[0],posX[j+1][k],posY[j+1][k],
                       biases[j][k],vmax=vmaxB,size=size)

    # Padding
    xflat = [x for xx in posX for x in xx]
    xmin = min(xflat)
    xmax = max(xflat)
    yflat = [y for yy in posY for y in yy]
    ymin = min(yflat)
    ymax = max(yflat)
    
    ax[0].set_xlim(xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin))
    ax[0].set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

    ax[0].axis('off')

    # now: the output of the network
    img=ax[1].imshow(np.reshape(y_out,[M,M]),origin='lower',
                    extent=[y0range[0],y0range[1],y1range[0],y1range[1]])
    ax[1].set_xlabel(r'$y_0$')
    ax[1].set_ylabel(r'$y_1$')

    axins1 = inset_axes(ax[1],
                    width="40%",  # width = 50% of parent_bbox width
                    height="5%",  # height : 5%
                    loc='upper right')

    imgmin=np.min(y_out)
    imgmax=np.max(y_out)
    color_bar=fig.colorbar(img, cax=axins1, orientation="horizontal",ticks=np.linspace(imgmin,imgmax,3))
    cbxtick_obj = plt.getp(color_bar.ax.axes, 'xticklabels')
    plt.setp(cbxtick_obj, color="white")
    axins1.xaxis.set_ticks_position("bottom")

    plt.show()

    return fig, ax, posX, posY

# linewidth=5.0, 
# neuron_size=400.0
def visualize_network(weights, biases, activations, y_in=None, 
                      xspacing=1.0, yspacing=1.0, figsize=(8, 8), 
                      neuron_kwargs={}, connection_kwargs={}, return_all=False
                     ):
    """
    Visualize a neural network

    weights is a list of the weight matrices for the
    layers, where weights[j] is the matrix for the connections
    from layer j to layer j+1 (where j==0 is the input)

    weights[j][m,k] is the weight for input neuron k going to output neuron m
    (note: internally, m and k are swapped, see the explanation of
    batch processing in lecture 2)

    biases[j] is the vector of bias values for obtaining the neurons in layer j+1
    biases[j][k] is the bias for neuron k in layer j+1

    activations is a list of the activation functions for
    the different layers: choose 'linear','sigmoid',
    'jump' (i.e. step-function), and 'reLU'

    y_in is the inputs
    """
    swapped_weights=[]
    for j in range(len(weights)):
        swapped_weights.append(np.transpose(weights[j]))

    y_all, z_all = apply_net(y_in, swapped_weights, biases, activations)
    y_out = y_all[-1]

    fig, ax=plt.subplots(figsize=figsize)

    # plot the network itself:

    # positions of neurons on plot:
    #posX=[[-0.5,+0.5]]; posY=[[0,0]]
    posX = []
    posY = []
    vmax=0.0 # for finding the maximum weight
    vmaxB=0.0 # for maximum bias
    for j in range(len(biases) + 1):
        if j == 0:
            n_neurons = len(y_in)
        else:
            n_neurons=len(biases[j-1])
        posX.append((np.array(range(n_neurons))-0.5*(n_neurons-1)) * xspacing)
        posY.append(np.full(n_neurons, j * yspacing))
        vmax=np.maximum(vmax,np.max(np.abs(weights[j-1])))
        vmaxB=np.maximum(vmaxB,np.max(np.abs(biases[j-1])))

    vmax = connection_kwargs.pop("vmax", vmax)
    # plot connections
    for j in range(len(biases)):
        for k in range(len(posX[j])):
            for m in range(len(posX[j+1])):
                plot_connection_line(ax, [posX[j][k], posX[j+1][m]],
                                     [posY[j][k], posY[j+1][m]],
                                     swapped_weights[j][k,m], y_all[j][k], vmax=vmax,
                                    **connection_kwargs)

    # plot neurons
    for k in range(len(posX[0])): # input neurons (have no bias!)
        plot_neuron(ax, posX[0][k], posY[0][k],
                   vmaxB, y_all[0][k], vmax=vmaxB, **neuron_kwargs)
    for j in range(len(biases)): # all other neurons
        for k in range(len(posX[j+1])):
            plot_neuron(ax, posX[j+1][k], posY[j+1][k],
                       biases[j][k], y_all[j+1][k], vmax=vmaxB, **neuron_kwargs)

    # Padding
    xflat = [x for xx in posX for x in xx]
    xmin = min(xflat)
    xmax = max(xflat)
    yflat = [y for yy in posY for y in yy]
    ymin = min(yflat)
    ymax = max(yflat)
    
    ax.set_xlim(xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin))
    ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

    ax.axis('off')

    return fig, ax, posX, posY, y_all, z_all