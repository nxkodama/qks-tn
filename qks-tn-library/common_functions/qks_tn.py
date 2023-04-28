import numpy as np
import tensorflow as tf
import tensornetwork as tn
from scipy.stats import unitary_group

import functools
import opt_einsum
from tensornetwork.network_components import get_all_nondangling, contract_parallel, contract_between
from tensornetwork.contractors.opt_einsum_paths import utils
from tensornetwork.network_components import Edge, AbstractNode
from typing import Any, Optional, Sequence, Iterable

########## Custom Layers ##########
# Feature encoding layer, initialized as a quantum kitchen sink
class FeatureEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, units, bond_dim, input_dim, sigma):
        super(FeatureEncodingLayer, self).__init__()
        # Defining variables for the layer
        self.chi = bond_dim
        # Weights
        self.W = tf.Variable(tf.random.normal(shape=(units, input_dim),
                                                  stddev=sigma),
                                 name="W", trainable=True)
        # Biases
        self.b = tf.Variable(2*np.pi*tf.random.uniform(shape=(units,)),
                                name="b", trainable=True)
    def call(self, inputs):
        # Parallelizing over a batch of examples using tf.vectorized_map
        def f(input_vec, chi, W, b):
            # Retrieving parameters
            units = b.shape[0]
            nqubits = np.log2(chi)
            # Computing driving angles and density matrices
            theta = tf.linalg.matvec(W, input_vec) + b
            R00 = tf.dtypes.complex(tf.math.cos(theta/2)**2,0.)
            R01 = tf.dtypes.complex(0.,tf.math.sin(theta)/2)
            R11 = tf.dtypes.complex(tf.math.sin(theta/2)**2,0.)            
            rho_ = tf.reshape(tf.stack([R00,R01,-R01,R11]),([2,2,units]))
            # Combining density matrices for higher bond dimension tensor networks
            # TODO: generalize bond dimension
            rho = tf.zeros((chi,chi,int(units/nqubits)),dtype=tf.complex64)
            for i in range(0,units,2):
                indices = np.zeros((chi,chi,int(units/nqubits)))
                indices[:,:,int(i/2)] = 1
                indices = np.stack(np.where(indices==1)).T
                rho = tf.tensor_scatter_nd_add(rho,indices,
                                                tf.reshape(tf.transpose(tf.tensordot(rho_[:,:,i],rho_[:,:,i+1],axes=0),perm=[0,2,1,3]),shape=(16,)))
            return rho
        # To deal with a batch of examples, we use tf.vectorized_map
        result = tf.vectorized_map(
            lambda vec: f(vec, self.chi, self.W, self.b), inputs)
        return result

# Tensor network layer, constructed as a dissipative tree tensor network
class TNLayer(tf.keras.layers.Layer):
    def __init__(self,chi,nlayers,path):
        super(TNLayer, self).__init__()
        # Defining the variables for the layer.
        self.path = path
        nvertices = int(2**(nlayers)-1)
        d = chi**2
        nepisodes = nvertices+1
        # Parameterizing unitary tensors
        params = np.zeros((int(d**2-1),nvertices))
        for m in range(nvertices):
            # Drawing a random Unitary matrix
            u = unitary_group.rvs(d)
            unitary = np.reshape(u, [chi]*4)
            # Obtaining the corresponding Hermitian matrix
            _,V = np.linalg.eig(u)
            H = V @ np.diag(np.log(np.diag(np.linalg.inv(V) @ u @ V))) @ np.linalg.inv(V)/1j
            # Retrieving the parameters of the Hermitian matrix
            H_re = H.real
            H_im = H.imag
            ind = np.triu_indices(d,1)
            ind_diag = np.diag_indices(d-1)
            params[:,m] = np.concatenate((H_re[ind],H_im[ind],H_re[ind_diag]))
        self.params = tf.Variable(params.astype('float32'))
    def call(self, inputs):
        # Parallelizing over a batch of examples using tf.vectorized_map
        def f(input_mat, params):
            # Retrieving parameters
            N = params.shape[0]
            nvertices = params.shape[1]
            nlayers = int(np.log2(nvertices+1))
            d = int(np.sqrt(N+1))
            chi = int(np.sqrt(d))
            n = int(d*(d-1)/2)
            nqubits = int(np.log2(chi))
            nepisodes = int(nqubits*(nvertices+1))
            # Obtaining each unitary tensor: parameters -> Hermitian matrix -> unitary matrix
            uni_array = tf.zeros(([chi]*4+[nvertices]),dtype=tf.complex64)
            for m in range(nvertices):
                # Parameterizing the corresponding Hermitian matrix
                ind_re = np.stack(np.triu_indices(d,1)).T
                ind_im = np.stack(np.triu_indices(d,1)).T
                ind_diag = np.stack(np.diag_indices(d-1)).T
                H = tf.constant(np.zeros((d,d),dtype='complex64'))
                H = tf.tensor_scatter_nd_add(H,ind_re,tf.cast(params[:n,m],'complex64'))
                H = H + tf.transpose(H)
                H = tf.tensor_scatter_nd_add(H,ind_diag,tf.cast(params[2*n:,m],'complex64'))
                H = tf.tensor_scatter_nd_add(H,np.array([[d-1,d-1]]),[-tf.math.reduce_sum(tf.cast(params[2*n:,m],'complex128'))])
                im = tf.constant(np.zeros((d,d),dtype='complex64'))
                im = tf.tensor_scatter_nd_add(H,ind_im,tf.cast(params[n:2*n,m],'complex64'))
                im = im - tf.transpose(im)
                H = H + 1j*im
                # Exponentiating the Hermitian to get the unitary matrix
                U = tf.linalg.expm(1j*H)
                # Reshaping unitary as a tensor
                unitary = tf.reshape(U,[chi]*4)
                indices = np.zeros(([chi]*4+[nvertices]))
                indices[:,:,:,:,m] = 1
                indices = np.stack(np.where(indices==1)).T
                uni_array = tf.tensor_scatter_nd_add(uni_array,indices,tf.experimental.numpy.ravel(unitary))
            # Preparing the density matrices for the contraction            
            observables = []
            for j in range(int(nepisodes/2)):
                observables.append(input_mat[:,:,j])
            # Defining the contraction.
            nodes_set, edge_order = construct_dttn(chi,uni_array,observables,nepisodes)
            # Contracting the network
            result, path = greedy(nodes_set, self.path, output_edge_order=edge_order)
            M = tf.math.real(tf.linalg.diag_part(tf.reshape(result.tensor,[chi**2]*2)))
            return M
        # To deal with a batch of items, we use tf.vectorized_map
        result = tf.vectorized_map(
            lambda matrix: f(matrix, self.params), inputs)
        return result

# Constant matrix multiplication, a fixed linear projection onto 10 classes
class ConstMul(tf.keras.layers.Layer):
    def __init__(self):
        super(ConstMul, self).__init__()
        self.const = np.array([[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]],dtype='float32')
    def call(self,inputs,**kwargs):
        return tf.linalg.matvec(self.const, inputs)    

# Constant matrix multiplication, a fixed linear projection onto 10 classes
class ConstMulBinary(tf.keras.layers.Layer):
    def __init__(self):
        super(ConstMulBinary, self).__init__()
        self.const = np.array([[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]],dtype='float32')
    def call(self,inputs,**kwargs):
        return tf.linalg.matvec(self.const, inputs)    

########## Tensor Network Constructors ##########
# Defining the contraction for a dissipative TTN
def construct_dttn(chi,uni_array,density_matrices,nepisodes):
    nqubits = int(np.log2(chi))
    nlayers = int(np.log2(nepisodes/nqubits))
    nvertices = int(nepisodes/nqubits-1)

    nodes_set = list()
    with tn.NodeCollection(nodes_set):
        # Initializing density matrices
        obs = []
        for j in range(int(nvertices+1)):
            obs.append(tn.Node(density_matrices[j],name="rho"+str(j)))

        # Initializing unitary nodes
        s = 0
        uni = []
        uni_conj = []
        nunitaries = int((nvertices+1)/2)
        for i in range(nlayers):
            uni_layer = []
            uni_layer_conj = []
            for j in range(nunitaries):
                uni_layer.append(tn.Node(uni_array[:,:,:,:,i],name="U"+str(s)))
                uni_layer_conj.append(tn.Node(tf.math.conj(uni_array[:,:,:,:,i])))
                s = s + 1
            uni.append(uni_layer)
            uni_conj.append(uni_layer_conj)
            nunitaries = int(nunitaries/2)

        # Connecting unitaries with density matrices (Layer 0)
        for i in range(len(uni[0])):
            for j in range(2):
                tn.connect(uni[0][i][j],obs[2*i+j][0],name=str(j))
                tn.connect(uni_conj[0][i][j],obs[2*i+j][1],name=str(j))

        # Connecting intermediate layers
        for i in range(nlayers-1):
            for j in range(len(uni[i])):
                tn.connect(uni[i][j][-1],uni[i+1][int(np.floor(j/2))][j%2])
                tn.connect(uni_conj[i][j][-1],uni_conj[i+1][int(np.floor(j/2))][j%2])

        # Connecting loops of every unitary
        for i in range(nlayers-1):
            for j in range(len(uni[i])):
                tn.connect(uni[i][j][2],uni_conj[i][j][2])

        edge_order = []
        edge_order.append(uni[-1][0][-1])
        edge_order.append(uni[-1][0][2])
        edge_order.append(uni_conj[-1][0][-1])
        edge_order.append(uni_conj[-1][0][2])

    return nodes_set, edge_order

########## Contraction Algorithms ##########
# Modifying the base/greedy contraction algorithms from TensorNetwork to return the contraction path
def base(nodes: Iterable[AbstractNode],
    algorithm: utils.Algorithm,
    path: Optional[Sequence[tuple]] = None,
    output_edge_order: Optional[Sequence[Edge]] = None,
    ignore_edge_order: bool = False) -> AbstractNode:
    
    nodes_set = set(nodes)
    edges = tn.network_operations.get_all_edges(nodes_set)
    
    if not ignore_edge_order:
        if output_edge_order is None:
            output_edge_order = list(tn.network_operations.get_subgraph_dangling(nodes))
            if len(output_edge_order) > 1:
                raise ValueError("The final node after contraction has more than "
                                 "one remaining edge. In this case `output_edge_order` "
                                 "has to be provided.")
        
        if set(output_edge_order) != tn.network_operations.get_subgraph_dangling(nodes):
            raise ValueError("output edges are not equal to the remaining "
                             "non-contracted edges of the final node.")
    
    for edge in edges:
        if not edge.is_disabled:  #if its disabled we already contracted it
            if edge.is_trace():
                nodes_set.remove(edge.node1)
                nodes_set.add(contract_parallel(edge))
                
    if len(nodes_set) == 1:
        # There's nothing to contract.
        if ignore_edge_order:
            return list(nodes_set)[0]
        return list(nodes_set)[0].reorder_edges(output_edge_order)
    
    if path is None:
        path, nodes = utils.get_path(nodes, algorithm)
    
    for a, b in path:
        new_node = contract_between(nodes[a], nodes[b], allow_outer_product=True)
        nodes.append(new_node)
        nodes = utils.multi_remove(nodes, [a, b])
    
    final_node = nodes[0]  # nodes were connected, we checked this
    if not ignore_edge_order:
        final_node.reorder_edges(output_edge_order)
    
    return final_node, path

def greedy(nodes: Iterable[AbstractNode],
    path: Optional[Sequence[tuple]] = None,
    output_edge_order: Optional[Sequence[Edge]] = None,
    memory_limit: Optional[int] = None,
    ignore_edge_order: bool = False) -> AbstractNode:
    
    alg = functools.partial(opt_einsum.paths.greedy, memory_limit=memory_limit)
    return base(nodes, alg, path, output_edge_order, ignore_edge_order)