import numpy as np

from model.stgat import tools


class AdjMatrixGraph:
    def __init__(self, skeleton, num_nodes):

        if skeleton == 'execheck_skeleton':
            outward = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (2, 7), (7, 8), (8, 9), 
                      (0, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (3, 16)] # proximal->distal

            if num_nodes == 21: # +hands+feet
                outward += [(6, 17), (9, 18), (12, 19), (15, 20)]
        elif skeleton == 'uiprmd_skeleton':
            outward = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0,7), (7,8), (8,9), (9,10), 
                    (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)] # prox->distal
        else:
            raise NotImplementedError(f'no dataset {skeleton}')                
        inward = [(j, i) for (i, j) in outward]
        neighbor = inward + outward

        self.num_nodes = num_nodes
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(neighbor, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(neighbor + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary_with_I)
        # self.A_sep = tools.seperated_adjacency(self.A_binary_with_I, [0,2,4,6,8]) 
        self.A_sep = tools.seperated_adjacency(self.A_binary_with_I, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

if __name__ == '__main__':
    
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    
    graph = AdjMatrixGraph()
    A, A_binary_with_I, A_sep = graph.A, graph.A_binary_with_I, graph.A_sep
    
    m, n = 3, 6
    f, ax = plt.subplots(m,n)
    ax[0,0].imshow(A, cmap='gray')
    ax[0,0].set_title("AdjMat(normailized)")
    ax[0,1].imshow(A_binary_with_I, cmap='gray')
    ax[0,1].set_title("A_binary_with_I")
    [ax[0,k].axis('off') for k in range(2,n)]

    for i in range(1,m):
        for j in range(n):
            ax[i,j].imshow(A_sep[(i-1)*n+j], cmap='gray')
            ax[i,j].set_title(f"A_sep #{(i-1)*n+j}")
        
    plt.show()
    
    print('A_binary_with_I:\n',A_binary_with_I)
    print(f"number of separated adj: {len(A_sep)}")
    print('A_sep last:\n',A_sep[-1])
