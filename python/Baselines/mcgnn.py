import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseChebConv(nn.Module):
    """
    Computes ChebConv using dense matrix multiplication.
    Y = \sum_{k=0}^{K-1} T_k(L) X W_k
    """
    def __init__(self, in_features, out_features, K):
        super().__init__()
        self.K = K
        self.W = nn.Parameter(torch.Tensor(K, in_features, out_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, L_dense):
        """
        x: [Num_Nodes, In_Features]
        L_dense: [Num_Nodes, Num_Nodes] Normalized Laplacian
        """
        num_nodes = x.shape[0]
        
        # Chebyshev recurrence
        # T_0(X) = X
        # T_1(X) = L X
        # T_k(X) = 2 L T_{k-1} - T_{k-2}
        
        T0 = x
        T1 = torch.mm(L_dense, x)
        
        outputs = []
        
        # k=0 term: T0 @ W[0]
        outputs.append(torch.mm(T0, self.W[0]))
        
        # k=1 term: T1 @ W[1]
        if self.K > 1:
            outputs.append(torch.mm(T1, self.W[1]))
            
        # k > 1 terms
        Tk_prev = T1
        Tk_2prev = T0
        
        for k in range(2, self.K):
            Tk = 2 * torch.mm(L_dense, Tk_prev) - Tk_2prev
            outputs.append(torch.mm(Tk, self.W[k]))
            Tk_2prev = Tk_prev
            Tk_prev = Tk
            
        return sum(outputs)

class sMGCNN_Layer(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, K=3):
        super().__init__()
        # 1. User Convolution (Row Graph)
        # Input: Matrix X [U, I]. Treats 'I' as features.
        self.user_conv = DenseChebConv(in_features=num_items, out_features=latent_dim, K=K)
        
        # 2. Item Convolution (Column Graph)
        # Input: Matrix X^T [I, Latent_Dim]. Treats 'Latent_Dim' as features.
        self.item_conv = DenseChebConv(in_features=latent_dim, out_features=latent_dim, K=K)
        
        self.act = nn.ReLU()

    def forward(self, x, L_user, L_item):
        # x shape: [Users, Items]
        
        # 1. Convolve Users (Rows)
        # We view X as signals on the User graph. Features = Items.
        h = self.user_conv(x, L_user) # Output: [Users, Latent_Dim]
        
        # 2. Convolve Items (Columns)
        # We view H as signals on the Item graph. Features = Users (but now compressed to Latent).
        # We need to transpose to apply conv on Item dimension.
        h = h.t() # [Latent_Dim, Users] -> Wait, this logic depends on the separable definition.
        
        # In Monti's sMGCNN, it's often:
        # H = Conv_User(X)  (transforms rows, features are columns)
        # Y = Conv_Item(H^T) (transforms columns, features are new row embeddings)
        
        # Let's assume standard "Row then Col" filtering:
        # User Conv output: [Users, Latent]
        # Transpose: [Latent, Users] - Wait, item graph needs [Items, ...]
        
        # Actually, the paper defines "Separable" as:
        # Feature extraction on both graphs simultaneously or sequentially.
        # Often implemented as: Project Users to Latent, Project Items to Latent, then outer product.
        
        # Let's stick to the Recurrent formulation from the paper:
        # Input is usually a diff matrix.
        # They apply filters on both sides.
        
        # Simplified implementation of "Separable"
        # 1. Apply User Graph Filter: Y1 = L_u @ X
        # 2. Apply Item Graph Filter: Y2 = X @ L_i
        # (This assumes simple diffusion without channel mixing first)
        
        pass

# --- REVISED SIMPLE IMPLEMENTATION (The "Recurrent" Logic) ---

class MontiMatrixCompletion(nn.Module):
    def __init__(self, num_users, num_items, K=3):
        super().__init__()
        self.K = K
        # Learnable weights for the diffusion process
        self.theta = nn.Parameter(torch.randn(K, K)) 
        # RNN to integrate the diffusion steps
        self.rnn = nn.GRUCell(input_size=num_items, hidden_size=num_items)

    def forward(self, x_mask, L_user, L_item, steps=10):
        """
        x_mask: The initial sparse matrix (0s where missing) [Users, Items]
        L_user: User Graph Laplacian [Users, Users]
        L_item: Item Graph Laplacian [Items, Items]
        """
        # Initialize reconstruction
        X = x_mask.clone()
        
        # Hidden state for RNN
        h = torch.zeros_like(X)
        
        for _ in range(steps):
            # 1. Multi-Graph Convolution Step (Diffusion)
            # Simple diffusion: L_u * X * L_i (Separable Laplacian filter)
            
            # User side diffusion (Row smoothing)
            # T_k(L_u) X
            diff_u = torch.mm(L_user, X)
            
            # Item side diffusion (Column smoothing)
            # X T_k(L_i)
            diff_i = torch.mm(X, L_item)
            
            # Combine (A simple separable layer fusion)
            diffusion_signal = diff_u + diff_i # Simplified for demo
            
            # 2. Recurrent Update
            # The RNN decides how much to update the matrix based on diffusion
            # Treating rows as sequences/batches
            h = self.rnn(diffusion_signal, h)
            delta_X = h
            
            X = X + delta_X
            
            # Important: Reset known values to ground truth (Force fidelity)
            # mask is 1 where we have data, 0 where missing
            mask_indices = (x_mask != 0)
            X[mask_indices] = x_mask[mask_indices]
            
        return X