from . import general

class path_t:
    def set_up(self, len):
        self.len = len
        self.nodes = [None]*len
        for i in range(0, len):
            self.nodes[i] = general.INVALID_NODE
        
    def __init__(self, pred, v):
        # Determine length
        len = 0
        u = v
        
        while (True):
            len = len + 1
            pred_u = (pred[u][0], pred[u][1])
            if (u == pred_u):
                break
            u = pred_u
            
        self.set_up(len)
        u = v
        
        while (True):
            len = len - 1
            self.path_set(len, u)
            pred_u = (pred[u][0], pred[u][1])
            if (u == pred_u):
                break
            u = pred_u

    def path_set(self, i, u):
        assert (i < self.len)
        self.nodes[i] = u
    
    def path_get(self, i):
        assert(i < self.len)
        return self.nodes[i]
        
    def path_len(self):
        return self.len
        