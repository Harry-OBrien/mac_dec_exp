import sys

from . import weight

class hidx_t:
    def __init__(self,i):
        self.i = i

def idx_of(i):
    return i.i - 1
    
INVALID_HIDX = hidx_t(0)
def hidx_is_invalid(i):
    return i.i == 0
    
hidx_first = hidx_t(1)

# 1-based operations
def parent(i):
    return hidx_t(i.i //2)
def left(i):
    return hidx_t(2*i.i)
def right(i):
    return hidx_t((2*i.i) + 1)
def has_parent(i):
    return parent(i).i > 0
def has_left(n, i):
    return left(i).i<= n
def has_right(n, i):
    return right(i).i <= n

class DPQ_t:
    def __init__(self, size):
        self.num_elem = size
        self.heap_size = 0
        self.D = [None]*size
        self.H = [None]*size
        self.I = [None]*size

        for i in range(0, self.num_elem):
            self.I[i] = INVALID_HIDX
            self.D[i] = weight.weight_inf()

    def DPQ_dist_free(self):
        res = self.D
        return res

    def DPQ_prio(self, u):
        assert(u<self.num_elem)
        return self.D[u]
        
    def _DPQ_hprio(self, i):
        return self.DPQ_prio(self.H[idx_of(i)])

    def _DPQ_adjustI(self, i):
        self.I[self.H[idx_of(i)]] = i
        
    # Helper function for swapping nodes in the tree
    def _DPQ_swap(self, i, j):
        t = self.H[idx_of(i)]
        self.H[idx_of(i)] = self.H[idx_of(j)]
        self.H[idx_of(j)] = t
        
        self._DPQ_adjustI(i)
        self._DPQ_adjustI(j)
        
    def _DPQ_sift_up(self, i):
        while (has_parent(i) and weight.weight_less(self._DPQ_hprio(i), self._DPQ_hprio(parent(i)))):
            self._DPQ_swap(i, parent(i))
            i = parent(i)
        
    def _DPQ_sift_down(self, i):
        while (has_right(self.heap_size, i)):
            w = self._DPQ_hprio(i)
            lw = self._DPQ_hprio(left(i))
            rw = self._DPQ_hprio(right(i))
            if (not weight.weight_less(lw, w) and not weight.weight_less(rw, w)):
                # No children, we're finisehd
                return
                
            if (weight.weight_less(lw, rw)):
                self._DPQ_swap(i, left(i))
                i = left(i)
            else:
                self._DPQ_swap(i, right(i))
                i = right(i)
                
        if (has_left(self.heap_size,i)):
            w = self._DPQ_hprio(i)
            lw = self._DPQ_hprio(left(i))
            if (weight.weight_less(lw, w)):
                self._DPQ_swap(i, left(i))

    def DPQ_contains(self, u):
        assert(u<self.num_elem)
        # Linear Search required as unordered
        return not hidx_is_invalid(self.I[u])

    def _DPQ_last_idx(self):
        return hidx_t(self.heap_size)

    def DPQ_insert(self, u, w):
        assert(u < self.num_elem)
        assert(not self.DPQ_contains(u))
        self.D[u]=w
        self.heap_size = self.heap_size + 1
        i = self._DPQ_last_idx()
        self.H[idx_of(i)] = u
        self._DPQ_adjustI(i)
        self._DPQ_sift_up(i)

    def DPQ_is_empty(self):
        return self.heap_size == 0

    def DPQ_pop_min(self):
        assert(not self.DPQ_is_empty())
        li = self._DPQ_last_idx()
        self._DPQ_swap(hidx_first, li)
        res = self.H[idx_of(li)]
        self.heap_size = self.heap_size - 1
        self.I[res] = INVALID_HIDX
        self._DPQ_sift_down(hidx_first)
        return res
        
    def DPQ_decrease_key(self, u, w):
        assert(u < self.num_elem)
        assert(self.DPQ_contains(u))
        assert(weight.weight_less(w, self.D[u]))
        self.D[u] = w
        
        i = self.I[u]
        self._DPQ_sift_up(i)
    
