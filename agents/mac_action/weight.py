import sys
from . import general

WEIGHT_MIN = -sys.maxsize
WEIGHT_MAX = sys.maxsize - 1

_WINF = sys.maxsize
_WNINF = -sys.maxsize - 1


class weight_t:
    def __init__(self,w):
        if (w == _WINF or w == _WNINF):
            general.error("Weight overflow")
        self.w = w
        
    def weight_is_inf(self):
        return self.w == _WINF
        
    def weight_is_neg_inf(self):
        return self.w == _WNINF
        
    def weight_is_finite(self):
        return (not self.weight_is_inf() and not self.weight_is_neg_inf())
        
    def weight_to_int(self):
        assert self.weight_is_finite(), "Weight must be finite"
        return self.w
        
    def print_weight(self, f):
        if (self.weight_is_inf()):
            f.write("inf")
        elif (self.weight_is_neg_inf()):
            f.write("-inf")
        else:
            f.write("%d" % (self.weight_to_int()))

class weight_inf(weight_t):
    def __init__(self):
        self.w = _WINF
        
class weight_neg_inf(weight_t):
    def __init__(self):
        self.w = _WNINF
        
def weight_one():
    return weight_t(1)

def weight_zero():
    return weight_t(0)
    
def weight_add(a, b):
    if (a.weight_is_inf()):
        assert (not b.weight_is_neg_inf()), "inf + -inf undefined"
        return weight_inf()
    elif (a.weight_is_neg_inf()):
        assert(not b.weight_is_inf()), "-inf + inf undefined"
        return weight_neg_inf()
    elif (b.weight_is_inf()):
        return weight_inf()
    elif (b.weight_is_neg_inf()):
        return weight_neg_inf()
    else:
        res = a.w + b.w
        return weight_t(res)

def weight_sub(a, b):
    if (a.weight_is_inf()):
        assert (not b.weight_is_inf()), "inf - inf undefined"
        return weight_inf()
    elif (a.weight_is_neg_inf()):
        assert (not b.weight_is_neg_inf()), "-inf - -inf undefined"
        return weight_neg_inf()
    elif (b.weight_is_inf()):
        return weight_neg_inf()
    elif (b.weight_is_neg_inf()):
        return weight_inf()
    else:
        res = a.w - b.w
        return weight_t(res)

def weight_less(a, b):
    return a.w < b.w
    
def weight_eq(a, b):
    return a.w == b.w
    
