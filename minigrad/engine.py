import math

class Value:
    def __init__(self, data, label = '', children = ()):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._prev = set(children)
        self._backward = lambda : None
        self._op = None
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __neg__(self):
        return self * -1

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, str(round(other, 2)))
        out = Value(self.data + other.data, self.label + '+' + other.label, (self, other))
        out._op = '+'
        
        def _backward() :
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, str(round(other, 2)))
        out = Value(self.data * other.data, self.label + '*' + other.label, (self, other))
        out._op = '*'
        
        def _backward() :
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, self.label + '**' + str(round(other, 2)), (self,))
        out._op = '**'
        
        def _backward() :
            self.grad += (other * (self.data**(other-1))) * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + (other)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, f'relu({self.label})', (self,))
        out._op = 'relu'
        def _backward() :
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, f'tanh({self})', (self,))
        out._op = 'tanh'
        def _backward() :
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self) :
        topo = []
        visited = set()
        def build_topo(v):
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()
