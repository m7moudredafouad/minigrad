from minigrad.nn import MLP
from minigrad.engine import Value
import torch

def test_auto_grad():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.retain_grad()
    y.backward()
    xpt, ypt = x, y

    assert ymg.data == ypt.data.item()
    assert xmg.grad == xpt.grad.item()
    return

def test_mlp():
    mlp = MLP(3, [4, 4, 1])
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0]

    for i in range(100):
        # forward
        ypred = [mlp(x) for x in xs]
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
        
        # backward
        mlp.zero_grad()
        loss.backward()

        # update
        for p in mlp.parameters():
            p.data += -0.1 * p.grad

    ypred = [mlp(x) for x in xs]
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

    assert loss.data < 0.01
    return

if __name__ == '__main__':
    test_auto_grad()
    test_mlp()