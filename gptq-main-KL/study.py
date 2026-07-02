import torch

a = 0
if (a >= 0):
    print("OK")

oneD=torch.tensor([[1,-3,4,5,5,6,6,0],[1,3,4,5,5,6,6,8],[1,3,4,5,5,6,6,0]])#3×8
a0 = oneD[0]
a1 = oneD[1]
a2 = oneD[2]
xmin = oneD.min(1)[0]
xmax = oneD.max(1)[0]
a0_clamp = torch.clamp(oneD[0], min=-2, max=5)
print(".size()方法：",oneD.size())
print("二维张量最外层元素数：",oneD.size(0))
print("二维张量的维数：",oneD.dim())
