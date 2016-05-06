require "nn"
require "BatchedMaxMarginCriterion"

crit = nn.BatchedMaxMarginCriterion(0.5, 2)

scores = torch.rand(21 * 2, 1)
ys = torch.DoubleTensor(21 * 2):fill(1)
ys[1] = -1
ys[22] = -1

print (crit:forward(scores))
print (crit:backward(scores, ys))



