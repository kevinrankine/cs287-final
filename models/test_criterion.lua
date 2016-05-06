require "nn"
require "BatchedMaxMarginCriterion"

crit = nn.BatchedMaxMarginCriterion(0.5, 10)

scores = torch.rand(21 * 10, 1)
print (crit:forward(scores))
