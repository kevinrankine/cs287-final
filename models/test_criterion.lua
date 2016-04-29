require 'nn'
require 'MaxMarginCriterion'

crit = nn.MaxMarginCriterion(1)

input = {torch.randn(10), torch.randn(10)}
y = torch.ones(10)

loss = crit:forward(input, y)
print ((input[1] - input[2]):cmul(-y), crit._max_index)
gradInput = crit:backward(input, y)
