require 'nn'

local BatchedMaxMarginCriterion, Parent = torch.class('nn.BatchedMaxMarginCriterion', 
					       'nn.Criterion')

function BatchedMaxMarginCriterion:__init(margin, nbatches)
    Parent:__init(self)
    
    margin = margin or 1
    nbatches = nbatches or 1
    self.nbatches = nbatches
    
    self.margin = margin
    self.maxes = torch.DoubleTensor(nbatches)
    self.max_indices = torch.LongTensor(nbatches)
    self.ngrads = torch.LongTensor(nbatches)
    self.gradInput = torch.zeros(21)
    self.loc_output = torch.DoubleTensor(21):zero()
end

function BatchedMaxMarginCriterion:updateOutput(input, y)
    self.inputs_ = input:split(21)
    for i = 1, #self.inputs_ do
	self.loc_output:zero()
	local local_input = self.inputs_[i]
	
	local pos_scores = local_input[1]:expand(20):copy()
	local neg_scores = local_input:narrow(1, 2, 20):copy()
	local_output:add(self.margin):cadd(neg_scores):csub(pos_scores)
	local_output:cmax(0)
	
	local maximum, max_index = torch.max(local_output, 1)
	
	self.maxes[i] = maximum
	self.max_indices[i] = max_index
	self.ngrads[i] = local_output:gt(0):sum()
	print (self.maxes, self.max_indices, self.ngrads)
    end
    
    self.output = self.maxes:mean()
    return self.output
end

function BatchedMaxMarginCriterion:updateGradInput(input, y)
    self.gradInput:zero()
    
    for i = 1, self.nbatches do
	local mindex = self.max_index[i]
	local mmax = self.maxes[i]
	
	if mmax > 0 then
	    if y[mindex] == 1 then
		self.gradInput[1 + mindex + (i - 1) * 21] = 1
		self.gradInput[1 + (i - 1) * 21] = -1 * self.ngrads[i]
	    end
	end
    end

    return self.gradInput
end
