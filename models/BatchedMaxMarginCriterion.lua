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
    self.gradInput = torch.zeros(21 * nbatches)
    self.loc_output = torch.DoubleTensor(21):zero()
end

function BatchedMaxMarginCriterion:updateOutput(input, y)
    self.inputs_ = input:split(21)
    for i = 1, #self.inputs_ do
	self.loc_output:zero()
	local local_input = self.inputs_[i]
	local pos_scores = local_input:narrow(1, 1, 1):expand(21):clone()
	local neg_scores = local_input:clone()
	self.loc_output:add(self.margin):add(neg_scores):csub(pos_scores)
	self.loc_output:cmax(0)
	self.loc_output[1] = 0
	
	local maximum, max_index = torch.max(self.loc_output, 1)
	
	self.maxes[i] = maximum[1]
	self.max_indices[i] = max_index[1]
    end
    self.output = self.maxes:mean()
    return self.output
end

function BatchedMaxMarginCriterion:updateGradInput(input, y)
    self.gradInput:zero()
    
    for b = 1, self.nbatches do
	local mindex = self.max_indices[b]
	local mmax = self.maxes[b]
	
	if mmax > 0 then
	    if y[mindex] == 1 then
		self.gradInput[mindex + (b - 1) * 21] = 1
		self.gradInput[1 + (b - 1) * 21] = -1
	    end
	end
    end

    return self.gradInput
end
