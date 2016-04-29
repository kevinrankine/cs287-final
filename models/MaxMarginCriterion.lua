local MaxMarginCriterion, Parent = torch.class('nn.MaxMarginCriterion', 
					       'nn.Criterion')

function MaxMarginCriterion:__init(margin)
   Parent:__init(self)
   
   margin = margin or 1
   self.margin = margin
   self.gradInput = {torch.Tensor(1), torch.Tensor(1)}
end

function MaxMarginCriterion:updateOutput(input, y)
   self._output = self._output or input[1]:clone()
   self._output:resizeAs(input[1])
   self._output:copy(input[1])

   self._output:add(-1, input[2])
   self._output:mul(-1):cmul(y)
   self._output:add(self.margin)

   self._output:cmax(0)
   self.output, self._max_index = self._output:max(1)
   
   self.output = self.output[1]
   self._max_index = self._max_index[1]
   return self.output
end

function MaxMarginCriterion:updateGradInput(input, y)
   self.gradInput[1]:resize(input[1]:size())
   self.gradInput[2]:resize(input[1]:size())
   
   self.gradInput[1]:zero()
   self.gradInput[2]:zero()
   self.gradInput[1][self._max_index] = -y[self._max_index]
   self.gradInput[2][self._max_index] = y[self._max_index]

   return self.gradInput
end
