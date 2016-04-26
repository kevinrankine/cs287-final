require 'nn'
LookupTable, parent = torch.class('FixedLookupTable', 'nn.LookupTable')


function LookupTable:__init(nIndex, nOutput, paddingValue, maxNorm, normType)
    parent.__init(self, nIndex, nOutput)
    
    self:reset()
end

function LookupTable:updateGradInput(input, gradOutput)
end

function LookupTable:accGradParameters(input, gradOutput, scale)
end
