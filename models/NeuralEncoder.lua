require('./MaxMarginCriterion')
require('./FixedLookupTable')

NeuralEncoder = torch.class('models.NeuralEncoder')

function NeuralEncoder:__init(model_type, 
			      embeddings, 
			      corpus, 
			      d_hid, 
			      eta, 
			      margin, 
			      gpu, 
			      modelfile,
			     nbatches,
			     dropout)
    self.corpus = corpus
    
    local nwords = embeddings:size(1)
    local d_hid = d_hid
    local d_in = embeddings:size(2)
    local start_padding = nwords
    
    self.nwords = nwords
    self.d_hid = d_hid
    self.d_in = d_in
    self.start_padding = start_padding
    self.eta = eta
    self.gpu = gpu
    self.nbatches = nbatches
    self.dropout = dropout
    
    local model
    local lookup_table

    if modelfile ~= '' then
       model = torch.load(modelfile)
    else
       model = nn.Sequential()
       local left_encoder = nn.Sequential()
       
       lookup_table = FixedLookupTable(nwords, d_in)

       if model_type == 'rnn' then
	   local lstm = nn.GRU(d_in, d_hid)
	   left_encoder:add(lookup_table)
	   left_encoder:add(nn.SplitTable(2))
	   left_encoder:add(nn.Sequencer(lstm))
	   left_encoder:add(nn.SelectTable(-1))
	   if self.dropout > 0 then
	       left_encoder:add(nn.Dropout(self.dropout))
	   end
       elseif model_type == 'cbow' then
	   left_encoder:add(lookup_table)
	   left_encoder:add(nn.Mean(2))
       end

       local PT = nn.ParallelTable()
       local right_encoder = left_encoder:clone('weight', 
						'bias', 
						'gradWeight',
						'gradBias')
       PT:add(left_encoder):add(right_encoder)
       model:add(PT):add(nn.CosineDistance())
       model:remember('neither')
       
       self.left_encoder = left_encoder
       self.right_encoder = right_encoder
    end
    
    local criterion = nn.MaxMarginCriterion(margin, self.nbatches)
    
    if gpu ~= 0 then
	model:cuda()
	criterion:cuda()
    end
    
    self.model = model
    self.model_params, self.model_grad_params = self.model:getParameters()
    self.criterion = criterion

    if modelfile == '' then
	self.model_params:rand(self.model_params:size()):add(-0.5):div(10)
	lookup_table.weight:copy(embeddings)
    end
end

function NeuralEncoder:batch_update(xq, xp, yy)
    function optim_func(params)
	self.model_grad_params:zero()
	self.model_params:copy(params)
	if self.gpu ~= 0 then
	    xq, xp, yy = xq:cuda(), xp:cuda(), yy:cuda()
	end

	local scores = self.model:forward({xq, xp})

	local loss = self.criterion:forward(scores, yy)
	local grad_loss = self.criterion:backward(scores, yy)
	
	self.model:backward({xq, xp}, grad_loss)
	
	return loss, self.model_grad_params
    end

    self.model_grad_params:zero()
    local params, loss = optim.adam(optim_func, 
				   self.model_params, 
				   {learningRate = self.eta})
    loss = loss[1]
    return loss
end

function NeuralEncoder:train(Xq, Xp, y, modelfile)
    self.model:training()
    local modelfile = modelfile or 'model.dat'
    local bsize = 101
    local nbatches = self.nbatches
    
    local total_loss = 0
    for i = 1, Xq:size(1), nbatches * bsize do
	local xq, xp, yy = self:batchify_inputs(Xp, Xq, y, i, nbatches)
	
	local loss = self:batch_update(xq, xp, yy)
	total_loss = total_loss + loss
    end
    
    torch.save(modelfile, self.model)
    return total_loss / (Xq:size(1) / (bsize * nbatches))
end

function NeuralEncoder:renorm_grad(thresh)
    local norm = self.model_grad_params:norm()
    
    if (norm > thresh) then
	self.model_grad_params:div(norm / thresh)
    end
end
	

function NeuralEncoder:similarity(s1, s2)
    local s1, s2 = self.corpus[s1 + 1]:reshape(1, 34):cuda(), self.corpus[s2 + 1]:reshape(1, 34):cuda()
    
    return self.model:forward({s1, s2})[1]
end

function NeuralEncoder:batchify_inputs(Xq, Xp, y, index, nbatches)
    local bsize = 101
    local normal_idx = torch.range(2, bsize):long()
    local bxq, bxp, by
    
    for b = 1, nbatches do
	local idx = torch.randperm(100):long()
	local xq, xp, yy = Xq:narrow(1, index + (b - 1) * bsize, bsize),  Xp:narrow(1, index + (b - 1) * bsize, bsize), y:narrow(1, index + (b - 1) * bsize, bsize)

	xq:indexCopy(1, normal_idx, xq:narrow(1, 2, bsize - 1):index(1, idx)) 
	xp:indexCopy(1, normal_idx, xp:narrow(1, 2, bsize - 1):index(1, idx)) 
	yy:indexCopy(1, normal_idx, yy:narrow(1, 2, bsize - 1):index(1, idx))
	
	xq = xq:narrow(1, 1, 21)
	xp = xp:narrow(1, 1, 21)
	yy = yy:narrow(1, 1, 21)
	
	if bxq and bxp and by then
	    bxq = torch.cat(bxq, xq, 1)
	    bxp = torch.cat(bxp, xp, 1)
	    by = torch.cat(by, yy, 1)
	else
	    bxq = xq
	    bxp = xp
	    by = yy
	end
    end
    
    return bxq, bxp, by
end

