require('./MaxMarginCriterion')
require('./FixedLookupTable')

NeuralEncoder = torch.class('models.NeuralEncoder')

function NeuralEncoder:__init(model_type, 
			      embeddings, 
			      title_corpus, 
			      body_corpus,
			      d_hid, 
			      eta, 
			      margin, 
			      gpu, 
			      modelfile,
			     nbatches,
			     dropout,
			     kernel_width,
			     pool,
			     body)
    self.title_corpus = title_corpus
    self.body_corpus = body_corpus
    
    local nwords = embeddings:size(1)
    local d_hid = d_hid
    local d_in = embeddings:size(2)
    local start_padding = nwords
    local seq_len = 34
    
    self.nwords = nwords
    self.d_hid = d_hid
    self.d_in = d_in
    self.start_padding = start_padding
    self.eta = eta
    self.gpu = gpu
    self.nbatches = nbatches
    self.dropout = dropout
    self.body = body
    
    local model
    local lookup_table

    if modelfile ~= '' then
       model = torch.load(modelfile)
    else
       model = nn.Sequential()
       local encoder = nn.Sequential()

       if model_type == 'rnn' then
	   lookup_table = FixedLookupTable(nwords, d_in)
	   local lstm = nn.GRU(d_in, d_hid)
	   
	   encoder:add(lookup_table)
	   encoder:add(nn.SplitTable(2))
	   encoder:add(nn.Sequencer(lstm))
	   if self.dropout > 0 then
	       encoder:add(nn.Sequencer(nn.Dropout(self.dropout)))
	   end
	   if pool == 'last' then
	       encoder:add(nn.SelectTable(-1))
	   elseif pool == 'mean' then
	       encoder:add(nn.JoinTable(2))
	       encoder:add(nn.View(-1, seq_len, self.d_hid))
	       encoder:add(nn.Mean(2))
	   else
	       error ("Incorrect pooling supplied")
	   end
       elseif model_type == 'cnn' then
	   lookup_table = FixedLookupTable(nwords, d_in)
	   
	   local conv_layer = nn.Sequential():
	       add(nn.TemporalConvolution(d_in, d_hid, kernel_width, 1)):
	       add(nn.Tanh())
	   
	   if self.dropout > 0 then
	       conv_layer:add(nn.Dropout(self.dropout))
	   end
	   
	   local pooling_layer
	   if pool == 'mean' then
	       pooling_layer = nn.Mean(2)
	   elseif pool == 'last' then
	       pooling_layer = nn.Max(2)
	   else
	       error ("Incorrect pooling supplied")
	   end
	   
	   encoder:add(lookup_table)
	   encoder:add(conv_layer)
	   encoder:add(pooling_layer)
       elseif model_type == 'cbow' then
	   lookup_table = nn.LookupTable(nwords, d_in)
	   encoder:add(lookup_table)
	   encoder:add(nn.Mean(2))
       end
       
       local PT = nn.ParallelTable()
       if body == 0 then
	   PT:add(encoder):add(encoder:clone('weight', 
					  'bias', 
					  'gradWeight',
					  'gradBias'))
       else
	   local left_table = nn.Sequential()
	   local right_table = nn.Sequential()
	   

	   left_table:add(nn.ParallelTable():add(encoder:clone('weight', 
				  'bias', 
				  'gradWeight',
				  'gradBias'))
	       :add(encoder:clone('weight', 
				  'bias', 
				  'gradWeight',
				  'gradBias')))
	       :add(nn.JoinTable(1))
	       :add(nn.View(-1, 2, self.d_hid))
	   --	       :add(nn.Mean(3))

	   right_table:add(nn.ParallelTable():add(encoder:clone('weight', 
				     'bias', 
				     'gradWeight',
				     'gradBias'))
			       :add(encoder:clone('weight', 
				  'bias', 
				  'gradWeight',
				  'gradBias')))
			       :add(nn.JoinTable(1))
			       :add(nn.View(-1, 2, self.d_hid))
	       --:add(nn.Mean(3))

	   PT:add(left_table):add(right_table)
       end
       
       model:add(PT):add(nn.CosineDistance())
       model:remember('neither')
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

function NeuralEncoder:batch_update(title_xq, title_xp, title_yy, body_xq, body_xp)
    function optim_func(params)
	self.model_grad_params:zero()
	self.model_params:copy(params)
	if self.body == 0 then
	    if self.gpu ~= 0 then
		xq, xp, yy = title_xq:cuda(), title_xp:cuda(), title_yy:cuda()
	    end
	else
	    if self.gpu ~= 0 then
		xq, xp, yy = {title_xq:cuda(), body_xq:cuda()}, {title_xp:cuda(), body_xp:cuda()}, title_yy:cuda()
	    end
	end
	print (xq[1]:size())
	local scores = self.model:forward({xq, xp})
	print (scores:size())

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

function NeuralEncoder:train(title_Xq, title_Xp, title_y, modelfile, body_Xq, body_Xp, body_y)
    self.model:training()
    local modelfile = modelfile or 'model.dat'
    local bsize = 101
    local nbatches = self.nbatches
    
    local total_loss = 0
    for i = 1, title_Xq:size(1), nbatches * bsize do
	if self.body == 0 then
	    local title_xq, title_xp, title_yy = self:batchify_inputs(title_Xp, title_Xq, title_y, i, nbatches)
	    local loss = self:batch_update(title_xq, title_xp, title_yy)
	else
	    local title_xq, title_xp, title_yy = self:batchify_inputs(title_Xp, title_Xq, title_y, i, nbatches)
	    local body_xq, body_xp, body_yy = self:batchify_inputs(body_Xp, body_Xq, body_y, i, nbatches)
	    local loss = self:batch_update(title_xq, title_xp, title_yy, body_xq, body_xp, body_yy)
	end
	
	print (loss)
	total_loss = total_loss + loss
    end
    
    torch.save(modelfile, self.model)
    return total_loss / (title_Xq:size(1) / (bsize * nbatches))
end

function NeuralEncoder:renorm_grad(thresh)
    local norm = self.model_grad_params:norm()
    
    if (norm > thresh) then
	self.model_grad_params:div(norm / thresh)
    end
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

