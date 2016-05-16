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
	       encoder:add(nn.Normalize(2)) -- NEW
	   elseif pool == 'mean' then
	       encoder:add(nn.Sequencer(nn.Normalize(2)))
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
	   self.d_hid = d_in
	   lookup_table = nn.LookupTable(nwords, d_in)
	   
	   encoder:add(lookup_table)
	   encoder:add(nn.Mean(2))
       elseif model_type == 'cbow++' then
	   -- expects split input
	   self.d_hid = d_in
	   lookup_table = nn.LookupTable(nwords, d_in)

	   local lookup = nn.Sequential()
	   lookup:add(lookup_table)
	   lookup:add(nn.View(-1, 1, d_in))

	   local gate = nn.Sequential()
	   gate:add(nn.LookupTable(nwords, 1))
	   gate:add(nn.View(-1, 1, 1))
	   gate:add(nn.Sigmoid())

	   local CT = nn.ConcatTable()
	   CT:add(gate):add(lookup)
	   
	   local sub_encoder = nn.Sequential()
	   sub_encoder:add(CT):add(nn.MM())
	   sub_encoder:add(nn.View(-1, d_in))
	   

	   encoder:add(nn.Sequencer(sub_encoder))
	   encoder:add(nn.JoinTable(2))
	   encoder:add(nn.View(-1, seq_len, d_in))
	   encoder:add(nn.Sum(2))
	   encoder:add(nn.Normalize(2))
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

	   local title_encoder = encoder
	   local body_encoder = encoder:clone() -- doesn't share params
	   

	   left_table:add(nn.ParallelTable():add(title_encoder)
	       :add(body_encoder))
	       :add(nn.JoinTable(2))
	       :add(nn.View(-1, self.d_hid, 2))
	       :add(nn.Mean(3))

	   right_table:add(nn.ParallelTable():add(title_encoder:clone('weight', 
								'bias', 
								'gradWeight',
								'gradBias'))
			       :add(body_encoder:clone('weight', 
						       'bias', 
						       'gradWeight',
						       'gradBias')))
	       :add(nn.JoinTable(2))
	       :add(nn.View(-1, self.d_hid, 2))
	       :add(nn.Mean(3))
	   
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
	lookup_table.weight:copy(embeddings)
	self.LT = lookup_table
    end
end

function NeuralEncoder:batch_update(title_xq, title_xp, yy, body_xq, body_xp)
    function optim_func(params)
	self.model_grad_params:zero()
	self.model_params:copy(params)
	if self.body == 0 then
	    if self.gpu ~= 0 then
		xq, xp, yy = title_xq:cuda(), title_xp:cuda(), yy:cuda()
	    else
		xq, xp, yy = title_xq, title_xp, yy
	    end
	else
	    if self.gpu ~= 0 then
		xq, xp, yy = {title_xq:cuda(), body_xq:cuda()}, {title_xp:cuda(), body_xp:cuda()}, yy:cuda()
	    else
		xq, xp, yy = {title_xq, body_xq}, {title_xp, body_xp}, yy
	    end
	end
	local scores = self.model:forward({xq, xp})

	local loss = self.criterion:forward(scores, yy)
	local grad_loss = self.criterion:backward(scores, yy)
	
	self.model:backward({xq, xp}, grad_loss)
	self.LT.gradWeight:zero()
	
	return loss, self.model_grad_params
    end

    local params, loss = optim.adam(optim_func, 
				   self.model_params, 
				   {learningRate = self.eta})
    loss = loss[1]
    return loss
end

function NeuralEncoder:train(title_Xq, title_Xp, y, modelfile, body_Xq, body_Xp)
    self.model:training()
    local modelfile = modelfile or 'model.dat'
    local bsize = 101
    local nbatches = self.nbatches
    
    local total_loss = 0
    local loss
    for i = 1, title_Xq:size(1), nbatches * bsize do
	if self.body == 0 then
	    local title_xq, title_xp, title_yy = self:batchify_inputs(title_Xp, title_Xq, y, i, nbatches)
	    loss = self:batch_update(title_xq, title_xp, title_yy)
	else
	    local title_xq, title_xp, body_xq, body_xp, yy = self:batchify_inputs(title_Xp, title_Xq, y, i, nbatches, body_Xq, body_Xp)
	    loss = self:batch_update(title_xq, title_xp, yy, body_xq, body_xp)
	end
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
	

function NeuralEncoder:batchify_inputs(title_Xq, title_Xp, y, index, nbatches, body_Xq, body_Xp)
    local bsize = 101
    local normal_idx = torch.range(2, bsize):long()
    local title_bxq, title_bxp, body_bxq, body_bxp
    
    for b = 1, nbatches do
	local idx = torch.randperm(100):long()
	local title_xq = title_Xq:narrow(1, index + (b - 1) * bsize, bsize)  
	local title_xp = title_Xp:narrow(1, index + (b - 1) * bsize, bsize)
	if self.body == 1 then
	    body_xq = body_Xq:narrow(1, index + (b - 1) * bsize, bsize)  
	    body_xp = body_Xp:narrow(1, index + (b - 1) * bsize, bsize)
	end
	local yy = y:narrow(1, index + (b - 1) * bsize, bsize)

	title_xq:indexCopy(1, normal_idx, title_xq:narrow(1, 2, bsize - 1):index(1, idx)) 
	title_xp:indexCopy(1, normal_idx, title_xp:narrow(1, 2, bsize - 1):index(1, idx)) 
	if self.body == 1 then
	    body_xq:indexCopy(1, normal_idx, body_xq:narrow(1, 2, bsize - 1):index(1, idx)) 
	    body_xp:indexCopy(1, normal_idx, body_xp:narrow(1, 2, bsize - 1):index(1, idx)) 
	end
	yy:indexCopy(1, normal_idx, yy:narrow(1, 2, bsize - 1):index(1, idx))
	
	title_xq = title_xq:narrow(1, 1, 21)
	title_xp = title_xp:narrow(1, 1, 21)
	if self.body == 1 then
	    body_xq = body_xq:narrow(1, 1, 21)
	    body_xp = body_xp:narrow(1, 1, 21)
	end
	yy = yy:narrow(1, 1, 21)
	
	if title_bxq then
	    title_bxq = torch.cat(title_bxq, title_xq, 1)
	    title_bxp = torch.cat(title_bxp, title_xp, 1)
	    if self.body == 1 then
		body_bxq = torch.cat(body_bxq, body_xq, 1)
		body_bxp = torch.cat(body_bxp, body_xp, 1)
	    end
	    by = torch.cat(by, yy, 1)
	else
	    title_bxq = title_xq
	    title_bxp = title_xp
	    if self.body == 1 then
		body_bxq = body_xq
		body_bxp = body_xp
	    end
	    by = yy
	end
    end
    if self.body == 1 then
	return title_bxq, title_bxp, body_bxq, body_bxp, by
    else
	return title_bxq, title_bxp, by
    end
end

