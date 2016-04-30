require('./MaxMarginCriterion')

LSTMEncoder = torch.class('models.LSTMEncoder')

function LSTMEncoder:__init(embeddings, corpus, d_hid, eta, gpu, modelfile)
    self.corpus = corpus
    
    local nwords = embeddings:size(1)
    local d_hid = d_hid
    local d_in = embeddings:size(2)
    local end_padding = nwords
    
    self.nwords = nwords
    self.d_hid = d_hid
    self.d_in = d_in
    self.end_padding = end_padding
    self.eta = eta
    self.gpu = gpu
    local model

    if modelfile ~= '' then
       model = torch.load(modelfile)
    else
       model = nn.Sequential()
       local encoder = nn.Sequential()
       
       local LT = nn.LookupTable(nwords, d_in)
       LT.weights = embeddings
       
       encoder:add(LT)
       encoder:add(nn.SplitTable(2))
       encoder:add(nn.Sequencer(nn.GRU(d_in, d_hid)))
       encoder:add(nn.SelectTable(-1))
       encoder:add(nn.Linear(d_hid, d_hid))
       encoder:add(nn.Tanh())

       local PT = nn.ParallelTable()

       PT:add(encoder):add(encoder:clone('weight', 
					 'gradWeight', 
					 'bias', 
					 'gradBias'))
       model:add(PT):add(nn.CosineDistance())
       model:remember('neither')
    end
    
    local criterion = nn.MaxMarginCriterion(0.1)

    if gpu ~= 0 then
	model:cuda()
	criterion:cuda()
    end

    self.model = model
    self.model_params, self.model_grad_params = self.model:getParameters()
    self.model_params:rand(self.model_params:size(1)):add(-0.5)
    self.criterion = criterion
end

function LSTMEncoder:update(qs, ps, y)
    self.model:forget()
    self.model_grad_params:zero()
    pos_q, pos_p = qs[1]:reshape(1, qs[1]:size(1)), ps[1]:reshape(1, ps[1]:size(1))
    
    qs, ps, y = qs:narrow(1, 2, qs:size(1) - 1), ps:narrow(1, 2, ps:size(1) - 1), y:narrow(1, 2, y:size(1) - 1)

    if self.gpu ~= 0 then
	qs, ps, y = qs:cuda(), ps:cuda(), y:cuda()
    end
    
    local pos_scores = self.model:forward({pos_q, pos_p}):clone():expand(qs:size(1))
    local neg_scores = self.model:forward({qs, ps})

    local loss = self.criterion:forward({pos_scores, neg_scores}, y)
    local grad_loss = self.criterion:backward({pos_scores, neg_scores}, y)
    
    self.model:backward({qs, ps}, grad_loss[2])
    self.model:forward({pos_q, pos_p})
    self.model:backward({pos_q, pos_p}, grad_loss[1]:sum(1))
    
    self:renorm_grad(5)
    self.model:updateParameters(self.eta)
    return loss
end

function LSTMEncoder:train(Xq, Xp, y, nepochs)
   local nepochs = nepochs or 1
   local bsize = 21
   for epoch = 1, nepochs do
      local total_loss = 0
      for i = 1, Xq:size(1), bsize do
	 local loss = self:update(Xq:narrow(1, i, bsize),
				  Xp:narrow(1, i, bsize),
				  y:narrow(1, i, bsize))
	 total_loss = total_loss + loss
	 local pct = ((i / Xq:size(1)) * 100)
	 print (pct, loss)
      end
      torch.save("model.dat", self.model)
      print ("The loss after %d epochs is %.3f" % {epoch, total_loss / (Xq:size(1) / bsize)})
   end
end

function LSTMEncoder:renorm_grad(thresh)
    local norm = self.model_grad_params:norm()
    
    if (norm > thresh) then
	self.model_grad_params:div(norm / thresh)
    end
end
	

function LSTMEncoder:similarity(s1, s2)
    local s1, s2 = self.corpus[s1 + 1]:reshape(1, 38):cuda(), self.corpus[s2 + 1]:reshape(1, 38):cuda()
    
    return self.model:forward({s1, s2})[1]
end
