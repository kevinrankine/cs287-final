require('./MaxMarginCriterion')
require('./FixedLookupTable')

LSTMEncoder = torch.class('models.LSTMEncoder')

function LSTMEncoder:__init(model, embeddings, corpus, d_hid, eta, margin, gpu, modelfile)
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
       local left_encoder = nn.Sequential()
       
       local LT = FixedLookupTable(nwords, d_in)
       LT.weights = embeddings
       
       left_encoder:add(LT)
       left_encoder:add(nn.SplitTable(2))
       left_encoder:add(nn.Sequencer(nn.FastLSTM(d_in, d_hid)))
       left_encoder:add(nn.Sequencer(nn.Dropout(0.3)))
       left_encoder:add(nn.SelectTable(-1))

       local PT = nn.ParallelTable()
       local right_encoder = left_encoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
       PT:add(left_encoder):add(right_encoder)
       
       model:add(PT):add(nn.CosineDistance())
       --[[
       local MLP1 = nn.Sequential()
	   :add(nn.Linear(2 * d_hid, 100))
	   :add(nn.Tanh())
	   :add(nn.Linear(100, 1))
	   :add(nn.Tanh())
	   :add(nn.View(1))
       model:add(PT):add(nn.JoinTable(2)):add(MLP1)
       ]]--
       model:remember('neither')
       
       self.left_encoder = left_encoder
       self.right_encoder = right_encoder
    end
    
    local criterion = nn.MaxMarginCriterion(margin)

    if gpu ~= 0 then
	model:cuda()
	criterion:cuda()
    end
    
    self.model = model
    self.model_params, self.model_grad_params = self.model:getParameters()
    if modelfile == '' then
	self.model_params:rand(self.model_params:size(1)):add(-0.5)
    end
    self.criterion = criterion
end

function LSTMEncoder:update(qs, ps, y)
    self.model:forget()
    self.model_grad_params:zero()
    
    pos_q, pos_p = qs[1]:reshape(1, qs[1]:size(1)), ps[1]:reshape(1, ps[1]:size(1))
    qs, ps, y = qs:narrow(1, 2, qs:size(1) - 1), ps:narrow(1, 2, ps:size(1) - 1), y:narrow(1, 2, y:size(1) - 1)

    if self.gpu ~= 0 then
	pos_q, pos_p, qs, ps, y = pos_q:cuda(), pos_p:cuda(), qs:cuda(), ps:cuda(), y:cuda()
    end
    
    local pos_scores = self.model:forward({pos_q, pos_p}):clone():expand(qs:size(1))
    local neg_scores = self.model:forward({qs, ps})

    local loss = self.criterion:forward({pos_scores, neg_scores}, y)
    local grad_loss = self.criterion:backward({pos_scores, neg_scores}, y)
    
    self.model:backward({qs, ps}, grad_loss[2])
    self.model:forward({pos_q, pos_p})
    self.model:backward({pos_q, pos_p}, grad_loss[1]:sum(1))
    
    self:renorm_grad(1)
    self.model:updateParameters(self.eta)
    return loss
end

function LSTMEncoder:train(Xq, Xp, y, nepochs, modelfile)
    self.model:training()
    local modelfile = modelfile or 'model.dat'
    local nepochs = nepochs or 1
    local bsize = 101 -- 21
    for epoch = 1, nepochs do
	local total_loss = 0
	for i = 1, Xq:size(1), bsize do
	    local idx = torch.randperm(100)
	    local xq, xp, yy = Xq:narrow(1, i, bsize), Xp:narrow(1, i, bsize), y:narrow(1, i, bsize)
	    
	    xq:indexCopy(1, torch.range(2, bsize):long(), xq:narrow(1, 2, bsize - 1):index(1, idx:long())) -- ????????
	    xp:indexCopy(1, torch.range(2, bsize):long(), xp:narrow(1, 2, bsize - 1):index(1, idx:long())) -- ????????
	    yy:indexCopy(1, torch.range(2, bsize):long(), yy:narrow(1, 2, bsize - 1):index(1, idx:long())) -- ????????
	    
	    xq = xq:narrow(1, 1, 21)
	    xp = xp:narrow(1, 1, 21)
	    yy = yy:narrow(1, 1, 21)
	    
	    local loss = self:update(xq, xp, yy)
	    total_loss = total_loss + loss
	    local pct = ((i / Xq:size(1)) * 100)
	    print ("Epoch %d is %.3f percent complete, loss of %.3f" % {epoch, pct, loss})
	end
	torch.save(modelfile, self.model)
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
    local s1, s2 = self.corpus[s1 + 1]:reshape(1, 34):cuda(), self.corpus[s2 + 1]:reshape(1, 34):cuda()
    
    return self.model:forward({s1, s2})[1]
end
