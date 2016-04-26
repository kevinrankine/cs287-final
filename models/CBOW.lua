do
    local CBOW = torch.class('models.CBOW')
    
    function CBOW:__init(embeddings, corpus, d_hid, eta, cuda)
	self.corpus = corpus
	self.model = nn.Sequential()
	self.d_hid = d_hid
	self.eta = eta
	self.cuda = cuda
	
	local LT = FixedLookupTable(embeddings:size(1), embeddings:size(2))
	LT.weight = embeddings

	local PT = nn.ParallelTable()
	
	local encoder = nn.Sequential()
	encoder:add(LT):add(nn.Mean(2)):add(nn.Linear(200, d_hid))
	PT:add(encoder):add(encoder:clone('weight', 'gradWeight', 'bias', 'gradBias'))
	self.model:add(PT)
	self.criterion = nn.CosineEmbeddingCriterion()
	if self.cuda > 0 then
	    self.model:cuda()
	    self.criterion:cuda()
	end
	self.model_params, self.model_grad_params = self.model:getParameters()
	
    end

    function CBOW:train(Xq, Xp, y)
	local bsize = 21
	for i = 1, Xq:size(1), bsize do
	    local loss = self:update(Xq:narrow(1, i, bsize), Xp:narrow(1, i, bsize), y:narrow(1, i, bsize))
	    print (i / Xq:size(1), loss)
	end
    end


    function CBOW:update(q, p, y)
	self.model_grad_params:zero()
	if (self.cuda > 0) then
	    q = q:cuda()
	    p = p:cuda()
	end
	local embeddings = self.model:forward({q, p})

	local loss = self.criterion:forward(embeddings, -y)
	local grad_loss = self.criterion:backward(embeddings, -y)
	
	self.model:backward({q, p}, grad_loss)
	self.model:updateParameters(self.eta)
	return loss
    end

    function CBOW:similarity(s1, s2)
	local s1 = self.corpus[s1 + 1]
	local s2 = self.corpus[s2 + 1]
	
	local out1 = self.model:forward(s1):clone()
	local out2 = self.model:forward(s2):clone()

	return (out1:dot(out2) / (out1:norm() * out2:norm()))
    end

    function CBOW:word_similarity(s1, s2)
	local s1 = self.corpus[s1 + 1]
	local s2 = self.corpus[s2 + 1]
	local score = 0

	local set2 = {}
	
	for i = 1, s2:size(1) do
	    set2[s2[i]] = 1
	end

	for i = 1, s1:size(1) do
	    if set2[s1[i]] then
		score = score + 1
	    end
	end
	score = score / s2:size(1)
	
	return score
	end
end
