LSTMEncoder = torch.class('models.LSTMEncoder')

function LSTMEncoder:__init(embeddings, corpus, d_hid, eta, gpu)
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

    local model = nn.Sequential()
    local encoder = nn.Sequential()
    
    local LT = nn.LookupTable(nwords, d_in)
    LT.weights = embeddings
    
    encoder:add(LT)
    encoder:add(nn.SplitTable(1))
    encoder:add(nn.Sequencer(nn.GRU(d_in, d_hid)))
    encoder:add(nn.SelectTable(-1))

    local PT = nn.ParallelTable()

    PT:add(encoder):add(encoder:clone())
    model:add(PT)
    
    local criterion = nn.CosineEmbeddingCriterion()

    if gpu ~= 0 then
	require('cutorch')
	require('cunn')
	model:cuda()
	criterion:cuda()
    end

    self.model = model
    self.model_params, self.model_grad_params = self.model:getParameters()
    self.criterion = criterion
end

function LSTMEncoder:truncate(sent)
    local index = 1
    for i = 1, sent:size(1) do
	if sent[i] == self.end_padding then
	    index = i
	    break
	end
    end
    sent = sent:narrow(1, 1, index)
    sent = sent:reshape(sent:size(1), 1)
    
    return sent
end

function LSTMEncoder:update(sent1, sent2, y)
    self.model:forget()
    self.model_grad_params:zero()

    sent1, sent2 = self:truncate(sent1), self:truncate(sent2)
    if self.gpu ~= 0 then
	sent1, sent2 = sent1:cuda(), sent2:cuda()
    end
    local out = self.model:forward({sent1, sent2})
    local loss = self.criterion:forward(out, y)
    local grad_loss = self.criterion:backward(out, y)
    
    self.model:backward({sent1, sent2}, grad_loss)
    self.model:updateParameters(self.eta)
    return loss
end

function LSTMEncoder:train(Xq, Xp, y)
    for i = 1, Xq:size(1) do
	print (self:update(Xq[i], Xp[i], y[i]))
    end
end

function LSTMEncoder:similarity(s1, s2)
    local s1, s2 = self.corpus[s1 + 1], self.corpus[s2 + 1]
end