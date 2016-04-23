EncoderDecoder = torch.class('models.EncoderDecoder')

function EncoderDecoder:__init(embeddings, corpus, d_hid, eta)
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

    local encoder = nn.Sequential()
    encoder:add(nn.LookupTable(nwords, d_in))
    encoder:add(nn.SplitTable(1))
    encoder:add(nn.Sequencer(nn.GRU(d_in, d_hid)))
    encoder:add(nn.Sequencer(nn.Linear(d_hid, d_hid)))
    encoder:add(nn.Sequencer(nn.Tanh()))
    encoder:add(nn.SelectTable(-1))
    

    local decoder = nn.Sequential()
    decoder:add(nn.GRU(d_hid, d_hid))
    decoder:add(nn.Linear(d_hid, nwords))
    decoder:add(nn.LogSoftMax())

    local criterion = nn.ClassNLLCriterion()

    self.encoder = encoder
    self.decoder = decoder
    self.encoder_params, self.encoder_grad_params = self.encoder:getParameters()
    self.decoder_params, self.decoder_grad_params = self.decoder:getParameters()
    
    self.criterion = criterion
end

function EncoderDecoder:update(sent)
    self.encoder:forget()
    self.decoder:forget()
    self.encoder_grad_params:zero()
    self.decoder_grad_params:zero()
    
    local index = 1
    for i = 1, sent:size(1) do
	if sent[i] == self.end_padding then
	    index = i
	    break
	end
    end
    
    local sent = sent:narrow(1, 1, index + 1)
    sent = sent:reshape(sent:size(1), 1)
    local hidden_state = self.encoder:forward(sent)

    local outputs = torch.DoubleTensor(sent:size(1), self.nwords)
    local hidden_states = torch.DoubleTensor(sent:size(1), self.d_hid)
    hidden_states[1]:set(hidden_state)

    for i = 1, sent:size(1) do
	local out = self.decoder:forward(hidden_state)
	if (i < sent:size(1)) then
	    outputs[i] = out
	end
    end
    local loss = self.criterion:forward(outputs, sent:reshape(sent:size(1)))
    local grad_loss = self.criterion:backward(outputs, sent:reshape(sent:size(1)))
    local grad_in = self.decoder:backward(hidden_state, grad_loss)
    self.encoder:backward(sent, grad_in)
    self.decoder:updateParameters(self.eta)
    self.encoder:updateParameters(self.eta)
    
    return loss
end

function EncoderDecoder:train()
    for i = 1, self.corpus:size(1) do
	if self.corpus[i][1] ~= 0 then
	    print (self.corpus[i])
	    local loss = self:update(self.corpus[i])
	    print (loss)
	end
    end
end

function EncoderDecoder:similarity(s1, s2)
    local s1, s2 = self.corpus[s1 + 1], self.corpus[s2 + 1]
end
