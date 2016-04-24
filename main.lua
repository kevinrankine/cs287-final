require 'rnn'
require 'hdf5'
require 'models'

cmd = torch.CmdLine()
cmd:option('-model', 'cbow', 'which model to use')
cmd:option('-d_hid', 100, 'size of rnn hidden state')
cmd:option('-eta', 1e-3, 'learning rate')
cmd:option('-cuda', 0, '1 if use GPU 0 o.w.')

function main()
    local opt = cmd:parse(arg)
    local f = hdf5.open('data/data.hdf5', 'r')
    
    local embeddings = f:read('embeddings'):all():double()
    local corpus = f:read('corpus'):all():long()
    local qs = f:read('qs'):all():long()
    local ps = f:read('ps'):all():long()
    local Qs = f:read('Qs'):all():long()
    if opt.model == 'bow' then
	model = models.CBOW(embeddings, corpus)
	print (score_model(model, qs, ps, Qs))
    elseif opt.model == 'rnn' then
	model = models.LSTMEncoder(embeddings, corpus, opt.d_hid, opt.eta, opt.cuda)
	model:train(qs, ps, Qs)
    end
end

function score_model(model, qs, ps, Qs)
    local precision = 0.0
    for i = 1, qs:size(1) do
	local good = model:similarity(qs[i][1], ps[i][1])
	local bad = 0

	for j = 1, Qs[i]:size(1) do
	    local score = model:similarity(qs[i][1], Qs[i][j])
	    if score > bad then
		bad = score
	    end
	end
	if good > bad then
	    precision = precision + 1
	end
    end
    precision = precision / (qs:size(1))
    return precision
end

main()
