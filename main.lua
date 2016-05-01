require 'rnn'
require 'hdf5'
require 'models'

cmd = torch.CmdLine()
cmd:option('-model', 'count', 'which model to use (count, cbow, rnn, cnn)')
cmd:option('-d_hid', 100, 'size of rnn hidden state')
cmd:option('-eta', 1e-3, 'learning rate')
cmd:option('-nepochs', 5, 'number of epochs of training')
cmd:option('-cuda', 0, '1 if use GPU 0 o.w.')
cmd:option('-modelfile', '' ,'File from which to load model')
cmd:option('-train', 1, '1 if train the model 0 o.w.')

function main()
    local opt = cmd:parse(arg)
    local f = hdf5.open('data/data.hdf5', 'r')
    
    local embeddings = f:read('embeddings'):all():double()
    local corpus = f:read('corpus'):all():long()
    local qs = f:read('qs'):all():long()
    local ps = f:read('ps'):all():long()
    local Qs = f:read('Qs'):all():long()
    local Xq = f:read('Xq'):all():long()
    local Xp = f:read('Xp'):all():long()
    local y = f:read('y'):all():double()

    if opt.cuda ~= 0 then
	require('cutorch')
	require('cunn')
    end
    
    if opt.model == 'count' then
	model = models.CountModel(embeddings:size(1), corpus)
	model:train()
	alt_MRR_score(model, qs, ps, Qs)
    elseif opt.model == 'cbow' then
	model = models.CBOW(embeddings, corpus, opt.d_hid, opt.eta, opt.cuda)
	if opt.train ~= 0 then
	    model:train(Xq, Xp, y, opt.nepochs)
	end
	MRR_score(model, ps, qs, Qs)
    elseif opt.model == 'rnn' then
	model = models.LSTMEncoder(opt.model, embeddings, corpus, opt.d_hid, opt.eta, opt.cuda, opt.modelfile)
	if opt.train ~= 0 then
	    model:train(Xq, Xp, y, opt.nepochs)
	end
	MRR_score(model, Xq, Xp, y)
    end
end

function score_model(model, qs, ps, Qs)
    local precision = 0.0
    for i = 1, qs:size(1) do
	local good = model:similarity(qs[i][1], ps[i][1])
	print ("Good: %.3f" % good)
	local bad = 0

	for j = 1, Qs[i]:size(1) do
	    local score = model:similarity(qs[i][1], Qs[i][j])
	    bad = bad + score/(Qs[i]:size(1))
	end
	print ("Bad: %.3f" % bad)
	
	if good > bad then
	    precision = precision + 1
	end

    end
    
    precision = precision / (qs:size(1))

    return precision
end

function MRR_score(model, Xq, Xp, ys)
    local mrr = 0.0
    local bsize = 21
    
    for i = 1, Xq:size(1), bsize  do
	local good_score = model.model:forward({Xq[i]:reshape(1, Xq[i]:size(1)):cuda(), 
						Xp[i]:reshape(1, Xp[i]:size(1)):cuda()})[1]

	print (ys[i])
	local rank = 1

	local bad_scores = model.model:forward({Xp:narrow(1, i + 1, bsize - 1):cuda(), 
					       Xq:narrow(1, i + 1, bsize - 1):cuda()})
	for i = 1, bad_scores:size(1) do
	    if good_score < bad_scores[i] then
		rank = rank + 1
	    end
	end
	print (rank)
	mrr = mrr + (1 / rank)
    end
    mrr = mrr / qs:size(1)
    print (mrr)
end

function alt_MRR_score(model, qs, ps, Qs)
    local mrr = 0.0
    for i = 1, qs:size(1) do
	local good_score = model:similarity(qs[i][1], ps[i][1])
	local rank = 1
	for j = 1, Qs[i]:size(1) do
	    if model:similarity(qs[i][1], Qs[i][j]) > good_score then
		rank = rank + 1
	    end
	end
	mrr = mrr + 1 / rank
	print ("The MRR is %.3f" % (mrr / i))
    end

end

main()
