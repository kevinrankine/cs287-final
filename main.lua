require 'rnn'
require 'hdf5'
require 'models'

cmd = torch.CmdLine()
cmd:option('-model', 'count', 'which model to use (count, cbow, rnn, cnn)')
cmd:option('-d_hid', 100, 'size of rnn hidden state')
cmd:option('-eta', 1e-3, 'learning rate')
cmd:option('-nepochs', 5, 'number of epochs of training')
cmd:option('-margin', 0.5, 'margin for the loss function')
cmd:option('-nbatches', 1, 'number of examples in each batch')
cmd:option('-cuda', 0, '1 if use GPU 0 o.w.')
cmd:option('-modelfile', '' ,'File from which to load model')
cmd:option('-train', 1, '1 if train the model 0 o.w.')


function main()
    local opt = cmd:parse(arg)
    local f = hdf5.open('data/data.hdf5', 'r')
    
    local embeddings = f:read('embeddings'):all():double()
    local corpus = f:read('corpus'):all():long()
    
    local train_qs = f:read('train_qs'):all():long()
    local train_ps = f:read('train_ps'):all():long()
    local train_Qs = f:read('train_Qs'):all():long()
    local dev_qs = f:read('dev_qs'):all():long()
    local dev_ps = f:read('dev_ps'):all():long()
    local dev_Qs = f:read('dev_Qs'):all():long()
    
    local train_Xq = f:read('train_Xq'):all():long()
    local train_Xp = f:read('train_Xp'):all():long()
    local train_y = f:read('train_y'):all():double()

    local dev_Xq = f:read('dev_Xq'):all():long()
    local dev_Xp = f:read('dev_Xp'):all():long()
    local dev_y = f:read('dev_y'):all():double()

    if opt.cuda ~= 0 then
	require('cutorch')
	require('cunn')
    end
    
    if opt.model == 'count' then
	model = models.CountModel(embeddings:size(1), corpus)
	model:train(train_Xq, train_Xp, train_y)
    elseif opt.model == 'rnn' or opt.model == 'cbow' then
	model = models.NeuralEncoder(opt.model, 
				     embeddings, 
				     corpus, 
				     opt.d_hid, 
				     opt.eta, 
				     opt.margin, 
				     opt.cuda, 
				     opt.modelfile,
				     opt.nbatches)
	
	if opt.train ~= 0 then
	    local num_ex = train_Xq:size(1) / 101
	    local delim = math.floor(num_ex / opt.nbatches) * 101 * opt.nbatches
	    
	    train_Xq = train_Xq:narrow(1, 1, delim)
	    train_Xp = train_Xp:narrow(1, 1, delim)
	    train_y = train_y:narrow(1, 1, delim)
	    
	    model:train(train_Xq, train_Xp, train_y, opt.nepochs, 'model.dat')
	end
	MRR_score(model, dev_qs, dev_ps, dev_Qs)
    end
end

function MRR_score(model, qs, ps, Qs)
    if model.model then
	model.model:evaluate()
    end
    
    local mrr = 0.0
    local p1 = 0.0
    for i = 1, qs:size(1)  do
	local good_idx = ps[i]:add(1)
	local bad_idx = Qs[i]:add(1)
	
	local num_good = 0
	local num_bad = 20
	for j = 1, ps[i]:size(1) do
	    if ps[i][j] == 1 then
		break
	    end
	    num_good = num_good + 1
	end

	good_idx = good_idx:narrow(1, 1, num_good)
	local xq_good = model.corpus[qs[i][1] + 1]:reshape(1, 34):expand(num_good, 34)
	local xp_good = model.corpus:index(1, good_idx)
	local xq_bad = model.corpus[qs[i][1] + 1]:reshape(1, 34):expand(num_bad, 34)
	local xp_bad = model.corpus:index(1, bad_idx)
	
	local good_score = model.model:forward({xq_good, xp_good}):max()
	local bad_scores = model.model:forward({xq_bad, xp_bad})
	local rank = 1
	
	for i = 1, bad_scores:size(1) do
	    if good_score < bad_scores[i] then
		rank = rank + 1
	    end
	end
	if rank == 1 then
	    p1 = p1 + 1
	end
	mrr = mrr + 1 / rank
    end
    mrr = mrr / qs:size(1)
    p1 = p1 / qs:size(1)
    print ("MRR : %.3f" % mrr)
    print ("P@1 : %.3f" % p1)
end

function alt_MRR_score(model, qs, ps, Qs)
    if model.model then
	model.model:evaluate()
    end
    local mrr = 0.0
    local smrr = 0.0
    local p_1 = 0

    for i = 1, qs:size(1) do
	smrr = 0.0
	local count = 0
	for k = 1, ps[i]:size(1) do
	    if (ps[i][k] == 0) then
		break
	    end
	    
	    count = count + 1
	    local good_score = model:similarity(qs[i][1], ps[i][k])
	    local rank = 1
	    for j = 1, Qs[i]:size(1) do
		if model:similarity(qs[i][1], Qs[i][j]) > good_score then
		    rank = rank + 1
		end
	    end
	    smrr = math.max(smrr, 1 / rank)
	end
	if smrr == 1 then
	    p_1 = p_1 + 1
	end
	mrr = mrr + smrr
	print ("RR : %.3f, %d" % {smrr, count})
    end
    print ("MRR : %.3f" % (mrr / qs:size(1)))
    print ("P@1 : %.3f" % (p_1 / qs:size(1)))
end

main()
