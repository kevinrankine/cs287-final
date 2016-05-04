require 'rnn'
require 'hdf5'
require 'models'

cmd = torch.CmdLine()
cmd:option('-model', 'count', 'which model to use (count, cbow, rnn, cnn)')
cmd:option('-d_hid', 100, 'size of rnn hidden state')
cmd:option('-eta', 1e-3, 'learning rate')
cmd:option('-nepochs', 5, 'number of epochs of training')
cmd:option('-margin', 0.5, 'margin for the loss function')
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
    local dev_qs = f:read('dev_qs'):all():long()
    local dev_ps = f:read('dev_ps'):all():long()
    local dev_Qs = f:read('dev_Qs'):all():long()
    
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
	alt_MRR_score(model, dev_qs, dev_ps, dev_Qs)
    elseif opt.model == 'rnn' or opt.model == 'cbow' then
	model = models.NeuralEncoder(opt.model, 
				     embeddings, 
				     corpus, 
				     opt.d_hid, 
				     opt.eta, 
				     opt.margin, 
				     opt.cuda, 
				     opt.modelfile)
	
	if opt.train ~= 0 then
	    model:train(Xq, Xp, y, opt.nepochs, 'model.dat')
	end
	alt_MRR_score(model, dev_qs, dev_ps, dev_Qs)
    end
end

function MRR_score(model, Xq, Xp, ys)
    local mrr = 0.0
    local bsize = 21
    model.model:evaluate()
    
    for i = 1, Xq:size(1), bsize  do
	local good_score = model.model:forward({Xq[i]:reshape(1, Xq[i]:size(1)):cuda(), 
						Xp[i]:reshape(1, Xp[i]:size(1)):cuda()})[1]
	local rank = 1

	local bad_scores = model.model:forward({Xp:narrow(1, i + 1, bsize - 1):cuda(), 
					       Xq:narrow(1, i + 1, bsize - 1):cuda()})
	for i = 1, bad_scores:size(1) do
	    if good_score < bad_scores[i] then
		rank = rank + 1
	    end
	end
	mrr = mrr + (1 / rank)
	print ("The MRR is %.3f" % (mrr / (i /bsize )))
    end
    mrr = mrr / qs:size(1)
    print ("The MRR is %.3f" % mrr)
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
