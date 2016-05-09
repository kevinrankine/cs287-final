require 'rnn'
require 'hdf5'
require 'models'
require 'optim'

cmd = torch.CmdLine()
cmd:option('-model', 'count', 'which model to use (count, cbow, rnn, cnn)')
cmd:option('-d_hid', 100, 'size of rnn hidden state')
cmd:option('-eta', 1e-3, 'learning rate')
cmd:option('-nepochs', 5, 'number of epochs of training')
cmd:option('-margin', 0.5, 'margin for the loss function')
cmd:option('-nbatches', 1, 'number of examples in each batch')
cmd:option('-dropout', 0, 'dropout value to use (0 if no dropout)')
cmd:option('-pool', 'last', 'state aggregation to use [mean or last]')
cmd:option('-kernel_width', 3, 'kernel width for the CNN')
cmd:option('-cuda', 0, '1 if use GPU 0 o.w.')
cmd:option('-from_file', '' ,'File from which to load model')
cmd:option('-to_file', 'model.dat', 'File to save model to')
cmd:option('-train', 1, '1 if train the model 0 o.w.')
cmd:option('-body', 0, '1 if use body else 0')



function main()
    local opt = cmd:parse(arg)
    local f = hdf5.open('data/data.hdf5', 'r')
    
    local embeddings = f:read('embeddings'):all():double()
    local title_corpus = f:read('title_corpus'):all():long()
    local body_corpus = f:read('body_corpus'):all():long()
    
    local train_qs = f:read('train_qs'):all():long()
    local train_ps = f:read('train_ps'):all():long()
    local train_Qs = f:read('train_Qs'):all():long()
    local dev_qs = f:read('dev_qs'):all():long()
    local dev_ps = f:read('dev_ps'):all():long()
    local dev_Qs = f:read('dev_Qs'):all():long()
    
    local title_train_Xq = f:read('title_train_Xq'):all():long()
    local title_train_Xp = f:read('title_train_Xp'):all():long()
    local title_train_y = f:read('title_train_y'):all():double()

    local title_dev_Xq = f:read('title_dev_Xq'):all():long()
    local title_dev_Xp = f:read('title_dev_Xp'):all():long()
    local title_dev_y = f:read('title_dev_y'):all():double()
    
    local body_train_Xq = f:read('body_train_Xq'):all():long()
    local body_train_Xp = f:read('body_train_Xp'):all():long()
    local body_train_y = f:read('body_train_y'):all():double()

    local body_dev_Xq = f:read('body_dev_Xq'):all():long()
    local body_dev_Xp = f:read('body_dev_Xp'):all():long()
    local body_dev_y = f:read('body_dev_y'):all():double()


    if opt.cuda ~= 0 then
	require('cutorch')
	require('cunn')
    end
    
    if opt.model == 'count' then
	model = models.CountModel(embeddings:size(1), corpus)
	model:train(title_train_Xq, title_train_Xp, title_train_y)
    elseif opt.model == 'rnn' or opt.model == 'cbow' or opt.model == 'cnn' then
	model = models.NeuralEncoder(opt.model, 
				     embeddings, 
				     title_corpus, 
				     body_corpus,
				     opt.d_hid, 
				     opt.eta, 
				     opt.margin, 
				     opt.cuda, 
				     opt.from_file,
				     opt.nbatches,
				     opt.dropout,
				     opt.kernel_width,
				     opt.pool,
				     opt.body)
	
	if opt.train ~= 0 then
	    local num_ex = title_train_Xq:size(1) / 101
	    local delim = math.floor(num_ex / opt.nbatches) * 101 * opt.nbatches
	    
	    title_train_Xq = title_train_Xq:narrow(1, 1, delim)
	    title_train_Xp = title_train_Xp:narrow(1, 1, delim)
	    title_train_y = title_train_y:narrow(1, 1, delim)

	    body_train_Xq = body_train_Xq:narrow(1, 1, delim)
	    body_train_Xp = body_train_Xp:narrow(1, 1, delim)
	    body_train_y = body_train_y:narrow(1, 1, delim)
	    
	    MRR_score(model, dev_qs, dev_ps, dev_Qs, opt.body)
	    for epoch = 1, opt.nepochs do
		local loss = model:train(title_train_Xq, 
					 title_train_Xp, 
					 title_train_y, 
					 opt.to_file,
					 body_train_Xq,
					 body_train_Xq,
					 body_train_y)
		print ("Loss after %d epochs: %.3f" % {epoch, loss})
	    end
	end
    end
end

function MRR_score(model, qs, ps, Qs, body)
    if model.model then
	model.model:evaluate()
    end
    
    local mrr = 0.0
    local p1 = 0.0
    for i = 1, qs:size(1)  do
	local good_idx = ps[i]:clone():add(1)
	local bad_idx = Qs[i]:clone():add(1)
	
	local num_good = 0
	local num_bad = 20
	for j = 1, ps[i]:size(1) do
	    if ps[i][j] == 0 then
		break
	    end
	    num_good = num_good + 1
	end

	good_idx = good_idx:narrow(1, 1, num_good)
	if body == 0 then
	    local title_xq_good = model.title_corpus[qs[i][1] + 1]:view(1, -1):expand(num_good, 34)
	    local title_xp_good = model.title_corpus:index(1, good_idx)
	    local title_xq_bad = model.title_corpus[qs[i][1] + 1]:view(1, -1):expand(num_bad, 34)
	    local title_xp_bad = model.title_corpus:index(1, bad_idx)
	    
	    good_score = model.model:forward({title_xq_good, title_xp_good}):max()
	    bad_scores = model.model:forward({title_xq_bad, title_xp_bad})
	else
	    local title_xq_good = model.title_corpus[qs[i][1] + 1]:view(1, -1):expand(num_good, 34)
	    local title_xp_good = model.title_corpus:index(1, good_idx)
	    local title_xq_bad = model.title_corpus[qs[i][1] + 1]:view(1, -1):expand(num_bad, 34)
	    local title_xp_bad = model.title_corpus:index(1, bad_idx)

	    local body_xq_good = model.body_corpus[qs[i][1] + 1]:view(1, -1):expand(num_good, 100)
	    local body_xp_good = model.body_corpus:index(1, good_idx)
	    body_xq_bad = model.body_corpus[qs[i][1] + 1]:view(1, -1):expand(num_bad, 100)
	    body_xp_bad = model.body_corpus:index(1, bad_idx)
	    
	    good_score = model.model:forward({{title_xq_good, body_xq_good}, {title_xp_good, body_xp_good}}):max()
	    bad_scores = model.model:forward({{title_xq_bad, title_xq_bad}, {title_xp_bad, body_xp_bad}})
	end
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
