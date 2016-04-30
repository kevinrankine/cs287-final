do
   local CBOW = torch.class('models.CBOW')
   
   function CBOW:__init(embeddings, corpus, d_hid, eta, cuda)
      self.corpus = corpus
      self.model = nn.Sequential()
      self.d_hid = d_hid
      self.eta = eta
      self.cuda = cuda
      
      local LT = nn.LookupTable(embeddings:size(1), embeddings:size(2))
      LT.weight = embeddings

      local PT = nn.ParallelTable()
      
      local encoder = nn.Sequential()
      encoder:add(LT):add(nn.Mean(2))
      
      PT:add(encoder):add(encoder:clone('weight', 'gradWeight', 'bias', 'gradBias'))
      self.model:add(PT)
      self.criterion = nn.CosineEmbeddingCriterion()
      
      if self.cuda > 0 then
	 require('cutorch')
	 require('cunn')
	 self.model:cuda()
	 self.criterion:cuda()
      end
      self.model_params, self.model_grad_params = self.model:getParameters()
      
   end

   function CBOW:train(Xq, Xp, y, nepochs)
      
      local nepochs = nepochs or 5
      local bsize = 21
      for epoch = 1, nepochs do
	  local total_loss = 0
	  for i = 1, Xq:size(1), bsize do
	      loss = self:update(Xq:narrow(1, i, bsize),
				 Xp:narrow(1, i , bsize),
				 y:narrow(1, i, bsize))
	      
	      total_loss = total_loss + loss
	      print (i / Xq:size(1))
	  end
      end
   end


   function CBOW:update(q, p, y)
      self.model_grad_params:zero()
      if (self.cuda > 0) then
	 q = q:cuda()
	 p = p:cuda()
	 y = y:cuda()
      end
      local embeddings = self.model:forward({q, p})

      local loss = self.criterion:forward(embeddings, -y)
      local grad_loss = self.criterion:backward(embeddings, -y)
      
      self.model:backward({q, p}, grad_loss)
      self.model:updateParameters(self.eta)
      return loss
   end

   function CBOW:similarity(s1, s2)
       local s1 = self.corpus[s1 + 1]:reshape(1, self.corpus[s1 + 1]:size(1))
       local s2 = self.corpus[s2 + 1]:reshape(1, self.corpus[s2 + 1]:size(1))

      if self.cuda > 0 then
	 s1 = s1:cuda()
	 s2 = s2:cuda()
      end
      
      local embeddings = self.model:forward({s1, s2})

      return (embeddings[1]:dot(embeddings[2]) / (embeddings[1]:norm() * embeddings[2]:norm()))
   end

   function CBOW:word_similarity(s1, s2)
      local s1 = self.corpus[s1 + 1]
      local s2 = self.corpus[s2 + 1]
      local score = 0
      
      local set1 = {}
      local set2 = {}
      local set1_size = 0
      local set2_size = 0
      
      for i = 1, s1:size(1) do
	 if not set1[s1[i]] then
	    set1_size = set1_size + 1
	    set1[s1[i]] = 1
	 end
	 if not set2[s2[i]] then
	    set2_size = set2_size + 1
	    set2[s2[i]] = 1
	 end
      end

      for k, v in pairs(set1) do
	 if set2[k] then
	    score = score + 1
	 end
      end
      score = score / (set1_size * set2_size)
      return score
   end
end
