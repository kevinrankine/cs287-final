require('nn')

do
   local CountModel = torch.class('models.CountModel')
   
   function CountModel:__init(nwords, corpus)
      self.corpus = corpus
      self.doc_dict = {} -- word -> set(doc1, doc2, ...)
      self.doc_counts = torch.LongTensor(corpus:size(1)):zero()
      self.nwords = nwords
      self.idf = torch.DoubleTensor(nwords):zero()
   end

   function CountModel:train()
       local ndocs = 0
       for doc = 1, self.corpus:size(1) do
	   if self.corpus[doc][1] ~= self.nwords then
	       ndocs = ndocs + 1
	       for j = 1, self.corpus:size(2) do
		   word = self.corpus[doc][j]
		   if not self.doc_dict[word] then
		       self.doc_dict[word] = {}
		   end
		   if not self.doc_dict[word][doc] then
		       self.doc_counts[word] = self.doc_counts[word] + 1
		       self.doc_dict[word][doc] = 1
		   end
	       end
	   end
       end
       
       for word = 1, self.nwords do
	   if (self.doc_counts[word] == 0) then
	       self.idf[word] = 0
	   else
	       self.idf[word] = 1 + torch.log(ndocs / self.doc_counts[word])
	   end
       end
   end
   
   function CountModel:similarity(s1, s2)
       local s1 = self.corpus[s1 + 1]
       local s2 = self.corpus[s2 + 1]

       local vec1 = torch.DoubleTensor(self.nwords):zero()
       local vec2 = torch.DoubleTensor(self.nwords):zero()
       
       vec1:indexAdd(1, s1, torch.ones(s1:size(1)))
       vec2:indexAdd(1, s2, torch.ones(s2:size(1)))

       vec1 = vec1:narrow(1, 1, self.nwords - 1)
       vec2 = vec2:narrow(1, 1, self.nwords - 1)
       
       vec1:div(vec1:sum(1)[1])
       vec2:div(vec2:sum(1)[1])
       
       local ndocs = self.corpus:size(1)

       s1_set = {}
       s2_set = {}

       for i = 1, s1:size(1) do
	   if s1[i] < self.nwords and not s1_set[s1[i]]then
	       vec1[s1[i]] = vec1[s1[i]] * self.idf[s1[i]]
	       s1_set[s1[i]] = 1
	   end
	   if s2[i] < self.nwords and not s2_set[s2[i]] then
	       vec2[s2[i]] = vec2[s2[i]] * self.idf[s2[i]]
	       s2_set[s2[i]] = 1
	   end
       end

       return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
   end
end
