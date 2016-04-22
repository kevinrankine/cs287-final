do
    local BOW = torch.class('BOW')

    function BOW:__init(embeddings, corpus)
	self.corpus = corpus
	self.model = nn.Sequential()
	LT = nn.LookupTable(embeddings:size(1), embeddings:size(2))
	LT.weight = embeddings
	self.model:add(LT):add(nn.Mean(1))
		       
    end

    function BOW:similarity(s1, s2)
	local s1 = self.corpus[s1 + 1]
	local s2 = self.corpus[s2 + 1]
	
	local out1 = self.model:forward(s1):clone()
	local out2 = self.model:forward(s2):clone()

	return (out1:dot(out2) / (out1:norm() * out2:norm()))
	
    end
end
