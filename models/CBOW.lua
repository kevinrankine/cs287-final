do
    local CBOW = torch.class('models.CBOW')
    function CBOW:__init(embeddings, corpus)
	self.corpus = corpus
	self.model = nn.Sequential()
	LT = nn.LookupTable(embeddings:size(1), embeddings:size(2))
	LT.weight = embeddings
	self.model:add(LT):add(nn.Mean(1))
		       
    end

    function CBOW:similarity(s1, s2)
	local s1 = self.corpus[s1 + 1]
	local s2 = self.corpus[s2 + 1]
	
	local out1 = self.model:forward(s1):clone()
	local out2 = self.model:forward(s2):clone()

	return (out1:dot(out2) / (out1:norm() * out2:norm()))
	
    end

    function CBOW:word_similarity(s1, s2)
	local s1 = self.corpus[s1 + 1]
	local s2 = self.corpus[s2 + 1]
	local score = 0

	local set2 = {}
	
	for i = 1, s2:size(1) do
	    set2[s2[i]] = 1
	end

	for i = 1, s1:size(1) do
	    if set2[s1[i]] then
		score = score + 1
	    end
	end
	score = score / s2:size(1)
	
	return score
    end
end
