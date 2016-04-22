require 'nn'
require 'hdf5'
require 'BOW'

function main()
    local f = hdf5.open('data/data.hdf5', 'r')
    
    local embeddings = f:read('embeddings'):all():double()
    local corpus = f:read('corpus'):all():long()
    local qs = f:read('qs'):all():long()
    local ps = f:read('ps'):all():long()
    local Qs = f:read('Qs'):all():long()
    
    model = BOW(embeddings, corpus)
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
    print (precision)
end

main()
