local M = {}
local DataLoader = torch.class('DataLoader', M)

function DataLoader.create(opt, dataset, ref)
   -- The train and valid loader
   local loaders = {}

   for i, split in ipairs{'train', 'valid'} do
      if opt[split .. 'Iters'] > 0 then
         loaders[split] = M.DataLoader(opt, dataset, ref, split)
      end
   end

   return loaders
end

function DataLoader:__init(opt, dataset, ref, split)
    self.iters = opt[split .. 'Iters']
    self.batchsize = opt[split .. 'Batch']
    self.nsamples = dataset:size(split)
    self.split = split
end

function DataLoader:size()
    return self.iters
end

function DataLoader:run()
    local size = self.iters * self.batchsize

    local idxs = torch.range(1,self.nsamples)
    for i = 2,math.ceil(size/self.nsamples) do
        idxs = idxs:cat(torch.range(1,self.nsamples))
    end
    -- Shuffle indices
    idxs = idxs:index(1,torch.randperm(idxs:size(1)):long())
    -- Map indices to training/validation/test split
    idxs = opt.idxRef[self.split]:index(1,idxs:long())
    idxs = idxs:narrow(1, 1, size)
    local n, idx, sample = 0, 1, nil


    local function loop()
        local bound = math.min(self.batchsize, size - idx + 1)
        if idx >= bound then return nil end
        local indices = idxs:narrow(1, idx, bound)
        local inp,out = loadData(split, indices)
        sample = {inp, out, indices}
        collectgarbage()
        idx = idx + self.batchsize
        n = n + 1
        return n, sample
    end

    return loop
end

return M.DataLoader
