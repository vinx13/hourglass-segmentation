local M = {}
Dataset = torch.class('voc.Dataset',M)

function Dataset:__init()
    local annot = {}
    local tags = {'name', 'bbox', 'tag'}
    local a = hdf5.open(paths.concat(projectDir,'data/voc.h5'),'r')
    self.nParts = 7
    for _, tag in ipairs(tags) do annot[tag] = a:read(tag):all() end


    -- Index reference
    if not opt.idxRef then
        local allIdxs = torch.range(1,annot.name:size(1))
        opt.idxRef = {}
        opt.idxRef.valid = allIdxs[annot.tag:eq(1)]
        opt.idxRef.train = allIdxs[annot.tag:eq(0)]

        -- torch.save(opt.save .. '/options.t7', opt)
    end

    self.annot = annot
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel()}
    self.a = a

    -- For final predictions
    opt.testIters = self.nsamples.test
    opt.testBatch = 1
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getName(idx)
    return ffi.string(self.annot.name[idx]:char():data())
end

function Dataset:loadImage(idx)
    local fullname = paths.concat(projectDir,'data/VOCdevkit/VOC2010/JPEGImages',self:getName(idx)..'.jpg')
    return image.load(fullname)
end

function Dataset:normalize(idx)
    return self.annot.normalize[idx]
end

function Dataset:getBoundingBox(idx)
    return self.annot['bbox'][idx]
end

function Dataset:getLabel(idx)
    return self.a:read('mask'):read{self:getName(idx)}:all()
end

return M.Dataset
