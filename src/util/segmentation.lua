-- Update dimension references to account for intermediate supervision
ref.predDim = {dataset.nParts,5}
ref.outputDim = {}
criterion = nn.ParallelCriterion()
for i = 1,opt.nStack do
    ref.outputDim[i] = {dataset.nParts, opt.outputRes, opt.outputRes}
    criterion:add(nn[opt.crit .. 'Criterion']())
end

-- Function for data augmentation, randomly samples on a normal distribution
local function rnd(x) return math.max(-2*x,math.min(2*x,torch.randn(1)[1]*x)) end

-- Code to generate training samples from raw images
function generateSample(set, idx)
    local img = dataset:loadImage(idx)
    local bbox = dataset:getBoundingBox(idx)
    local label_ = dataset:getLabel(idx)

    label = torch.Tensor(dataset.nParts, label_:size()[1], label_:size()[2]):zero()

    for i = 1,dataset.nParts do
        label[i][label_:eq(i)] = 1
    end

    inp = image.crop(img, bbox[1], bbox[2], bbox[3], bbox[4])
    out = image.crop(label, bbox[1], bbox[2], bbox[3], bbox[4])
    inp = image.scale(inp, opt.inputRes..'x'..opt.inputRes)
    out = image.scale(out, opt.outputRes..'x'..opt.outputRes)

    if set == 'train' then
        -- Flipping and color augmentation
        if torch.uniform() < .5 then
            inp = flip(inp)
            out = shuffleLR(flip(out))
        end
        inp[1]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[2]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[3]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
    end

    return inp,out
end

-- Load in a mini-batch of data
function loadData(set, idxs)
    if type(idxs) == 'table' then idxs = torch.Tensor(idxs) end
    local nsamples = idxs:size(1)
    local input,label

    for i = 1,nsamples do
        local tmpInput,tmpLabel
        tmpInput,tmpLabel = generateSample(set, idxs[i])
        tmpInput = tmpInput:view(1,unpack(tmpInput:size():totable()))
        tmpLabel = tmpLabel:view(1,unpack(tmpLabel:size():totable()))
        if not input then
            input = tmpInput
            label = tmpLabel
        else
            input = input:cat(tmpInput,1)
            label = label:cat(tmpLabel,1)
        end
    end

    if opt.nStack > 1 then
        -- Set up label for intermediate supervision
        local newLabel = {}
        for i = 1,opt.nStack do newLabel[i] = label end
        return input,newLabel
    else
        return input,label
    end
end

function accuracy_(output, label)
    local total = output:numel()
    local true_positive = 0
    local true_negative = 0
    for i=1,output:size()[1] do
      for j=1,dataset.nParts do
        local a=output[i][j]:gt(0.5)
        local b=label[i][j]:gt(0.5)
--        print('a pos = ' .. a:sum())
--        print('b pos = ' .. b:sum())
        local mask = a:eq(b)
        local correct = mask:sum()
--        print('correct = '..correct)
        local true_positive_ = b[mask]:sum()
        -- print('true positive = '.. true_positive_)
        local true_negative_ = correct - true_positive
        -- print('correct = '..correct)
        -- print('true_positive_ = '..true_positive_)
        true_positive = true_positive + true_positive_
        true_negative = true_negative + true_negative_
      end
    end
--    print('total')
--    print('tp'..true_positive)
--    print('tn'..true_negative)
--    print('total'..total)
    local union = total - true_negative
    if union == 0 then return 0.0 else return true_positive / union end
end


function accuracy(output,label)
    if type(output) == 'table' then
        return accuracy_(output[#output],label[#label])
    else
        return accuracy_(output,label)
    end
end
