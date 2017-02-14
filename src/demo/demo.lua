require 'paths'
require 'image'
require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'hdf5'
require 'string'
nnlib = nn

paths.dofile('../models/hg.lua')
paths.dofile('predict.lua')
model = torch.load('../../exp/voc/default/model_55.t7')
model:evaluate()

function demo(imgname, bbox)
  img = image.load(imgname)
  output = predict(img,bbox)
  outname = imgname:sub(0,-5)..'_out.jpg'
  image.save(outname, output)
end

demo('1.jpg', torch.Tensor({10,3,550,1440}))
