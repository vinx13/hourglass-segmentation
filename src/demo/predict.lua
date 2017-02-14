
function predict(img, bbox)
  local out = img:clone()
  local width = bbox[3] - bbox[1]
  local height = bbox[4] - bbox[2]
  local img = image.crop(img, bbox[1], bbox[2], bbox[3], bbox[4])

  print('Bounding box:')
  print('x1='..bbox[1]..' y1='..bbox[2])
  print('x2='..bbox[3]..' y2='..bbox[4])
  print('Bounding box size: width='..width..' height='..height)
  print('Original image size: width='..out:size()[3]..' height='..out:size()[2])

  img = image.scale(img, '256x256')
  local inp = torch.Tensor(1,3,256,256)
  inp[1] = img
  inp=inp:float()

  local labels = model:forward(inp)
  labels = labels[3] -- get output of last stack
  label = labels[1] -- there is only one sample

  label = image.scale(label, width..'x'..height)
  -- convert one-hot label to mask
  local mask, indices = torch.max(label, 1)
  mask = mask[1]
  indices = indices[1]

  -- draw mask on the original image
  nParts = 7

  local colormap= image.colormap(nParts)
  for i=1,height do
    for j=1,width do
      if mask[{i,j}] > 0.5 then
        color = colormap[indices[{i,j}]]
        idx={{},bbox[2]+i-1,bbox[1]+j-1}
        out[idx] = color
      end -- if
    end -- for j
  end -- for i
  return out
end
