% extract the ground truth in posedata for the images in the sequence.
% if you use another dataset, you need to change this function accordingly.
function gt = get_groundtruth_for_seq(files, posedata)
persistent gt_framenum;
if isempty(gt_framenum)
  gt_framenum = arrayfun(@(x) get_framenum(x.imname), posedata); % every frame has a unique name in the dataset.
end

framenum = arrayfun(@(x) get_framenum(x.name), files);

n = length(files);
[gt_idx, frameid] = deal(zeros(n,1));
cnt = 0; 
for i=1:n    
    idx = find(gt_framenum == framenum(i));
    if ~isempty(idx)
        idx = idx(1);    
        cnt = cnt + 1;
        gt_idx(cnt) = idx;                 
    end
end
gt_idx = gt_idx(1:cnt);
gt_idx(gt_idx==0)=[]; % remove null entires if any.
gt = posedata(gt_idx);% we will add the frameid also to gt struct.
end