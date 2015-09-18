%
% eval_pix_error: this function returns the average keypoint localization error for 
% a given error threshold. accuracy = eval_pix_error(det, gt, thresh) takes a 
% pose detection in det, the ground truth in gt, and an error threshold thresh, 
% and returns the average accuracy with this threshold for each keypoint.
% 
% det and gt are matlab array of structures. The array is of length N
% corresponding to the number of images. Each structure element has an
% attribute: 'point' (that is, det(i).point and gt(i).point), which is an nx2
% matrix, where the first and the second columns are the x and y
% coordinates of the keypoints respectively and n is the number of keypoints on
% the pose model. In addition, the structure for gt will hold an additional
% attribute: 'visible' which is a boolean vector of size nx1 and
% corresponds to the visibility of the corresponding keypoint in the ground
% truth. That is, gt(i).visible(k) =  1, if the k-th keypoint is visible
% in the i-th image. 
%
% The parameter 'thresh' is a scalar and corresponds to the pixel radius
% around a ground-truth keypoint inside which a detected keypoint is deemed
% Its default value is 15, corresponding to the 15-pixel error
% used in our paper.
% 
% This function returns a vector 'accuracy' of size nx1, each dimension of
% which corresponds to the average number of times the respective
% keypoints were localized correctly in the entire set of images. For
% symmetic body-parts, we used the maximum of these averages to obtain the
% results reported in our CVPR'14 paper.
%
% Other comments:
% In this evaluation, we assume only one ground truth and one human pose
% detection. 
%
% For questions, queries, or report bugs, please contact Anoop Cherian,
% anoop.cherian@inria.fr.
% 
% 
function pck = eval_pix_error(det, gt, thresh)

if nargin < 3
  thresh = 0.1;
end

assert(numel(det) == numel(gt));

for n = 1:length(gt)
  if isempty(det(n).point)
      fprintf('empty values in det inside eval_pck for test point = %d\n', n);
      continue;
  end  
  dist = sqrt(sum((det(n).point-gt(n).point).^2,2));    
  tp(:,n) = dist <= thresh;
end

% now let us see which all wrists and elbows were visible. 
pck = mean(tp,2)'; 
parts = 1:size(det(1).point,1);
for j=1:length(parts)  
    tpp=[];
    for i=1:length(gt)
        if gt(i).visible(parts(j))            
            tpp=[tpp, tp(parts(j),i)];
        end
    end
    if ~isempty(tpp)
        pck(parts(j)) = mean(tpp);
    end
end
end