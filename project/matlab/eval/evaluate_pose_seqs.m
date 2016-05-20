% evaluation of the detected_poses against the ground truth.
function pix_error = evaluate_pose_seqs(detected_pose, gt, pck_thresh)

if nargin == 2
	pck_thresh = 15; % default is 15 pix error
end

% first get the annotated poses from the gt corresponding to the images
% used in the detection. 
detected_annotated_poses = get_annotated_poses(detected_pose, gt);

% now transform the detections to the ground truth format.
det = piw_transback(detected_annotated_poses);

pix_error = eval_pix_error(det, gt, pck_thresh);

if length(pix_error)==13
     pix_error = pix_error(1:11); % the last two entries might not be accurate. Note that this was not used in cvpr paper evaluation.
end
end
