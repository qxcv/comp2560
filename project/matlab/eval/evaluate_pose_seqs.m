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

function dap = get_annotated_poses(detected_pose, gt)
detposes = zeros(numel(detected_pose),1);
E = size(detected_pose,1); N = size(detected_pose,2); M = size(detected_pose,3);
for e=1:E
    for d=1:N
        for f=1:M
            if ~isempty(detected_pose(e,d,f).frame)
                idx = sub2ind([E, N, M], e, d, f);
                detposes(idx) = detected_pose(e,d,f).frame;
            end
        end
    end
end

dap = cell(length(gt),1);
for i=1:length(gt)
    idx = find(detposes == gt(i).frame);
    if ~isempty(idx)
        [epi, d, f] = ind2sub([E, N, M], idx);
        try
            dap{i} = detected_pose(epi,d,f).bestpose;
        catch
           fprintf('annot not found for file %s', num2str(gt(i).frame));
        end
    end
end
end