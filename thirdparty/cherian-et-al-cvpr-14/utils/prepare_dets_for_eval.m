% prepare for evaluation
function detected_pose_seqs = prepare_dets_for_eval(new_merged_poses, frames, seq)
detected_pose_type = struct('seq', {}, 'filename', {}, 'bestpose',{}, 'all_clear', {});
detected_pose_seqs = repmat(detected_pose_type, [1,1,1]);
mov = 3; d=3; % useful if working with multiple movies and multiple sequences.    
for i=1:length(frames)              
    detected_pose_seqs(mov-2, d-2,i).epi = mov;
    detected_pose_seqs(mov-2, d-2,i).seq = seq;
    detected_pose_seqs(mov-2, d-2,i).filename = frames(i).name;            
    detected_pose_seqs(mov-2, d-2,i).frame = get_framenum(frames(i).name);        

    % if we could find a pose sequence path
    if ~isempty(new_merged_poses)            
        detected_pose_seqs(mov-2, d-2,i).bestpose = new_merged_poses(i,:);            
    else
        detected_pose_seqs(mov-2, d-2,i).bestpose = [];
    end
end  