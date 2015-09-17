% returns the joint indices for the recombination tree. Here we combine
% some of the nodes (such as the mid-limbs) returned by Y&R into a single
% limb node when computing the shortest part sequence.
function pose_joints = get_recombination_tree()    
    neckjoint=1; sholjoint=2; elbowjoint=3; wristjoint=4; % pose tree for recombination
    
    % ids and shortest path weights for neck joint
    pose_joints(neckjoint).keyjoints_left = 1;  
    pose_joints(neckjoint).keyjoints_right = 1;
    % XXX: These don't look like they ever get used. What's up here?
    pose_joints(neckjoint).weights = [0 1 0.50 0.0 0.5 1,5];
        
    pose_joints(sholjoint).keyjoints_left = 11; % 7 
    pose_joints(sholjoint).keyjoints_right = 3; % 2 
    pose_joints(sholjoint).weights = [0 1 0.50 0.0 0.5 1,5];
            
    pose_joints(elbowjoint).keyjoints_left = [12 13]; % [8 9]
    pose_joints(elbowjoint).keyjoints_right = [4 5]; % [3 4]
    pose_joints(elbowjoint).weights = [2 1 0.50 0.0 0.1 0.10,5];
        
    pose_joints(wristjoint).keyjoints_left = [14 15]; % [10 11]
    pose_joints(wristjoint).keyjoints_right = [6 7]; % [5 6]
    pose_joints(wristjoint).weights = [2 1 0.01 0.5 0.1 0.05,5];
end
