% this function specifies the parameters used in tracking and recombination
% of poses. You  might need to play around with these weights over the
% training set to find good combinations. Here are some hints on selecting
% them:
% 1. if there is significant optical flow for a certain body part, then put
% the first two parameters (see defs below) as high. 
% 2. If there is no significant motion, then put the 5th parameter
% relatively high
% 3. If the MRF single image model is quite accurate, then use a high score
% for parameter 6
% 4. If color/skin color is helpful in tracking the parts, then use
% relatively high values for the 3rd and 4th paramters
% 5. Finally, if the parent is detectable with high accuracy, then putting
% a high value for parent link (parameter 7) generally results in good
% detections and recombinations!
%
% For a new pose model, you need to define your own function depending
% on the keyjoints parameter
%
function weights = Get_Distance_Weights(~, keyjoints)
if isempty(keyjoints)
    % Corresponds to midpoint of upper arm (4) down to wrists on left side,
    % then on right.
    keyjoints = [12 13 14 15 4 5 6 7]; % [3 4 5 6 8 9 10 11]; 
end

% weights format:
% weights[1] = optical_flow_paramter
% weights[2] = sumflow (see practical ext.)
% weights[3] = color tracking
% weights[4] = skin color tracking
% weights[5] = motion (see practical ext.)
% weights[6] = MRF unary score
% weights[7] = parent-child link strength in recombination

if keyjoints == 1 % root 
    weights = [0 0 0 0 0 1 0];
elseif any(intersect(keyjoints, [3 11])) % shoulders
    weights = [0 0 0 0 0 1 0];
elseif any(intersect(keyjoints, [5 13])) % elbows
    weights = [0 2 0 0 0.1 1, 0.5];
elseif any(intersect(keyjoints, [7 15])) % wrists
    weights = [2 1 0 0 0.5 1.3 1];
end
end
