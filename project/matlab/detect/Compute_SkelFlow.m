% Perhaps the most important function in this package !
% given a frame-pair, corresponding candidate poses in each frame
% and the optical flow between every consecutive frame pair, this function
% computes the pair-wise cost between every pose pair. 
% It takes the the boxes (output of detect_MM) b1 and b2 corresponding to
% the candidate poses in the first and second frames, image size imsize, 
% optical flow optical_flow, images img1, img2, and several other
% parameters. Returns a matrix of size n1xn2, where n1 is the number of
% candidates from img1 and n2 from img2.
%
function dist = Compute_SkelFlow(b1, b2, imsize, C, opticalflow, img1, img2, ...
    numpts_along_limb, keyjoints, feat_type, parent_pos, parent)
display_poses = 0;
pat_size = 8; % size of the patch used for flow computation.

% number of candidates in img1 and img2
N1 = size(b1,1); N2=size(b2,1);
B1=b1(:,end); B2=b2(:,end); % last column is the pose confidence score

b1 = b1(:,1:end-2); b2 = b2(:,1:end-2); % the last two columns are not keypoint related.
k1 = get_keypoints(b1); k2 = get_keypoints(b2);
if isempty(keyjoints)
    % Keypoints correspond to upper arm midpoint, elbow, lower arm midpoint
    % and wrist on the left (3 4 5 6) and right (8 9 10 11), respectively.
    % TODO: Change this for my model!
    % keyjoints = [3 4 5 6 8 9 10 11];
    keyjoints = [4 5 6 7 12 13 14 15];  
end

% some special processing for head if it exists.
if min(keyjoints)>1 && ~isempty(parent_pos) && sum(parent_pos(:))>0 
    pkey = parent(min(keyjoints))-1; % the min will have the topmost node in the graph. we will take the parent of that node.    
    parent_loc_img1 = [sum(parent_pos(1, pkey*4+[1,3]))/2, sum(parent_pos(1,pkey*4+[2,4]))/2];  % position of parent in img1, 
    parent_loc_img2 = [sum(parent_pos(2, pkey*4+[1,3]))/2, sum(parent_pos(2,pkey*4+[2,4]))/2];  % position of parent in img2.
end

x1 = k1(:,1:2:end); y1 = k1(:,2:2:end);
x2 = k2(:,1:2:end); y2 = k2(:,2:2:end);

% load the skin color (learned from a random set of images from the
% dataset).
persistent skin_color_hist;
if isempty(skin_color_hist)
    load('./data/skin_color_parameters.mat');
end 

% get extra points on the arms.
skelpts1 = get_skeleton_pts(img1, x1, y1, pat_size, keyjoints, numpts_along_limb, parent); 
skelpts2 = get_skeleton_pts(img2, x2, y2, pat_size, keyjoints, numpts_along_limb, parent); 

% compute flow along skeletons.
skelflow = get_motion_from_flow_Ex(skelpts1, opticalflow, keyjoints, numpts_along_limb, display_poses);   
skelclr1 = get_patch_descr_Ex(skelpts1, img1, keyjoints, numpts_along_limb, display_poses);
skelclr2 = get_patch_descr_Ex(skelpts2, img2, keyjoints, numpts_along_limb, display_poses);
    
% % compute the distance of each hist from skin color.
%skelskin1 = get_clrdist_from_skinclr(skelclr1, skin_color_hist, keyjoints, numpts_along_limb);
%skelskin2 = get_clrdist_from_skinclr(skelclr2, skin_color_hist, keyjoints, numpts_along_limb);

f = 1/30;
wri = [7 15]; % this is for 18 part model and corresponds to the wrist joint
wristidx = intersect(wri, keyjoints); % find which joint are we working with here.
if ~isempty(wristidx)
    wrist_smooth = 0;
    for kk=1:length(wristidx)
        lrwrist1 = [x1(:,wristidx(kk)), y1(:,wristidx(kk))]; lrwrist2 = [x2(:,wristidx(kk)), y2(:,wristidx(kk))];                        
        wrist_smooth = wrist_smooth + (f*(mypdist2(lrwrist1, lrwrist2, 'euclidean')));    
    end
else
    wrist_smooth = 0 ;
end
   
skelptsnorm1=box_normalize(skelpts1, imsize);
skelptsnorm2=box_normalize(skelpts2, imsize);
skelflownorm=box_normalize(skelflow, imsize);

% we use an exponential weight for the points on the extended skeleton.
% The points near the wrists are weighted higher.
grad_weight = 1:size(skelpts1,2); grad_weight=exp(grad_weight/size(skelpts1,2)); 
if size(skelpts1,1) == size(skelpts2,1)
    G = repmat(grad_weight, [size(skelpts1,1),1]); G1=G; G2=G;
else
    G1 = repmat(grad_weight, [size(skelpts1,1),1]);
    G2 = repmat(grad_weight, [size(skelpts2,1),1]); 
end
P = mypdist2((skelptsnorm1+skelflownorm).*G1, skelptsnorm2.*G2, 'euclidean');

% compute the color and skin color matrix.
color_matrix = compute_colorhist_dist(skelclr1, skelclr2, keyjoints, numpts_along_limb);
skin_matrix1 = compute_colorhist_dist(skelclr1, ones(size(color_matrix,2),1)*skin_color_hist, keyjoints(end),1); %
skin_matrix2 = compute_colorhist_dist(skelclr2, ones(size(color_matrix,1),1)*skin_color_hist, keyjoints(end),1)'; %, numpts_along_limb)'; %

S = sum(abs(skelflownorm),2); 

% compute the distance of the keypoints from the previous track.
if  min(keyjoints)>1 && ~isempty(parent_pos) && sum(parent_pos(:))>0
    key = min(keyjoints); valid_range = (key-1)*numpts_along_limb*2+1:key*numpts_along_limb*2;
    candidate_pts1 = skelpts1(:,valid_range(1:2)); % either left most or rightmost of skelpts are the ones closest to the parent.
    candidate_pts2 = skelpts2(:,valid_range(1:2));    
    parent_track_score1 = sum(((candidate_pts1-ones(N1,1)*parent_loc_img1)./(ones(N1,1)*[size(img1,1), size(img1,2)])).^2,2); % ||(a-b)/sz||^2, normalize by imsize and compute distance.
    parent_track_score2 = sum(((candidate_pts2-ones(N2,1)*parent_loc_img2)./(ones(N2,1)*[size(img2,1), size(img2,2)])).^2,2);       
    parent_track_score = kron(parent_track_score1, ones(1,N1)) + kron(ones(N2,1), parent_track_score2'); % we do a kron sum of the two scores to create the matrix.    
    
else
    parent_track_score = 0;
end

F=num2cell(feat_type); [alpha, beta, gamma, lambda, delta, xi, zeta] = deal(F{:});
dist = -repmat((alpha*(S)),[1,size(S,1)]) + (beta*P) + gamma*color_matrix + lambda*skin_matrix1 + lambda*skin_matrix2 + delta*wrist_smooth + ...
                    zeta*parent_track_score;
dist_ranking = (-xi*(kron(B1,ones(1,N2)) + kron(ones(1,N1), B2)'));
dist = dist + dist_ranking;
end

function skelpts = get_skeleton_pts(img, x, y, pat_size, keyjoints, numpts_along_limb, parent)
szx = size(img,2); szy=size(img,1);
n = size(x,1);

skelpts = deal(zeros(n,numpts_along_limb*2*length(keyjoints)));
for j=keyjoints
    limb = j;  palimb = parent(j);
    if palimb > 0
        pts = get_pts_along_limb(img, x(:,palimb), y(:,palimb), x(:,limb), y(:,limb), pat_size, szx, szy, numpts_along_limb);    
        skelpts(:,(j-1)*numpts_along_limb*2+1 : j*numpts_along_limb*2) = pts; %cell2mat(pts');       
    else
        pts = get_pts_along_limb(img, x(:,limb), y(:,limb), x(:,limb), y(:,limb), pat_size, szx, szy, 1);    
        skelpts(:,(j-1)*numpts_along_limb*2+1 : j*numpts_along_limb*2) = pts; %cell2mat(pts');
    end
end
end

function C = compute_colorhist_dist(clr1, clr2, keyjoints, numpts_along_limb)
C = 0;
if iscell(clr2)
    for k=1:length(keyjoints)
        for s=1:numpts_along_limb
            X = cumsum(clr1{keyjoints(k),s},2); Y=cumsum(clr2{keyjoints(k),s},2);
            C = C + mypdist2(X, Y, 'euclidean');            
        end
    end    
else
    % we are dealing with skin color hist.    
    for k=1:length(keyjoints)
        for s=1:numpts_along_limb
            X = cumsum(clr1{keyjoints(k),s},2); Y=cumsum(clr2,2);                       
            C = C +  mypdist2(X, Y, 'euclidean');            
        end
    end
end
end

