function boxes = in_frame_detect(count, pyra, unary_map, idpr_map, ...
    num_components, components, apps, nms_thresh, nms_parts)
% Detect at most `count` poses in input using a given CNN-computed feature
% pyramid and model.
%
% The function returns a matrix with one row per detected object.  The
% last column of each row gives the score of the detection.  The
% column before last specifies the component used for the detection.
% Each set of the first 4 columns specify the bounding box for a part.

levels = 1:length(pyra);
boxes = cell(numel(levels), 1);

% Iterate over scales and components,
parfor level = levels,
  % As far as I can tell, num_components is always 1
  for c  = 1:num_components,
    parts    = components{c};
    p_no = length(parts);
    % Local scores
    for p = 1:p_no
      % assign each deformation scores
      parts(p).defMap = idpr_map{level}{p};
      parts(p).appMap = unary_map{level}{p};
      
      f = parts(p).appid;
      % These are just the unaries for each location in the grid. We'll
      % modify the elements of .score in passmsg.
      parts(p).score = parts(p).appMap * apps{f};
      parts(p).level = level;
    end
    
    % Walk from leaves to root of tree, passing message to parent
    for p = p_no:-1:2,
      child = parts(p);
      par = parts(p).parent;
      parent = parts(par);
      cbid = find(child.nbh_IDs == parent.pid);
      pbid = find(parent.nbh_IDs == child.pid);
      
      [msg,parts(p).Ix,parts(p).Iy,parts(p).Im{cbid},parts(par).Im{pbid}] ...
        = passmsg(child, parent, cbid, pbid);
      parts(par).score = parts(par).score + msg;
    end
    
    % Add bias to root score
    parts(1).score = parts(1).score + parts(1).b;
    rscore = parts(1).score;
    
    % Walk back down tree following pointers
    % Chen & Yuille actually applied a score threshold in find, instead of
    % using our silly way of doing things. It really wouldn't be that much
    % effort to implement Ramanan's (much more efficient) diverse n-best
    % algorithm...
    [Y,X] = find(true(size(rscore)));
    if ~isempty(X)
      I   = (X-1)*size(rscore,1) + Y;
      box = backtrack(X,Y,parts,pyra(level));
      numx = length(X); numparts = length(parts);
      box = reshape(box,numx,4*numparts);

      boxes{level} = [box repmat(c,length(I),1) rscore(I)];
    end
  end
end
% Join together detections from each level
boxes = cat(1, boxes{:});
% Sort the detections descending by score, and return only the count best
boxes = boxes(nms_pose(boxes, nms_thresh, nms_parts), :);
[~,idx] = sort(boxes(:,end), 'descend');
trunc_idx = idx(1:min(length(idx), count));
boxes = boxes(trunc_idx,:);

% Backtrack through DP msgs to collect ptrs to part locations
function box = backtrack(x,y,parts,pyra)
numx     = length(x);
numparts = length(parts);

xptr = zeros(numx,numparts);
yptr = zeros(numx,numparts);
box  = zeros(numx,4,numparts);

for k = 1:numparts,
  p   = parts(k);
  if k == 1,
    xptr(:,k) = x;
    yptr(:,k) = y;
  else
    % I = sub2ind(size(p.Ix),yptr(:,par),xptr(:,par),mptr(:,par));
    par = p.parent;
    [h,~,~] = size(p.Ix);
    I = (xptr(:,par)-1)*h + yptr(:,par);
    xptr(:,k) = p.Ix(I);
    yptr(:,k) = p.Iy(I);
    
  end
  scale = pyra.scale;
  x1 = (xptr(:,k) - 1 - double(pyra.padx))*scale+1;
  y1 = (yptr(:,k) - 1 - double(pyra.pady))*scale+1;
  x2 = x1 + double(p.sizx)*scale - 1;
  y2 = y1 + double(p.sizy)*scale - 1;
  box(:,:,k) = [x1 y1 x2 y2];
end