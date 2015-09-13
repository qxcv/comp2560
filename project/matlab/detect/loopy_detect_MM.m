function boxes = loopy_detect_MM(im1, im2, fim, model, thresh, cycled_nodes, PartIDs, flow_param)
% boxes = detect(im1, im2, fim, model, thresh)
% also uses optical flow between im1 and im2 into the detection of poses in
% consecutive frames simultaneously.
% Detect objects in input using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% The function returns a matrix with one row per detected object.  The
% last column of each row gives the score of the detection.  The
% column before last specifies the component used for the detection.
% Each set of the first 4 columns specify the bounding box for a part

% Compute the feature pyramid and prepare filter
pyra1     = featpyramid(im1,model);
pyra2     = featpyramid(im2,model);
flowpyra     = flowpyramid(im1, im2, fim,model); %, 
interval = model.interval;
levels   = 1:length(pyra1.feat);
thresh = -10;

if nargin<8
    flow_param=1e-4;
end

% Cache various statistics derived from model
[components1,filters1,resp1] = modelcomponents(model,pyra1);
[components2,filters2,resp2] = modelcomponents(model,pyra2);
boxes = []; %zeros(10000,length(components1{1})*4+2);
cnt = 0;

% Iterate over scales and components,
for rlevel = levels,
  for c  = 1:length(model.components),
    parts1    = components1{c};
    parts2    = components2{c};
    numparts = length(parts1);
    boxes_t  = zeros(100000,numparts*4+2);

    [parts1, fl, resp1] = compute_localscore(numparts, parts1, pyra1, filters1, resp1, rlevel, interval, flowpyra);
    [parts2, ~, resp2] = compute_localscore(numparts, parts2, pyra2, filters2, resp2, rlevel, interval, []);    

    %[parts1, rscore, Ik] = cycled_treewalk(numparts, parts1, parts2, flowpyra, cycled_nodes)    
    parts1 = cycled_treewalk(numparts, parts1, parts2, fl, cycled_nodes, flow_param);        
    
    % Add bias to root score
    %parts1(1).score = parts1(1).score + parts1(1).b;
    parts1(1).mm = parts1(1).score + parts1(1).b;
    %[rscore Ik]    = max(parts1(1).score,[],3);

    % Walk down from root to leaves, computing max-marginals.
    for k = 2:numparts
      par1 = parts1(k).parent;
      [msg_d, Ixd, Iyd, Ikd] = passmsg_d(parts1(par1),parts1(k));
      parts1(k).mm = parts1(k).score + msg_d;
      parts1(k).Ixd = Ixd;
      parts1(k).Iyd = Iyd;
      parts1(k).Ikd = Ikd;
    end    
    
    % Backtrack from given partIDs
    for k = PartIDs
      [score mix]= max(parts1(k).mm,[],3);	
      %[Y,X] = find(score > thresh);
      [Y,X] = nms_loop(score,thresh);
      if length(X) > 0
        I = (X-1)*size(score,1) + Y;
        box = backtrack(X,Y,mix(I),k,parts1,pyra1);
        i   = cnt+1:cnt+length(I);
        boxes_t(i,:) = [box repmat(c,length(I),1) score(I)];
        cnt = i(end);
      end
    end
    boxes = cat(1,boxes,unique(boxes_t(1:cnt,:),'rows'));
  end
end

% remove zero entries from the boxes.
zboxes = sum(boxes,2);
boxes(zboxes==0,:) = [];

end

function [Y,X] = nms_loop(score,thresh,r)

% iterate over the following
% 1) find the location of highest score
% 2) suppress (r+1)x(r+1) block around the location

if nargin < 3
  r = 2;
end

Y = zeros(numel(score),1);
X = zeros(numel(score),1);
cnt = 0;
siz = size(score);

[ss ys] = max(score);
[s x]   = max(ss);
y       = ys(x);
while s >= thresh
  cnt = cnt + 1;
  Y(cnt) = y; 
  X(cnt) = x;
  x1 = max(x-r,1);
  y1 = max(y-r,1);
  x2 = min(x+r,size(score,2));
  y2 = min(y+r,size(score,1));
  score(y1:y2,x1:x2) = -inf;
  [ss ys] = max(score);
  [s x]   = max(ss);
  y       = ys(x);
end

Y = Y(1:cnt);
X = X(1:cnt);
end

% Cache various statistics from the model data structure for later use  
function [components,filters,resp] = modelcomponents(model,pyra)
  components = cell(length(model.components),1);
  for c = 1:length(model.components),
    for k = 1:length(model.components{c}),
      p = model.components{c}(k);
      [p.w,p.defI,p.starty,p.startx,p.step,p.level,p.Ix,p.Iy] = deal([]);
      [p.scale,p.level,p.Ix,p.Iy] = deal(0);
      
      % store the scale of each part relative to the component root
      par = p.parent;      
      assert(par < k);
      p.b = [model.bias(p.biasid).w];
      p.b = reshape(p.b,[1 size(p.biasid)]);
      p.biasI = [model.bias(p.biasid).i];
      p.biasI = reshape(p.biasI,size(p.biasid));
      p.sizx  = zeros(length(p.filterid),1);
      p.sizy  = zeros(length(p.filterid),1);
      
      for f = 1:length(p.filterid)
        x = model.filters(p.filterid(f));
        [p.sizy(f) p.sizx(f) foo] = size(x.w);
%         p.filterI(f) = x.i;
      end
      for f = 1:length(p.defid)	  
        x = model.defs(p.defid(f));
        p.w(:,f)  = x.w';
        p.defI(f) = x.i;
        ax  = x.anchor(1);
        ay  = x.anchor(2);    
        ds  = x.anchor(3);
        p.scale = ds + components{c}(par).scale;
        % amount of (virtual) padding to hallucinate
        step     = 2^ds;
        virtpady = (step-1)*pyra.pady;
        virtpadx = (step-1)*pyra.padx;
        % starting points (simulates additional padding at finer scales)
        p.starty(f) = ay-virtpady;
        p.startx(f) = ax-virtpadx;      
        p.step   = step;
      end
      components{c}(k) = p;
    end
  end
  
  resp    = cell(length(pyra.feat),1);
  filters = cell(length(model.filters),1);
  for i = 1:length(filters),
    filters{i} = model.filters(i).w;
  end
end

% Given a 2D array of filter scores 'child',
% (1) Apply distance transform
% (2) Shift by anchor position of part wrt parent
% (3) Downsample if necessary
function [score,Ix,Iy,Ik] = passmsg(child,parent)
  INF = 1e10;
  K   = length(child.filterid);
  Ny  = size(parent.score,1);
  Nx  = size(parent.score,2);  
  [Ix0,Iy0,score0] = deal(zeros([Ny Nx K]));

  for k = 1:K
    [score0(:,:,k),Ix0(:,:,k),Iy0(:,:,k)] = shiftdt(child.score(:,:,k), child.w(1,k), child.w(2,k), child.w(3,k), child.w(4,k),child.startx(k),child.starty(k),Nx,Ny,child.step);
  end

  % At each parent location, for each parent mixture 1:L, compute best child mixture 1:K
  L  = length(parent.filterid);
  N  = Nx*Ny;
  i0 = reshape(1:N,Ny,Nx);
  [score,Iy,Ix,Ik] = deal(zeros(Ny,Nx,L));
  for l = 1:L
    b = child.b(1,l,:);
    [score(:,:,l),I] = max(bsxfun(@plus,score0,b),[],3);
    i = i0 + N*(I-1);
    Ix(:,:,l)    = Ix0(i);
    Iy(:,:,l)    = Iy0(i);
    Ik(:,:,l)    = I;
  end
end

function phi_new = recompute_phi(phi, Fu, Fv, imgr, imgc)
phi_new = zeros(imgr, imgc);
for i=1:size(phi,1)    
    for j=1:size(phi,2)
        u = round(i+Fv(i,j)); v=round(j+Fu(i,j));% i runs the vertical, j horiz.
        u = max(1, min(size(phi,1), u)); v = max(1,min(size(phi,2),v));
        phi_new(u,v) = phi(i,j);
    end
end
end

% pseudo code for the loopy message passing with optical flow.
% initialize m_{e2e1} = 0;
% m_{e1w1} = passmsg(phi(e1), psi(e1,w1));
% m_{w1w2} = passmsg(|phi(w1+flow(w1->w2)) - phi(w2)|+m_{e1w1}, psi(w1,w2))
% m_{w2e2} = passmsg(phi(w2)+m_{w1w2}, -psi(e2,w2))
% m_{e2e1} = passmsg(|phi(e2-flow(e1->e2))-phi(e1)|+m_{w2e2}, psi(e2,e1))
function [score, msg_to_child, Ix, Iy, Ik] = message_loop(w1, e1, w2, e2, F, flow_param_val)
INF = 1e10;
K = length(w1.filterid);
Ny = size(e1.score,1); Nx = size(e1.score,2);
[Ix0, Iy0, score0, msg_to_child] = deal(zeros([Ny, Nx, K]));

% global flow_param_val;
% if isempty(flow_param_val), flow_param_val = 0.7e-4; end
%flow_param_val=1e-2;
[weight_e1e2, weight_e2e1, weight_w1w2, weight_w2w1] = deal([1 0 1 0]*flow_param_val); % 0.7e-4 - large values will make node potentials insignificant.

msg_suffixes = {'w1e1', 'e1e2', 'e2w2', 'w2w1', 'e1w1', 'w1w2', 'w2e2', 'e2e1'};
Fu = F(:,:,1); Fv=F(:,:,2); %opticalflow vectors.
for k=1:K    
    phi_w1 = w1.score(:,:,k); phi_e1=e1.score(:,:,k); phi_w2=w2.score(:,:,k); phi_e2=e2.score(:,:,k);   
    S_w2w1 = zeros(Ny,Nx); S_e2e1 = zeros(Ny,Nx); % some initial messages
    old_msg_diff = INF; % convergence parameter.
    
    % LBP loop count.     
    for lc=1:1000
        % anticlockwise messages
        % msg: w1->e1
        [S_w1e1,~,~] = shiftdt(phi_w1+S_w2w1, w1.w(1,k), w1.w(2,k), w1.w(3,k), w1.w(4,k), w1.startx(k), w1.starty(k), Nx, Ny, w1.step);                
        %[S_w1e1,~,~] = shiftdt(phi_w1+S_w2w1, w1_scale(1), w1_scale(2), w1_scale(3), w1_scale(4), 1, 1, Nx, Ny, w1.step);                
        
        % msg: e1->e2
        M_e1e2 = recompute_phi(phi_e1+S_w1e1, Fu, Fv, Ny,Nx);
        [S_e1e2,~,~] = shiftdt(M_e1e2, weight_e1e2(1), weight_e1e2(2), weight_e1e2(3), weight_e1e2(4), 1, 1, Nx, Ny, e1.step); %e1.startx(k), e1.starty(k),Nx, Ny, e1.step);                
        
        % msg: e2->w2
        [S_e2w2, ~,~] = shiftdt(phi_e2+S_e1e2, w2.w(1,k), w2.w(2,k), w2.w(3,k), w2.w(4,k), w2.startx(k), w2.starty(k), Nx, Ny, e2.step);        
        %[S_e2w2, ~,~] = shiftdt(phi_e2+S_e1e2, e2_scale(1), e2_scale(2), e2_scale(3), e2_scale(4), 1, 1, Nx, Ny, e2.step);        
        
        % msg: w2->w1        
        M_w2w1 = recompute_phi(phi_w2+S_e2w2, -Fu, -Fv, Ny,Nx);
        [S_w2w1,~,~] = shiftdt(M_w2w1, weight_w2w1(1), weight_w2w1(2), weight_w2w1(3), weight_w2w1(4), 1,1, Nx, Ny, w2.step); %w2.startx(k), w2.starty(k),Nx, Ny, w2.step);                
        
        % clockwise messages
        [S_e1w1, ~,~] = shiftdt(phi_e1+S_e2e1, w1.w(1,k), w1.w(2,k), w1.w(3,k), w1.w(4,k), w1.startx(k), w1.starty(k), Nx, Ny, e1.step);        
        %[S_e1w1, ~,~] = shiftdt(phi_e1+S_e2e1, e1_scale(1), e1_scale(2), e1_scale(3), e1_scale(4), 1,1, Nx, Ny, e1.step);        
                
        % msg: w1->w2        
        M_w1w2 = recompute_phi(phi_w1+S_e1w1, Fu, Fv, Ny,Nx);
        [S_w1w2,~,~] = shiftdt(M_w1w2, weight_w1w2(1), weight_w1w2(2), weight_w1w2(3), weight_w1w2(4), 1,1, Nx, Ny, w1.step); %w1.startx(k), w1.starty(k),Nx, Ny, w1.step);                
        
        % msg: w2->e2
        [S_w2e2, ~,~] = shiftdt(phi_w2+S_w1w2, w2.w(1,k), w2.w(2,k), w2.w(3,k), w2.w(4,k), w2.startx(k), w2.starty(k), Nx, Ny, w2.step);        
        %[S_w2e2, ~,~] = shiftdt(phi_w2+S_w1w2, w2_scale(1), w2_scale(2), w2_scale(3), w2_scale(4), 1, 1, Nx, Ny, w2.step);        
        
        % msg: e2->e1        
        M_e2e1 = recompute_phi(phi_e2+S_w2e2, -Fu, -Fv, Ny,Nx);
        [S_e2e1,~,~] = shiftdt(M_e2e1, weight_e2e1(1), weight_e2e1(2), weight_e2e1(3), weight_e2e1(4), 1, 1, Nx, Ny, e2.step); %e2.startx(k), e2.starty(k),Nx, Ny, e2.step);                
        
        if lc>2
            msg_diff = 0;
            for m=1:length(msg_suffixes)/2
                eval(['msg_diff = msg_diff + sum(sum(abs(S_' msg_suffixes{m} '_old-S_' msg_suffixes{m} ')));']);
            end
            %fprintf('msg_diff = %f\n', msg_diff);
            % in min-sum msg passing, the msg difference will never be
            % zero, due to a log-normalization factor. Thus check if the
            % msg_diff is getting constant, to check convergence.
            if abs(old_msg_diff - msg_diff)<1e-5
               %fprintf('converged!\n');
                break;
            end
            old_msg_diff = msg_diff;
        end
        for m=1:length(msg_suffixes)/2
            eval(['S_' msg_suffixes{m} '_old = S_' msg_suffixes{m} ';']); % assign to old variables: S_e1e2_old = S_e1e2; and such
        end                              
    end   
    
%     % now lets fill in the things that we need to pass up the tree.
%     score_e1 = phi_e1 + S_e2e1 + S_w1e1; % marginal at e1.
%     score_w1 = phi_w1 + S_e1w1 + S_w2w1; % marginal at w1.
    
    % we will use the messages and the potentials once more to the parent node from
    % the child node to find the best min-marginals for each pixel as
    % returned by shiftdt. This is the message w1->e1.
    [score0(:,:,k),Ix0(:,:,k),Iy0(:,:,k)] = shiftdt(phi_w1+S_w2w1, w1.w(1,k), w1.w(2,k), w1.w(3,k), w1.w(4,k), w1.startx(k), w1.starty(k), Nx, Ny, w1.step);    
    score0(:,:,k) = score0(:,:,k) + S_e2e1; % the messages to the node e1 are S_w2w1 and S_e2e1.      
    score0(:,:,k) = score0(:,:,k)-mean(mean(score0(:,:,k))); % remove the bias.
    msg_to_child(:,:,k) = S_w2w1; % this is the msg to update the child node.    
end

    % At each parent location, for each parent mixture 1:L, compute best child mixture 1:K
    child=w1; parent = e1;
    L  = length(parent.filterid);
    N  = Nx*Ny;
    i0 = reshape(1:N,Ny,Nx);
    [score,Iy,Ix,Ik] = deal(zeros(Ny,Nx,L));
    for l = 1:L
        b = child.b(1,l,:);
        [score(:,:,l),I] = max(bsxfun(@plus,score0,b),[],3);
        i = i0 + N*(I-1);
        Ix(:,:,l)    = Ix0(i);
        Iy(:,:,l)    = Iy0(i);
        Ik(:,:,l)    = I;
    end
end

% bottom-up tree walks : we will generically represnet the cycle as
% a1->b1->b2->a2.
%numparts, parts1, parts2, flowpyra.flow, [6, 11], rlevel, interval);
function parts1 = cycled_treewalk(numparts, parts1, parts2, optflow, cycled_nodes, flow_param)
    % Walk from leaves to root of tree, passing message to parent    
    for k=numparts:-1:2
        if nnz(cycled_nodes==k) % we have a node to be cycled with its parent
            par1=parts1(k).parent;
            par2=parts2(k).parent;
            %level = rlevel-parts1(k).scale*interval; 
            [parts1(k).msg, ~, parts1(k).Ix, parts1(k).Iy, parts1(k).Ik] = message_loop(parts1(k), parts1(par1), parts2(k), parts2(par2), optflow,flow_param);
            parts1(par1).score = parts1(par1).score + parts1(k).msg; % say the score of the elbow is now phi(elb) + M_e2e1 + M_w1e1, where M_w1e1 = phi(w1) + psi(e1,w1) + M_w2s1.
            %parts1(k).score = parts1(k).score + msg_to_child; % update the child with message from its peer.
        else             
            par = parts1(k).parent;
            [parts1(k).msg,parts1(k).Ix,parts1(k).Iy,parts1(k).Ik] = passmsg(parts1(k),parts1(par));
            parts1(par).score = parts1(par).score + parts1(k).msg;
        end        
    end
end


function box = backtrack(x,y,mix,rt,parts,pyra)

  numx     = length(x);
  numparts = length(parts);
  
  xptr = zeros(numx,numparts);
  yptr = zeros(numx,numparts);
  mptr = zeros(numx,numparts);
  box  = zeros(numx,4,numparts);

  % reorder part indices so that "rt" is new root
  [list downptr] = reorder(rt,parts); 
  for i = 1:length(list)
    k = list(i);
    if k == rt
      xptr(:,k) = x;
      yptr(:,k) = y;
      mptr(:,k) = mix;
    else
      if downptr(i)
        chld = list(i-1);
        p    = parts(chld);
        assert(parts(chld).parent == k);
        [h,w,foo] = size(p.Ixd);
        I   = (mptr(:,chld)-1)*h*w + (xptr(:,chld)-1)*h + yptr(:,chld);
        xptr(:,k) = p.Ixd(I);
        yptr(:,k) = p.Iyd(I);
        mptr(:,k) = p.Ikd(I);
      else
        p   = parts(k);
        par = p.parent;
        [h,w,foo] = size(p.Ix);
        I   = (mptr(:,par)-1)*h*w + (xptr(:,par)-1)*h + yptr(:,par);
        xptr(:,k) = p.Ix(I);
        yptr(:,k) = p.Iy(I);
        mptr(:,k) = p.Ik(I);
      end
    end
    scale = pyra.scale(parts(k).level);
    x1 = (xptr(:,k) - 1 - pyra.padx)*scale+1;
    y1 = (yptr(:,k) - 1 - pyra.pady)*scale+1;
    x2 = x1 + parts(k).sizx(mptr(:,k))*scale - 1;
    y2 = y1 + parts(k).sizy(mptr(:,k))*scale - 1;
    box(:,:,k) = [x1 y1 x2 y2];
  end
  
  box = reshape(box,numx,4*numparts);
end
  
function [list down] = reorder(k,parts)
  
  numparts = length(parts);
  down     = false(1,numparts);
  rem      = true(1,numparts);
  rem(k)   = 0;
  
  list = k;
  par  = parts(k).parent;
  i = 1;
  while par ~= 0
    list = [list par];
    i = i + 1;
    down(i)  = 1;
    rem(par) = 0;
    k = par;
    par = parts(k).parent;
  end
  
  list = [list find(rem)];
end

% compute local scores.
function [parts , flowresp, resp] = compute_localscore(numparts, parts, pyra,filters, resp, rlevel, interval, flowpyra)
    % Local scores    
    for k = 1:numparts,
      f     = parts(k).filterid;
      level = rlevel-parts(k).scale*interval;
      if isempty(resp{level}),
        resp{level} = fconv(pyra.feat{level},filters,1,length(filters));
      end                
      for fi = 1:length(f)
        parts(k).score(:,:,fi) = resp{level}{f(fi)};
      end           
      parts(k).level = level;
    end            
        
    % make the resolution of the flow proper.
    if ~isempty(flowpyra)
        H = zeros(size(filters{1},1)); H(ceil(size(H,1)/2), ceil(size(H,2)/2))=1; % H=[O 0 0;0 1 0; 0 0 0]; something like that.
        flowresp = cat(3, filter2(H, flowpyra.flow{level}(:,:,1), 'valid'),...
                        filter2(H, flowpyra.flow{level}(:,:,2), 'valid'));
    else
        flowresp=[];
    end
end

function [score,Ix,Iy,Ik] = passmsg_d(parent,child)
  K  = length(child.filterid);
  L  = length(parent.filterid);
  Ny = size(child.score,1);
  Nx = size(child.score,2);  
  N  = size(parent.score,1);
  [Ix,Iy,Ik,score] = deal(zeros([Ny Nx K]));
  score0 = parent.mm - child.msg;

  for k = 1:K
    [tmp Ik0] = max(bsxfun(@plus,score0,reshape(child.b(1,:,k),[1 1 L])),[],3);
    [score(:,:,k),Ix(:,:,k),Iy(:,:,k)] = shiftdt(tmp, child.w(1,k), -child.w(2,k), child.w(3,k), -child.w(4,k),2-child.startx(k),2-child.starty(k),Nx,Ny,child.step);
    Ik(:,:,k) = Ik0((Ix(:,:,k)-1)*N+Iy(:,:,k));
  end
end%%
