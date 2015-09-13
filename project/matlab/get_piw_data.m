% read the piw annotations. The annotation database file we use here is
% different in format from the one supplied in the original dataset. This
% code generates the mid-keypoints for the lower and upper-limbs.
%
function pos = get_piw_data(name, piw_seqs_path) 
cachedir = './cache/';
cls = [name '_data'];
try 
    load([cachedir cls]);
catch        
    piw_data_path = './data/';
    load([piw_data_path 'piw_dataset.mat']);  % this is slightly different in format from poses_in_the_wild_dataset.mat shared with the dataset.
    inria_pose = piw_dataset;
    
    % lets build the positive training examples.
    inria_seqs = inria_pose.seqs;
    clipindex = 0;
    pos = [];
    numpos = 0;
    for i=1:length(inria_seqs)
        clipindex = clipindex + 1;
        for j=1:length(inria_pose.seqs(i).pose)
            pose_annots = inria_seqs(i).pose;
            numpos = numpos + 1;
            pos(numpos).im = [piw_seqs_path inria_seqs(i).seqname '/' get_framename(pose_annots(j).im)]; 
            pos(numpos).frame = get_framenum(get_framename(pose_annots(j).im)); % some unique number
            pos(numpos).imname = get_framename(pose_annots(j).im); 
            pos(numpos).epi = 1; 
            pos(numpos).clipindex = clipindex;
            parts = pose_annots(j).parts_order;
            [pos(numpos).point, pos(numpos).visible] = fix_part_coordinates(parts, pose_annots(j).point, pose_annots(j).visible);                        
        end
    end

    save([cachedir cls], 'pos');
end
end

function [points, vis] = fix_part_coordinates(parts, annotpoints, visibility)
    partnames = {'head', 'torso', 'larm', 'lhand', 'rarm', 'rhand'};    
    points = -ones(13,2);
    parent = [0 1 2 3 4 5 1 7 8 9 10 1 12];
    vis = zeros(13,1);
    head = 1; lshol=2; lsholmid=3; lelbow=4; lhandmid=5; lwrist=6; rshol=7; rsholmid=8;
    relbow=9; rhandmid=10; rwrist=11; torsomidmid=12; torsomid=13;        
    
    % fix all the visibilities. every joint is shared by two parts. so
    % check if the joint is visible in either part to decide its
    % visibility.
    % hands are not shared with any other joint.
    %
    % left body parts first.
    [st_lhand, en_lhand, v_lhand] = get_part_coors(parts, annotpoints, visibility, 'lhand');
    vis(lwrist) = v_lhand;
    if v_lhand
        points(lwrist, :) = en_lhand;
    end    
               
    [st_larm, en_larm, v_larm] = get_part_coors(parts, annotpoints, visibility, 'larm');
    vis(lelbow) = v_lhand | v_larm; % either of left hand or left arm is visible, then the elbow is deemed visible.
    if vis(lelbow)
       if v_larm
           points(lelbow,:) = en_larm; % if larm is visible, then use the end of it (other end is shoulder)
       else
           points(lelbow,:) = st_lhand; % if larm is not visible, then use the start of left hand (other end is wrist).
       end        
    end    
    vis([lshol]) = v_larm; % if the arm is not visible, then so is the shoulder.
    if v_larm
       points(lshol,:) = st_larm; 
    end
    
    if vis(lelbow) && vis(lwrist)
        points(lhandmid,:) = (points(lwrist,:) + points(lelbow,:))/2;
        vis(lhandmid) = 1;
    end
    
    if vis(lshol)
        points(lsholmid,:) = (points(lshol,:) + points(lelbow,:))/2;
        vis(lsholmid) = 1;
    end
    
        
    % right body parts.
    [st_rhand, en_rhand, v_rhand] = get_part_coors(parts, annotpoints, visibility, 'rhand');
    vis([rwrist]) = v_rhand;
    if v_rhand
        points(rwrist,:) = en_rhand;
    end
    
    [st_rarm, en_rarm, v_rarm] = get_part_coors(parts, annotpoints, visibility, 'rarm');
    vis([relbow]) = v_rhand | v_rarm; % either of right hand or right arm is visible, then the elbow is deemed visible.    
    if vis(relbow)
        if v_rarm
            points(relbow,:) = en_rarm;
        else
            points(relbow,:) = st_rhand;
        end
    end
    vis(rshol) = v_rarm; % if the arm is not visible, then so is the shoulder.            
    if v_rarm
        points(rshol,:) = st_rarm;
    end
    
    if vis(relbow) && vis(rwrist)
        points(rhandmid,:) = (points(rwrist,:) + points(relbow,:))/2;
        vis(rhandmid) = 1;
    end
    
    if vis(rshol)
        points(rsholmid,:) = (points(rshol,:) + points(relbow,:))/2;
        vis(rsholmid) = 1;
    end
    
    
    % if left and right sholders are visible, we will assume the head is
    % also visible.
    [st_head, en_head, v_head] = get_part_coors(parts, annotpoints, visibility, 'head');         
    vis(head) = v_head | (vis(rshol) & vis(lshol));
    if vis(head)
        if v_head
            points(head,:) = en_head; % lower end of head.
        else
            points(head,:) = (points(lshol,:) + points(rshol,:))/2;
        end
    end
    
    [st_torso, en_torso, v_torso] = get_part_coors(parts, annotpoints, visibility, 'torso');        
    vis([torsomid, torsomidmid]) = v_torso;
    if v_torso
        points(torsomid,:) = (st_torso+en_torso)/2;
        points(torsomidmid,:) = (points(torsomid,:) + st_torso')/2;
    end
    
    % make sure the visibility is properly handled
    for k=2:length(parent)
        if vis(k) == 1 
            if ~isempty(find(points(k,:)==-1)) || ~isempty(find(points(parent(k),:)==-1))                
                vis(k) = 0;
            end
        end
    end            
end

function [st, en, vis] = get_part_coors(parts, points, visibility, partname)
for i=1:length(parts)
    if strcmp(parts{i}, partname)       
        coors = points(:,i);
        st = coors([1,3]); en = coors([2,4]);        
        vis = visibility(i);
        %vis(1) = visibility(i);
        %vis(2) = visibility(i);
        return;
    end
end
end

% function fn = get_framenum(imname, clipdir)
% x=imname;
% [a,b]=strtok(x,'/'); b=b(end:-1:2); [a,b]=strtok(b,'.'); b=b(2:end); [a,b]=strtok(b,'-'); str=a(end:-1:1);
% %[a,~]=strtok(imname(end:-1:1),'/'); [~,b]=strtok(a,'.'); str=b(end:-1:2);
% fn = str2double([strrep(clipdir,'-',''), str]);
% end

function framename = get_framename(imname)
    imname = imname(end:-1:1);
    [a,~]=strtok(imname,'/');
    framename = a(end:-1:1);
end

function frameid = get_framenum(imname)
    a = strtok(imname, '.');
    a = strrep(a,'-','');
    a = a(a<'A');
    frameid = str2num(a);
end

