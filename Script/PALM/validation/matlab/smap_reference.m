% Copyright (C) 2026 Wenlan Yu
% SPDX-License-Identifier: GPL-3.0-or-later
%
% SMAP_REFERENCE  Run the SMAP NPC workflow in MATLAB on an existing _sml.mat and
% export every stage, as the reference for validation/run_validation.py.
%
%   smap_reference('<run folder>\config.json')
%
% Stages exported to <run folder>\matlab_reference\:
%   S2 load       load sml, group, layer-1 filters    -> S2_loaded.mat
%   S4 segment    makeCellGrid, cells, segmentNPC     -> S4_cells.mat, S4_sites.mat
%   S5 cleanup    NPCsegmentCleanup on every site     -> S5_cleanup.mat
%   S6 locmofit   LocMoFit on every accepted site     -> S6_<model>.mat
% Requires SMAP (cfg.smap_dir). Progress and errors: matlab_log.txt, status.json.
function status = smap_reference(configfile)

if nargin < 1 || isempty(configfile)
    [f, p] = uigetfile('config.json', 'Select config.json of the validation run');
    if isequal(f, 0)
        error('smap_reference needs the config.json written by run_validation.py');
    end
    configfile = fullfile(p, f);
end
cfg = jsondecode(fileread(configfile));
outdir = fullfile(cfg.output_dir, 'matlab_reference');
if ~exist(outdir, 'dir'); mkdir(outdir); end
logfile = fullfile(outdir, 'matlab_log.txt');
if exist(logfile, 'file'); delete(logfile); end
diary(logfile); diary on
cleanupObj = onCleanup(@() diary('off'));

status = struct();
status.started = datestr(now, 'yyyy-mm-dd HH:MM:SS');
status.matlab_version = version;
status.computer = computer;
fprintf('==== SMAP reference run: %s\n', status.started);
fprintf('MATLAB %s on %s\n', status.matlab_version, status.computer);
fprintf('config: %s\n', configfile);

g = [];
try
    t0 = tic;
    cd(cfg.smap_dir);
    addpath(genpath(fullfile(cfg.smap_dir, 'shared')));
    addpath(genpath(fullfile(cfg.smap_dir, 'LocMoFit')));
    set(0, 'DefaultFigureVisible', 'off');
    set(0, 'DefaultFigureCreateFcn', '');
    set(0, 'DefaultFigureWindowStyle', 'normal');
    global SMAP_stopnow %#ok<TLEV>
    SMAP_stopnow = false;
    g = gui.GuiMainSMAP;
    g.makeGui;
    try g.handle.Visible = 'off'; catch; end
    status.init = stage_ok(t0);
catch err
    status.init = stage_err(err);
    write_status(outdir, status);
    return
end

smlfile = cfg.input_sml;
if isempty(smlfile) || ~exist(smlfile, 'file')
    fprintf('Localization file not found: %s\n', smlfile);
    status.S2_load = struct('ok', false, 'error', 'input_sml not found');
    write_status(outdir, status);
    return
end

try
    t0 = tic;
    load_and_filter(g, cfg, smlfile);
    loc = g.locData.loc; %#ok<NASGU>
    grouploc = g.locData.grouploc; %#ok<NASGU>
    sel = g.locData.getloc({'xnm', 'ynm', 'frame', 'locprecnm', 'PSFxnm', 'LLrel', 'phot', ...
        'numberInGroup', 'filenumber'}, 'layer', 1, 'position', 'all'); %#ok<NASGU>
    filtertable = layer_filter_state(g); %#ok<NASGU>
    status.layers_on = find(g.getPar('sr_layerson'));
    fprintf('render layers on: %s (drift correction and cell images use all layers that are on)\n', ...
        mat2str(status.layers_on));
    save(fullfile(outdir, 'S2_loaded.mat'), 'loc', 'grouploc', 'sel', 'filtertable', '-v7');
    fprintf('S2: %d localizations, %d grouped, %d in layer 1\n', numel(g.locData.loc.xnm), ...
        numel(g.locData.grouploc.xnm), numel(sel.xnm));
    status.S2_load = stage_ok(t0);
catch err
    status.S2_load = stage_err(err);
    write_status(outdir, status);
    return
end
write_status(outdir, status);

try
    t0 = tic;
    [cells, sites] = run_segmentation(g, cfg); %#ok<ASGLU>
    save(fullfile(outdir, 'S4_cells.mat'), 'cells', '-v7');
    save(fullfile(outdir, 'S4_sites.mat'), 'sites', '-v7');
    fprintf('S4: %d cells, %d sites\n', numel(cells), numel(sites));
    status.S4_segment = stage_ok(t0);
catch err
    status.S4_segment = stage_err(err);
    write_status(outdir, status);
    return
end
write_status(outdir, status);

ev = [];
try
    t0 = tic;
    [cleanup, ev] = run_cleanup(g, cfg); %#ok<ASGLU>
    save(fullfile(outdir, 'S5_cleanup.mat'), 'cleanup', '-v7');
    fprintf('S5: %d of %d sites accepted\n', sum([cleanup.use]), numel(cleanup));
    status.S5_cleanup = stage_ok(t0);
catch err
    status.S5_cleanup = stage_err(err);
end
write_status(outdir, status);

lcfgs = cfg.locmofit;
if isstruct(lcfgs); lcfgs = num2cell(lcfgs); end
for c = 1:numel(lcfgs)
    lc = lcfgs{c};
    name = matlab.lang.makeValidName(['S6_' lc.name]);
    try
        t0 = tic;
        res = run_locmofit(g, cfg, lc, ev); %#ok<NASGU>
        save(fullfile(outdir, [name '.mat']), 'res', '-v7');
        status.(name) = stage_ok(t0);
    catch err
        status.(name) = stage_err(err);
    end
    write_status(outdir, status);
end
status.finished = datestr(now, 'yyyy-mm-dd HH:MM:SS');
write_status(outdir, status);
fprintf('==== finished %s\n', status.finished);
end

function setpopup(m, field, sel)
pm = m.getGuiParameters;
h = pm.(field);
idx = find(strcmp(strtrim(h.String), sel), 1);
if isempty(idx); error('option %s not found for %s', sel, field); end
h.Value = idx;
m.setGuiParameters(struct(field, h));
end

function s = simplify(p)
s = struct();
fns = fieldnames(p);
for k = 1:numel(fns)
    v = p.(fns{k});
    if isnumeric(v) || islogical(v)
        if numel(v) < 100; s.(fns{k}) = double(v); end
    elseif ischar(v)
        s.(fns{k}) = v;
    elseif isstruct(v) && isfield(v, 'selection')
        s.(fns{k}) = v.selection;
    end
end
end

function load_and_filter(g, cfg, smlfile)
[p, f, e] = fileparts(smlfile);
g.children.guiFile.loadbutton_callback([], [], 0, [p filesep], [f e]);
g.setPar('group_dx', cfg.grouping.dx);
g.setPar('group_dt', cfg.grouping.dt);
g.setPar('group_mode', 1);
try
    g.children.guiFile.setGuiParameters(struct('group_dx', cfg.grouping.dx, 'group_dt', cfg.grouping.dt));
catch
end
g.locData.regroup(cfg.grouping.dx, cfg.grouping.dt, 1);
try
    g.setPar('layer1_groupcheck', logical(cfg.layer1.grouped));
catch err
    fprintf('note: could not set layer1 groupcheck: %s\n', err.message);
end
fns = fieldnames(cfg.layer1.filters);
for k = 1:numel(fns)
    v = double(cfg.layer1.filters.(fns{k}));
    v(v > 1e299) = inf;
    v(v < -1e299) = -inf;
    g.locData.setPar('layer1_selectedField', {fns{k}, v(1), v(2), 1, 1});
end
g.locData.filter;
fprintf('layer 1 grouped: %d\n', g.locData.isgrouped(1));
end

function ft = layer_filter_state(g)
ft = struct();
try
    s = g.locData.getPar('filtertable', 'layer', 1);
    for k = 1:size(s, 1)
        if s{k, 7}
            ft.(s{k, 1}) = [s{k, 2}, s{k, 6}];
        end
    end
catch err
    ft.error = err.message;
end
end

function m = openplug(g, pth)
m = plugin(pth{:});
m.handle = figure('Visible', 'off', 'MenuBar', 'none', 'Toolbar', 'none', 'Name', pth{end});
m.attachPar(g.P);
m.attachLocData(g.locData);
pp.Vrim = 100;
pp.Xrim = 10;
m.setGuiAppearence(pp);
m.makeGui;
end

function set_render(g, cfg)
L = g.children.guiRender.children.Layer1;
setpopup(L, 'rendermode', 'Gauss');
setpopup(L, 'intensitycoding', 'normal');
setpopup(L, 'lut', cfg.render.lut);
L.setGuiParameters(struct('colorauto', true));
L.setGuiParameters(struct('remout', false));
L.setGuiParameters(struct('lutinv', false));
if strcmp(cfg.render.colormode, 'field')
    hf = L.guihandles.renderfield;
    if ~any(strcmp(hf.String, cfg.render.renderfield))
        hf.String = fieldnames(g.locData.loc);
    end
    setpopup(L, 'renderfield', cfg.render.renderfield);
end
setpopup(L, 'render_colormode', cfg.render.colormode);
L.updateLayerField;
h = L.guihandles.render_colormode;
cb = h.Callback;
cb{1}(h, [], cb{2:end});
L.updateLayerField;
pl = g.getPar('layer1_');
fprintf('layer 1 render (as used by the renderer): %s, colormode %s, field %s, lut %s, colour range [%g %g]\n', ...
    selname(pl.rendermode), selname(pl.render_colormode), selname(pl.renderfield), ...
    selname(pl.lut), double(pl.colorfield_min), double(pl.colorfield_max));
if strcmp(cfg.render.colormode, 'field') && ~strcmp(selname(pl.lut), cfg.render.lut)
    error('could not set the LUT of layer 1 to %s (it is %s)', cfg.render.lut, selname(pl.lut));
end
end

function s = selname(v)
if isstruct(v) && isfield(v, 'selection'); s = v.selection; elseif ischar(v); s = v; else; s = '?'; end
end

function [cells, sites] = run_segmentation(g, cfg)
set_render(g, cfg);
gs = g.children.guiSites;
gs.make_siteexplorer;
se = g.locData.SE;
prev = se.processors.preview;
seset = struct('se_cellpixelsize', cfg.se.cell_pixelsize, 'se_cellfov', cfg.se.cell_fov, ...
    'se_sitepixelsize', cfg.se.site_pixelsize, 'se_sitefov', cfg.se.site_fov, ...
    'se_siteroi', cfg.se.site_roi);
try
    gs.children.settings.setGuiParameters(seset);
catch err
    fprintf('note: %s\n', err.message);
end
fns = fieldnames(seset);
for k = 1:numel(fns)
    g.setPar(fns{k}, seset.(fns{k}));
    fprintf('%s = %g\n', fns{k}, g.getPar(fns{k}));
end
prev.clearall;
mc = getplug(g, 'Segment', 'makeCellGrid');
mc.run(mc.getAllParameters);
prev.redrawall('cells');
cells = struct('ID', {}, 'pos', {}, 'filenumber', {}, 'image', {}, 'image_sum', {}, 'rangex', {}, ...
    'rangey', {}, 'pixrec', {}, 'numberOfLocs', {}, 'nlayers', {});
for k = 1:numel(se.cells)
    c = se.cells(k);
    cells(k).ID = double(c.ID);
    cells(k).pos = double(c.pos(:)');
    cells(k).filenumber = double(c.info.filenumber);
    ri = c.image.layers(1).images.rawimage;
    cells(k).image = double(ri.image);
    srim = 0; nl = 0;
    for l = 1:numel(c.image.layers)
        if ~isempty(c.image.layers(l).images)
            srim = c.image.layers(l).images.rawimage.image + srim; nl = nl + 1;
        end
    end
    cells(k).image_sum = double(srim);
    cells(k).nlayers = nl;
    cells(k).rangex = double(c.image.rangex) * 1000;
    cells(k).rangey = double(c.image.rangey) * 1000;
    cells(k).pixrec = double(c.image.parameters.sr_pixrec);
    try cells(k).numberOfLocs = double(ri.numberOfLocs); catch; cells(k).numberOfLocs = NaN; end
end
try
    lp = se.cells(1).image.parameters.layerparameters;
    if iscell(lp); lp = lp{1}; end
    rp = simplify(lp);
    fprintf('---- layer 1 render parameters used for cells\n'); disp(rp);
    fid = fopen(fullfile(cfg.output_dir, 'matlab_reference', 'S4_render_parameters.json'), 'w');
    fprintf(fid, '%s', jsonencode(rp)); fclose(fid);
catch err
    fprintf('note: could not export render parameters: %s\n', err.message);
end
sn = getplug(g, 'Segment', 'segmentNPC');
sn.setGuiParameters(struct('cutoff', cfg.segment.cutoff, 'diameterNPC', cfg.segment.diameter, 'rim', cfg.segment.rim));
sn.setGuiParameters(struct('saveon', true));
sn.setGuiParameters(struct('getmask', false));
psn = sn.getAllParameters;
fprintf('segmentNPC: cutoff %g, diameterNPC %g, rim %g, saveon %d, getmask %d\n', psn.cutoff, ...
    psn.diameterNPC, psn.rim, psn.saveon, psn.getmask);
if ~psn.saveon || psn.getmask
    error('segmentNPC: saveon must be on and getmask off for a scripted run');
end
sn.run(psn);
sites = export_sites(se.sites);
end

function m = getplug(g, group, name)
m = [];
try
    m = g.children.guiSites.children.(group).children.(name);
catch
end
if isempty(m) || ~isvalid(m)
    m = openplug(g, {'ROIManager', group, name});
end
end

function sites = export_sites(sl)
sites = struct('ID', {}, 'pos', {}, 'cell', {}, 'filenumber', {});
for k = 1:numel(sl)
    sites(k).ID = double(sl(k).ID);
    sites(k).pos = double(sl(k).pos(:)');
    sites(k).cell = double(sl(k).info.cell);
    sites(k).filenumber = double(sl(k).info.filenumber);
end
end

function [cleanup, ev] = run_cleanup(g, cfg)
se = g.locData.SE;
fh = figure('Visible', 'off');
ev = plugin('ROIManager', 'Evaluate', 'NPCsegmentCleanup', fh, g.P);
ev.attachLocData(g.locData);
ev.makeGui;
ev.name = 'NPCsegmentCleanup';
ev.display = false;
c = cfg.cleanup;
ev.setGuiParameters(struct('R', c.R, 'dR', c.dR, 'center', logical(c.center), 'inside', c.inside, ...
    'outside', c.outside, 'radiusrange', c.radiusrange(:)', 'minsize', c.minsize, ...
    'minlocs', c.minlocs, 'maxPSF', c.maxPSF));
pc = ev.getAllParameters;
fprintf('NPCsegmentCleanup: R %g dR %g center %d inside %g outside %g radiusrange %s minsize %g minlocs %g maxPSF %g siteroi %g\n', ...
    pc.R, pc.dR, pc.center, pc.inside, pc.outside, mat2str(pc.radiusrange), pc.minsize, pc.minlocs, pc.maxPSF, pc.se_siteroi);
cleanup = struct('ID', {}, 'pos_before', {}, 'pos_after', {}, 'use', {}, 'R0', {}, 'inside', {}, ...
    'outside', {}, 'inring', {}, 'sizeav', {}, 'locs', {});
for k = 1:numel(se.sites)
    s = se.sites(k);
    p0 = double(s.pos(:)');
    ev.evaluate(s);
    r = s.evaluation.NPCsegmentCleanup;
    cleanup(k).ID = double(s.ID);
    cleanup(k).pos_before = p0;
    cleanup(k).pos_after = double(s.pos(:)');
    cleanup(k).use = double(s.annotation.use);
    cleanup(k).R0 = getnum(r, 'R0');
    cleanup(k).inside = getnum(r, 'inside');
    cleanup(k).outside = getnum(r, 'outside');
    cleanup(k).inring = getnum(r, 'inring');
    cleanup(k).sizeav = getnum(r, 'sizeav');
    cleanup(k).locs = getnum(r, 'locs');
end
end

function v = getnum(s, f)
if isstruct(s) && isfield(s, f) && ~isempty(s.(f)); v = double(s.(f)); else; v = NaN; end
end

function res = run_locmofit(g, cfg, lc, ev)
se = g.locData.SE;
if isempty(ev)
    fh = figure('Visible', 'off');
    ev = plugin('ROIManager', 'Evaluate', 'NPCsegmentCleanup', fh, g.P);
    ev.attachLocData(g.locData);
    ev.makeGui;
end
fields = {'locprecnm', 'PSFxnm', 'xnm', 'ynm', 'frame', 'xnmrot', 'ynmrot'};
res = struct('ID', {}, 'pos', {}, 'xnm', {}, 'ynm', {}, 'locprecnm', {}, 'parId', {}, ...
    'value', {}, 'fix', {}, 'LLfit', {}, 'AIC', {}, 'AICc', {}, 'BGDensity', {}, 'nlocs', {}, ...
    'runtime', {}, 'error', {});
n = 0;
for k = 1:numel(se.sites)
    s = se.sites(k);
    if strcmp(cfg.locmofit_sites, 'used') && ~(isfield(s.annotation, 'use') && s.annotation.use)
        continue
    end
    n = n + 1;
    if ~isempty(cfg.locmofit_max_sites) && n > cfg.locmofit_max_sites
        break
    end
    ev.site = s;
    locs = ev.getLocs(fields, 'layer', 1, 'size', cfg.se.site_roi / 2);
    fl.xnm = double(locs.xnmrot(:));
    fl.ynm = double(locs.ynmrot(:));
    fl.locprecnm = double(locs.locprecnm(:));
    fl.PSFxnm = double(locs.PSFxnm(:));
    fl.frame = double(locs.frame(:));
    fl.layer = ones(size(fl.xnm));
    res(n).ID = double(s.ID);
    res(n).pos = double(s.pos(:)');
    res(n).xnm = fl.xnm;
    res(n).ynm = fl.ynm;
    res(n).locprecnm = fl.locprecnm;
    res(n).nlocs = numel(fl.xnm);
    res(n).error = '';
    try
        t = tic;
        fitter = build_fitter(lc, cfg.se.site_roi);
        fitter.fit(fl);
        res(n).runtime = toc(t);
        a = fitter.allParsArg;
        pid = cell(numel(a.name), 1);
        for j = 1:numel(a.name)
            pid{j} = sprintf('m%d.%s.%s', a.model(j), a.type{j}, a.name{j});
        end
        res(n).parId = pid;
        res(n).value = double(a.value(:));
        res(n).fix = double(a.fix(:));
        fi = fitter.fitInfo;
        res(n).LLfit = double(fi.LLfit);
        res(n).AIC = double(fi.AIC);
        res(n).AICc = double(fi.AICc);
        res(n).BGDensity = double(fi.BGDensity);
    catch err
        res(n).error = err.message;
        fprintf('LocMoFit site %d: %s\n', s.ID, err.message);
    end
end
fprintf('S6 %s: %d sites fitted\n', lc.name, numel(res));
end

function fitter = build_fitter(lc, roisize)
fitter = LocMoFit('SolverName', 'fminsearchbnd');
fitter.dataDim = 2;
fitter.roiSize = roisize;
models = lc.models;
if isstruct(models); models = num2cell(models); end
for k = 1:numel(models)
    geo = feval(models{k}.class);
    mod = functionModel(geo);
    mod.layer = models{k}.layer;
    fitter.addModel(mod);
end
pars = lc.pars;
if isstruct(pars); pars = num2cell(pars); end
for k = 1:numel(pars)
    p = pars{k};
    args = {};
    flds = {'value', 'fix', 'lb', 'ub', 'min', 'max'};
    for j = 1:numel(flds)
        if isfield(p, flds{j}) && ~isempty(p.(flds{j}))
            v = p.(flds{j});
            if isnumeric(v); v(v > 1e299) = inf; v(v < -1e299) = -inf; end
            if strcmp(flds{j}, 'fix'); v = logical(v); end
            args = [args, {flds{j}, v}]; %#ok<AGROW>
        end
    end
    fitter.setParArg(p.id, args{:});
end
end

function s = stage_ok(t0)
s = struct('ok', true, 'seconds', toc(t0));
end

function s = stage_err(err)
try
    rep = getReport(err, 'extended', 'hyperlinks', 'off');
catch
    rep = err.message;
end
s = struct('ok', false, 'error', err.message, 'report', rep);
fprintf(2, '\n*** ERROR: %s\n', rep);
end

function write_status(outdir, status)
fid = fopen(fullfile(outdir, 'status.json'), 'w');
fprintf(fid, '%s', jsonencode(status));
fclose(fid);
end

