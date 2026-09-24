#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_vessel_features.py
==========================
Single-file consolidation of the vessel-feature pipeline used for the post-TIPS OHE model.
It replaces the former chain of scripts:

    nifiti_segment.py       -> split labels 1..5, keep largest 26-connected component   (step 1)
    compute_centerline.py   -> vessel mesh + vmtk centerlines                            (step 2)
    compute_params_0314.py  -> length / tortuosity / curvature / diameters / circularity (step 3)
    cal_csa.py              -> diameter -> area (pi*(d/2)^2) and pairwise ratios          (step 4)

Input : multi-label vessel mask (NIfTI, labels 1..5, e.g. <case>_vascular.nii.gz from nnUNet Task703),
        or a directory of such masks.
Output: per case, in <out>/<case>/ :
        <case>_<i>.nii.gz               binary mask of vessel i (largest connected component)
        <case>_<i>.vtk                  surface mesh
        <case>_<i>_centerlines.vtk      vmtk centerlines (with MaximumInscribedSphereRadius)
        <case>_vessel_features.csv/json all per-vessel parameters + derived features + the 5 model features
        and <out>/vessel_features_all.csv (one row per case).

The 5 features used by the 26-feature model:  MIA_12, maximum area_3, tortuosity_3,
curvature_2, curvature_3   (units: mm, mm^2).

Dependencies (the same environment that ran compute_centerline.py):
    numpy, scipy, scikit-image, nibabel, vtk, itk (ITK python), vmtk
`--selftest-core` needs only numpy / scipy / scikit-image.

Usage
-----
    python compute_vessel_features.py --mask C1001_vascular.nii.gz --out out
    python compute_vessel_features.py --mask masks_dir --out out --labels 1 2 3 --workers 2
    python compute_vessel_features.py --selftest-core      # pure-python part, no vtk/vmtk needed
    python compute_vessel_features.py --selftest --out st  # full pipeline on a synthetic phantom

Design note: the numerical steps (mesh generation, vmtkCenterlines, vmtkCenterlineGeometry,
plane sections + Delaunay2D areas, median aggregation) are kept identical to the original
scripts so that features stay comparable with the training data. The few deliberate
deviations are listed in the DEVIATIONS constant right below.
"""
import argparse
import csv
import json
import logging
import math
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np

LOG = logging.getLogger("vessel")

DEVIATIONS = """
1. nib.load().get_data() (removed in nibabel>=5) -> np.asanyarray(img.dataobj); skeletonize_3d falls back to skeletonize.
2. Endpoint counting uses a zero-padded 3x3x3 convolution instead of Python slicing (identical for interior voxels,
   removes the negative-index wrap-around at the volume border). Point order is unchanged (C order).
3. Isolated-in-z voxel patching: slice start clamped at 0 (original wrapped around at z=0).
4. Dilation uses scipy.ndimage.binary_dilation(ball(2)) (== grayscale dilation of a binary mask), on a cropped box for speed.
5. The dead 'portal_vein' inlet branch is removed; if no endpoint/vertex is found a clear error is raised instead of NameError.
6. Cross-section stations whose cut contour is empty are skipped (original raised IndexError and lost the whole vessel).
7. Excel (xlwt/xlrd) output replaced by CSV/JSON. Per-vessel volume is computed from the mask array (same value).
8. Optional --fix-inlet-target removes the inlet from the target list (OFF by default = original behaviour).
"""

# order of the 10 per-vessel parameters in the original Excel / training table
PARAM_COLUMNS = [
    ("vol", "vol_{i}"),
    ("total_length", "Total length_{i}"),
    ("tortuosity", "tortuosity_{i}"),
    ("curvature", "curvature_{i}"),
    ("max_diameter", "maximum diameter_{i}"),
    ("eq_diameter", "equivalent diameter_{i}"),
    ("min_diameter", "minimum diameter_{i}"),
    ("circularity", "circularity_{i}"),
    ("n_end_nodes", "Number of end nodes_{i}"),
    ("n_branches", "Number of branches_{i}"),
]
MODEL_FEATURES = ["MIA_12", "maximum area_3", "tortuosity_3", "curvature_2", "curvature_3"]
NAN = float("nan")


# --------------------------------------------------------------------------------------
# Step 1: label splitting (pure numpy / scipy)
# --------------------------------------------------------------------------------------
def largest_component(mask):
    """Largest 26-connected component of a boolean mask (== SimpleITK FullyConnected)."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask > 0, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return (lab == int(counts.argmax())).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Step 2a: skeleton end points (pure numpy / scipy / skimage)
# --------------------------------------------------------------------------------------
def _ball(r):
    from skimage import morphology
    return morphology.ball(r)


def _skeletonize(vol):
    from skimage import morphology
    if hasattr(morphology, "skeletonize_3d"):
        try:
            return morphology.skeletonize_3d(vol)
        except Exception:  # removed / deprecated in newer scikit-image
            pass
    return morphology.skeletonize(vol)


def skeleton_endpoints(mask, dilate_radius=2, margin=4):
    """Voxel indices (array order i,j,k) of skeleton end points of dilate(mask, ball(2)).
    An end point is a skeleton voxel with exactly one skeleton neighbour in its 3x3x3 neighbourhood
    (neighbourhood sum == 2 incl. itself), as in the original get_endpoints()."""
    from scipy import ndimage as ndi
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=int)
    pad = dilate_radius + margin
    lo = np.maximum(idx.min(axis=0) - pad, 0)
    hi = np.minimum(idx.max(axis=0) + pad + 1, mask.shape)
    crop = mask[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))] > 0
    dil = ndi.binary_dilation(crop, structure=_ball(dilate_radius))
    skel = np.asarray(_skeletonize(dil)) > 0
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3, 3), dtype=np.uint8), mode="constant", cval=0)
    ends = np.argwhere(skel & (nb == 2))
    return ends + lo


def voxels_to_world(voxels, spacing, affine):
    """Original conversion: (index * spacing, 1) -> 4x4 matrix (RAS->LPS adjusted qform/sform)."""
    voxels = np.asarray(voxels, dtype=float).reshape(-1, 3)
    if len(voxels) == 0:
        return []
    pts = np.c_[voxels * np.asarray(spacing, dtype=float), np.ones(len(voxels))]
    return (pts @ np.asarray(affine, dtype=float).T)[:, :3].tolist()


# --------------------------------------------------------------------------------------
# Step 2b: mask preparation for meshing (pure numpy / scipy)
# --------------------------------------------------------------------------------------
def prepare_mask_for_mesh(arr_zyx):
    """arr in (z, y, x) order (as returned by itk.GetArrayFromImage). Returns uint8 {0, 255}:
    threshold, patch voxels that are isolated along z (add one neighbour in z), dilate with ball(2)."""
    from scipy import ndimage as ndi
    out = np.where(np.asarray(arr_zyx) > 0, 255, 0).astype(np.uint8)
    nz = out.shape[0]
    zs, ys, xs = np.nonzero(out)
    for z, y, x in zip(zs, ys, xs):
        if np.count_nonzero(out[max(z - 1, 0):z + 2, y, x]) == 1:
            zz = z + 1 if z + 1 < nz else z - 1
            out[zz, y, x] = out[z, y, x]
    dil = ndi.binary_dilation(out > 0, structure=_ball(2))
    return dil.astype(np.uint8) * 255


# --------------------------------------------------------------------------------------
# Step 3 helpers (pure python / numpy)
# --------------------------------------------------------------------------------------
def perimeter_and_max_dist(points):
    """Chain traversal used for the section contour: start at points[0], repeatedly jump to the nearest
    remaining point. Returns (open-polyline perimeter, largest 'farthest-remaining-point' distance)."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    alive = np.ones(len(P), dtype=bool)
    tp0 = P[0]
    perimeter, maxd = 0.0, []
    while alive.any():
        idx = np.flatnonzero(alive)
        d2 = np.sum((P[idx] - tp0) ** 2, axis=1)
        k = int(np.argmin(d2))
        maxd.append(math.sqrt(float(d2.max())))
        perimeter += math.sqrt(float(d2[k]))
        tp0 = P[idx[k]]
        alive[idx[k]] = False
    return perimeter, max(maxd)


def topology_metrics(cells):
    """cells: {cell_id: [point tuples]} exactly as built by read_cells() (first point of every cell is
    dropped, consecutive duplicates removed). Returns total_length, tortuosity of the 'main' first segment
    of cell 0, number of end nodes, number of branches, number of branch nodes."""
    label_points = Counter(p for pts in cells.values() for p in pts)
    cell_length, euclid_length, cell_label = {}, {}, {}
    for key, pts in cells.items():
        label_cell = [label_points[p] for p in pts]
        cell_dist, euclid_dist, labels = [], [], []
        start_id, _id, dist = 0, 0, 0.0
        for ind, val in enumerate(label_cell):
            if val == label_cell[_id]:
                dist += float(np.linalg.norm(np.array(pts[_id]) - np.array(pts[ind])))
                _id = ind
                if ind == len(label_cell) - 1:
                    labels.append(label_cell[start_id])
                    cell_dist.append(dist)
                    euclid_dist.append(float(np.linalg.norm(np.array(pts[start_id]) - np.array(pts[_id]))))
                    dist, start_id, _id = 0.0, ind, ind
            else:
                labels.append(label_cell[start_id])
                cell_dist.append(dist)
                euclid_dist.append(float(np.linalg.norm(np.array(pts[start_id]) - np.array(pts[_id]))))
                dist, start_id, _id = 0.0, ind, ind
        cell_length[key], euclid_length[key], cell_label[key] = cell_dist, euclid_dist, labels

    total_length = 0.0
    for key, vals in cell_length.items():
        for i, val in enumerate(vals):
            total_length += val / cell_label[key][i]

    first_len = cell_length.get(0) or [0.0]
    first_euc = euclid_length.get(0) or [0.0]
    terminal_nodes = len(cells) + 1
    branches = 0.0
    for key, vals in cell_label.items():
        for val in vals:
            branches += 1.0 / val if val > 1 else 1.0
    branch_nodes = int(branches) + 1 - terminal_nodes
    tortuosity = (first_len[0] / first_euc[0]) - 1.0 if first_euc[0] != 0 else 0.0
    return dict(total_length=total_length, tortuosity=tortuosity, n_end_nodes=terminal_nodes,
                n_branches=branches, n_branch_nodes=branch_nodes)


def _safe_div(a, b):
    try:
        if b is None or a is None or not np.isfinite(a) or not np.isfinite(b) or b == 0:
            return NAN
        return float(a) / float(b)
    except Exception:
        return NAN


def derive_features(per_vessel):
    """per_vessel: {label: params dict (keys of PARAM_COLUMNS) or None}. Reproduces cal_csa.py:
    diameters -> areas (pi*(d/2)^2), pairwise ratios MIA/MAA/EA over all computed vessel pairs."""
    row = {}
    areas = {}
    for i, p in per_vessel.items():
        p = p or {}
        for key, colfmt in PARAM_COLUMNS:
            row[colfmt.format(i=i)] = float(p.get(key, NAN))
        for dkey, akey in (("max_diameter", "maximum area"), ("eq_diameter", "equivalent area"),
                           ("min_diameter", "minimum area")):
            d = p.get(dkey, NAN)
            a = math.pi * (d / 2.0) ** 2 if d is not None and np.isfinite(d) else NAN
            row[f"{akey}_{i}"] = a
            areas[(akey, i)] = a
    labs = sorted(per_vessel)
    for a, b in combinations(labs, 2):
        row[f"MAA_{a}{b}"] = _safe_div(areas[("maximum area", a)], areas[("maximum area", b)])
        row[f"MIA_{a}{b}"] = _safe_div(areas[("minimum area", a)], areas[("minimum area", b)])
        row[f"EA_{a}{b}"] = _safe_div(areas[("equivalent area", a)], areas[("equivalent area", b)])
    return row


def gui_fields(row):
    """Names used by app.py::cal_vascular (area_pv = minimum area_1, area_sv = minimum area_2,
    area_lpv = maximum area_3, ...), so the GUI can be fed directly."""
    g = lambda k: row.get(k, NAN)
    return {
        "area_pv": g("minimum area_1"), "area_sv": g("minimum area_2"), "area_lpv": g("maximum area_3"),
        "tortuosity_lpv": g("tortuosity_3"), "curvature_sv": g("curvature_2"), "curvature_lpv": g("curvature_3"),
    }


# --------------------------------------------------------------------------------------
# VTK / ITK / vmtk dependent part (calls kept identical to the original scripts)
# --------------------------------------------------------------------------------------
def _need(name):
    try:
        return __import__(name)
    except ImportError as e:
        raise ImportError("missing dependency '%s' (%s). See the module docstring." % (name, e))


def get_affine_and_spacing(nifti_path):
    """vtkNIFTIImageReader qform/sform, RAS->LPS sign flip of the first two rows (as in the original)."""
    vtk = _need("vtk")
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti_path)
    reader.Update()
    transform = vtk.vtkTransform()
    if reader.GetQFormMatrix():
        transform.SetMatrix(reader.GetQFormMatrix())
    elif reader.GetSFormMatrix():
        transform.SetMatrix(reader.GetSFormMatrix())
    flat = [0] * 16
    m = transform.GetMatrix()
    m.DeepCopy(flat, m)
    affine = np.array(flat).reshape(4, 4) * np.array(
        [[-1, -1, -1, -1], [-1, -1, -1, -1], [1, 1, 1, 1], [0, 0, 0, 1]])
    return affine, reader.GetOutput().GetSpacing()


def nifti_to_mesh(nifti_path, mesh_path):
    """== original nifti2stl(): itk read -> prepare mask -> BinaryMask3DMeshSource -> vtk clean/smooth/normals."""
    vtk = _need("vtk")
    itk = _need("itk")
    t0 = time.perf_counter()
    image_type = itk.Image[itk.UC, 3]
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(nifti_path)
    try:
        reader.Update()
    except Exception:
        LOG.warning("itk could not read %s; resetting qform from sform and retrying", nifti_path)
        nib = _need("nibabel")
        info = nib.load(nifti_path)
        info.set_qform(info.get_qform())
        info.set_sform(info.get_sform())
        nib.save(info, nifti_path)
        reader.Update()

    npy = itk.GetArrayFromImage(reader.GetOutput())
    npy = prepare_mask_for_mesh(npy)
    image = itk.GetImageFromArray(npy)
    image.SetSpacing(reader.GetOutput().GetSpacing())
    image.SetOrigin(reader.GetOutput().GetOrigin())
    image.SetDirection(reader.GetOutput().GetDirection())
    image.CopyInformation(reader.GetOutput())

    mesh_type = itk.Mesh[itk.D, 3]
    mesh_filter = itk.BinaryMask3DMeshSource[image_type, mesh_type].New()
    mesh_filter.SetInput(image)
    mesh_filter.SetObjectValue(255)
    writer = itk.MeshFileWriter[mesh_type].New()
    writer.SetFileName(mesh_path)
    writer.SetInput(mesh_filter.GetOutput())
    writer.SetFileTypeAsBINARY()
    writer.Update()

    r = vtk.vtkPolyDataReader()
    r.SetFileName(mesh_path)
    r.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(r.GetOutput())
    clean.Update()
    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputData(clean.GetOutput())
    smooth.SetNumberOfIterations(20)
    smooth.SetRelaxationFactor(0.1)
    smooth.SetFeatureAngle(175)
    smooth.SetFeatureEdgeSmoothing(1)
    smooth.SetBoundarySmoothing(1)
    smooth.Update()
    normal = vtk.vtkPolyDataNormals()
    normal.SetInputData(smooth.GetOutput())
    normal.SetAutoOrientNormals(1)
    normal.SplittingOff()
    normal.ConsistencyOn()
    normal.ComputePointNormalsOn()
    normal.ComputeCellNormalsOn()
    normal.Update()
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(mesh_path)
    w.SetInputData(normal.GetOutput())
    w.Write()
    LOG.info("mesh done in %.1fs -> %s", time.perf_counter() - t0, mesh_path)
    return normal.GetOutput()


def compute_centerlines(nifti_path, out_dir, prefix, fix_inlet_target=False):
    """== original compute_centerlines(). Returns (mesh_path, centerlines_path)."""
    vtk = _need("vtk")
    nib = _need("nibabel")
    vtk.vtkObject.GlobalWarningDisplayOff()
    mesh_path = os.path.join(out_dir, prefix + ".vtk")
    cl_path = os.path.join(out_dir, prefix + "_centerlines.vtk")

    affine, spacing = get_affine_and_spacing(nifti_path)
    img = nib.load(nifti_path)
    mask = np.asanyarray(img.dataobj)
    vox = skeleton_endpoints(np.squeeze(mask))
    endpoints = voxels_to_world(vox, spacing, affine)
    LOG.info("%s: %d skeleton end points", prefix, len(endpoints))
    if not endpoints:
        raise RuntimeError("no skeleton end points found (mask empty or too small)")

    surface = nifti_to_mesh(nifti_path, mesh_path)
    locator = vtk.vtkKdTree()
    locator.BuildLocatorFromPoints(surface.GetPoints())
    distances = []
    for p in endpoints:
        res = vtk.vtkIdList()
        locator.FindClosestNPoints(1, p, res)
        if res.GetNumberOfIds() < 1:
            continue
        p1 = surface.GetPoint(res.GetId(0))
        distances.append(float(np.linalg.norm(np.array(p) - np.array(p1))))
    if not distances:
        raise RuntimeError("could not relate skeleton end points to the mesh surface")
    if min(distances) > 8.0:
        LOG.warning("%s: nearest end point is %.1f mm from the surface -> voxel/world coordinate "
                    "mismatch? (mesh and end points should coincide)", prefix, min(distances))
    # original rule: inlet = end point farthest from the surface
    inlet = endpoints[distances.index(max(distances))]
    targets = [e for e in endpoints if not (fix_inlet_target and e == inlet)]
    flat_targets = []
    for e in targets:
        flat_targets += list(e)

    from vmtk import vmtkscripts
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface
    cl.SeedSelectorName = "pointlist"
    cl.SourcePoints = inlet
    cl.TargetPoints = flat_targets
    cl.RadiusArrayName = "MaximumInscribedSphereRadius"
    cl.Execute()
    lines = cl.Centerlines
    lines.BuildCells()
    lines.RemoveDeletedCells()
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(cl_path)
    w.SetInputData(lines)
    w.Write()
    return mesh_path, cl_path


def read_cells(centerlines):
    """Cell -> list of points; first point of each cell dropped and consecutive duplicates removed (original)."""
    vtk = _need("vtk")
    cells = {}
    for cell_id in range(centerlines.GetNumberOfCells()):
        pts = []
        ids = vtk.vtkIdList()
        centerlines.GetCellPoints(cell_id, ids)
        for p_id in range(ids.GetNumberOfIds()):
            pt = centerlines.GetPoint(ids.GetId(p_id))
            if p_id > 0 and centerlines.GetPoint(ids.GetId(p_id - 1)) != pt:
                pts.append(pt)
        cells[cell_id] = pts
    return cells


def section_stats(mesh_surface, centerlines, cells):
    """== original compute_Radius2(): a plane cut every 10 centerline points. Returns {point: [max diameter,
    equivalent diameter, 2*MIS radius, circularity]}."""
    vtk = _need("vtk")
    radius_array = centerlines.GetPointData().GetArray("MaximumInscribedSphereRadius")
    plane, cutter = vtk.vtkPlane(), vtk.vtkCutter()
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    out = {}
    for c_id, c_points in cells.items():
        if len(c_points) < 10:
            continue
        for ind in range(0, len(c_points), 10):
            if ind + 5 >= len(c_points):
                continue
            p0, p1 = c_points[ind], c_points[ind + 5]
            vec = np.array(p0) - np.array(p1)
            nrm = np.linalg.norm(vec)
            if nrm == 0:
                continue
            plane.SetOrigin(p0)
            plane.SetNormal(vec / nrm)
            cutter.SetCutFunction(plane)
            cutter.SetInputData(mesh_surface)
            cutter.GenerateTrianglesOn()
            cutter.Update()
            connectivity.SetInputData(cutter.GetOutput())
            connectivity.SetClosestPoint(p0)
            connectivity.SetExtractionModeToClosestPointRegion()
            connectivity.Update()
            delaunay = vtk.vtkDelaunay2D()
            delaunay.SetInputData(connectivity.GetOutput())
            delaunay.SetTolerance(0.00001)
            delaunay.Update()
            massprop = vtk.vtkMassProperties()
            massprop.SetInputData(delaunay.GetOutput())
            area = massprop.GetSurfaceArea()
            r = radius_array.GetTuple1(centerlines.FindPoint(p0))
            if area >= 5 * math.pi * r * r:
                continue
            npts = connectivity.GetOutput().GetNumberOfPoints()
            if npts < 3:
                continue
            contour = [connectivity.GetOutput().GetPoint(k) for k in range(npts)]
            perimeter, max_d = perimeter_and_max_dist(contour)
            if perimeter <= 0:
                continue
            rd = 4 * math.pi * area / (perimeter * perimeter)
            if rd > 1:
                continue
            out[p0] = [max_d, 2 * math.sqrt(area / math.pi), 2 * r, rd]
    return out


def compute_vessel_params(nifti_path, mesh_path, centerlines_path, vol_ml):
    """== original compute_params(), returning a dict instead of writing Excel."""
    vtk = _need("vtk")
    vmtkscripts = __import__("vmtk", fromlist=["vmtkscripts"]).vmtkscripts
    vtk.vtkObject.GlobalWarningDisplayOff()
    rd = vtk.vtkPolyDataReader()
    rd.SetFileName(centerlines_path)
    rd.Update()
    centerlines = rd.GetOutput()

    cells = read_cells(centerlines)
    topo = topology_metrics(cells)

    geo = vmtkscripts.vmtkCenterlineGeometry()
    geo.Centerlines = centerlines
    geo.CurvatureArrayName = "Curvature"
    geo.Execute()
    curv_arr = geo.Centerlines.GetPointData().GetScalars("Curvature")
    curv = []
    for cell_id in range(geo.Centerlines.GetNumberOfCells()):
        ids = vtk.vtkIdList()
        geo.Centerlines.GetCellPoints(cell_id, ids)
        for p_id in range(ids.GetNumberOfIds()):
            curv.append(curv_arr.GetTuple1(ids.GetId(p_id)))
    curvature = float(np.nanmedian(curv)) if curv else NAN

    mr = vtk.vtkPolyDataReader()
    mr.SetFileName(mesh_path)
    mr.Update()
    sec = section_stats(mr.GetOutput(), centerlines, cells)
    if sec:
        med = np.median(np.array(list(sec.values())), axis=0)
    else:
        LOG.warning("%s: no valid cross-sections -> diameters/circularity = NaN", os.path.basename(mesh_path))
        med = [NAN] * 4
    return dict(vol=vol_ml, total_length=topo["total_length"], tortuosity=topo["tortuosity"],
                curvature=curvature, max_diameter=float(med[0]), eq_diameter=float(med[1]),
                min_diameter=float(med[2]), circularity=float(med[3]),
                n_end_nodes=topo["n_end_nodes"], n_branches=topo["n_branches"],
                n_branch_nodes=topo["n_branch_nodes"], n_sections=len(sec))


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def run_vessel_job(job):
    """Top-level (picklable) worker: one vessel -> centerlines + parameters."""
    label = job["label"]
    prefix = job["prefix"]
    t0 = time.perf_counter()
    try:
        mesh_path = os.path.join(job["dir"], prefix + ".vtk")
        cl_path = os.path.join(job["dir"], prefix + "_centerlines.vtk")
        if not (job["reuse"] and os.path.exists(mesh_path) and os.path.exists(cl_path)):
            mesh_path, cl_path = compute_centerlines(job["nifti"], job["dir"], prefix, job["fix_inlet_target"])
        params = compute_vessel_params(job["nifti"], mesh_path, cl_path, job["vol_ml"])
        return dict(label=label, ok=True, params=params, error=None, seconds=time.perf_counter() - t0)
    except Exception as e:  # keep going with the other vessels
        LOG.error("%s failed: %s", prefix, e)
        LOG.debug(traceback.format_exc())
        return dict(label=label, ok=False, params=None, error=str(e), seconds=time.perf_counter() - t0)


def case_id_of(path):
    name = os.path.basename(path)
    for suf in (".nii.gz", ".nii"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return os.path.splitext(name)[0]


def process_case(mask_path, out_root, labels=(1, 2, 3), workers=1, reuse=False, fix_inlet_target=False):
    nib = _need("nibabel")
    case = case_id_of(mask_path)
    if case.endswith("_vascular"):
        case = case[: -len("_vascular")]
    case_dir = os.path.join(out_root, case)
    os.makedirs(case_dir, exist_ok=True)

    img = nib.load(mask_path)
    arr = np.squeeze(np.rint(np.asanyarray(img.dataobj)).astype(np.int32))
    if arr.ndim != 3:
        raise ValueError("mask must be 3D, got shape %s" % (arr.shape,))
    zooms = img.header.get_zooms()[:3]
    unit_ml = float(np.prod(zooms)) / 1000.0
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)

    jobs, results, missing = [], {}, {}
    for lab in labels:
        comp = largest_component(arr == lab)
        n = int(comp.sum())
        if n == 0:
            missing[lab] = "label %d not present in mask" % lab
            continue
        vessel_path = os.path.join(case_dir, "%s_%d.nii.gz" % (case, lab))
        nib.save(nib.Nifti1Image(comp, img.affine, hdr), vessel_path)
        jobs.append(dict(label=lab, prefix="%s_%d" % (case, lab), dir=case_dir, nifti=vessel_path,
                         vol_ml=n * unit_ml, reuse=reuse, fix_inlet_target=fix_inlet_target))
    LOG.info("%s: %d vessel job(s), labels %s", case, len(jobs), [j["label"] for j in jobs])

    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(run_vessel_job, jobs))
    else:
        outs = [run_vessel_job(j) for j in jobs]

    per_vessel, errors = {}, dict(missing)
    for o in outs:
        if o["ok"]:
            per_vessel[o["label"]] = o["params"]
            LOG.info("%s_%d done in %.0fs", case, o["label"], o["seconds"])
        else:
            errors[o["label"]] = o["error"]
    for lab in labels:
        per_vessel.setdefault(lab, None)

    row = derive_features(per_vessel)
    model = {k: row.get(k, NAN) for k in MODEL_FEATURES}
    write_case_outputs(case_dir, case, row, model, gui_fields(row), errors)
    return case, row, model, errors


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def write_case_outputs(case_dir, case, row, model, gui, errors):
    rec = {"ID": case}
    rec.update(row)
    with open(os.path.join(case_dir, case + "_vessel_features.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(rec.keys()))
        w.writerow(["" if (isinstance(v, float) and not np.isfinite(v)) else v for v in rec.values()])
    with open(os.path.join(case_dir, case + "_vessel_features.json"), "w", encoding="utf-8") as f:
        json.dump({"ID": case, "model_features": {k: _clean(v) for k, v in model.items()},
                   "gui_fields": {k: _clean(v) for k, v in gui.items()},
                   "all_features": {k: _clean(v) for k, v in row.items()},
                   "errors": {str(k): v for k, v in errors.items()}}, f, ensure_ascii=False, indent=2)


def append_summary(path, case, row):
    rec = {"ID": case}
    rec.update(row)
    exists = os.path.exists(path)
    old_rows = []
    if exists:
        with open(path, newline="", encoding="utf-8") as f:
            old_rows = [r for r in csv.DictReader(f) if r.get("ID") != case]
    cols = list(dict.fromkeys(list(old_rows[0].keys()) + list(rec.keys()))) if old_rows else list(rec.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in old_rows:
            w.writerow(r)
        w.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v) else v) for k, v in rec.items()})


# --------------------------------------------------------------------------------------
# Self tests
# --------------------------------------------------------------------------------------
def make_phantom(shape=(100, 100, 130)):
    """Synthetic multi-label vessel mask (isotropic 1 mm): label 1 straight tube (r=6), label 2 Y-shaped tree
    (trunk r=5, branches r=3.5), label 3 circular arc (R=80 mm, r=5)."""
    from scipy import ndimage as ndi
    arr = np.zeros(shape, dtype=np.uint8)

    def tube(points, radius):
        cl = np.zeros(shape, dtype=bool)
        pts = np.asarray(points, dtype=float)
        for a, b in zip(pts[:-1], pts[1:]):
            n = int(np.ceil(np.linalg.norm(b - a) * 2)) + 1
            for t in np.linspace(0, 1, n):
                x, y, z = np.rint(a + (b - a) * t).astype(int)
                cl[x, y, z] = True
        return ndi.distance_transform_edt(~cl) <= radius

    arr[tube([(20, 50, 10), (20, 50, 110)], 6)] = 1
    y_mask = (tube([(50, 50, 10), (50, 50, 60)], 5) | tube([(50, 50, 60), (35, 50, 110)], 3.5)
              | tube([(50, 50, 60), (65, 50, 110)], 3.5))
    arr[y_mask & (arr == 0)] = 2
    t = np.linspace(0, 1.2, 80)
    arc = np.c_[80 * (1 - np.cos(t)) + 10, np.full_like(t, 80), 80 * np.sin(t) + 10]
    arr[tube(arc, 5) & (arr == 0)] = 3
    return arr


PHANTOM_EXPECT = {  # label -> (diameter mm, curvature upper/lower bound hint)
    1: dict(diam=12.0, curv=(0.0, 0.03)),
    2: dict(diam=10.0, curv=(0.0, 0.05)),
    3: dict(diam=10.0, curv=(0.004, 0.03)),
}


def selftest_core():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + (("  " + extra) if extra else ""))
        ok &= bool(cond)

    ph = make_phantom()
    # largest component: add a tiny separate blob to label 1 -> must be removed
    noisy = ph.copy()
    noisy[90, 5, 5] = 1
    comp = largest_component(noisy == 1)
    check("largest_component drops stray voxel", comp[90, 5, 5] == 0 and comp.sum() == (ph == 1).sum())
    # endpoints
    e1 = skeleton_endpoints(largest_component(ph == 1))
    e2 = skeleton_endpoints(largest_component(ph == 2))
    check("straight tube has 2 skeleton end points", len(e1) == 2, "got %d" % len(e1))
    check("Y tree has 3 skeleton end points", len(e2) == 3, "got %d" % len(e2))
    z_ends = sorted(int(v) for v in e1[:, 2])
    check("straight tube end points near z=10 and z=110 (after dilation ~ +/-2)",
          abs(z_ends[0] - 10) <= 5 and abs(z_ends[1] - 110) <= 5, str(z_ends))
    # world coordinates: identity-ish affine with 2 mm spacing
    aff = np.eye(4)
    w = voxels_to_world([[1, 2, 3]], (2.0, 2.0, 3.0), aff)
    check("voxels_to_world scales by spacing", np.allclose(w[0], [2, 4, 9]))
    # mesh preparation
    zyx = np.transpose(ph == 1, (2, 1, 0))
    prep = prepare_mask_for_mesh(zyx)
    check("prepare_mask_for_mesh -> uint8 {0,255}, dilated", set(np.unique(prep)) <= {0, 255} and (prep > 0).sum() > zyx.sum())
    iso = np.zeros((5, 4, 4), dtype=np.uint8)
    iso[2, 1, 1] = 1
    iso_prep = prepare_mask_for_mesh(iso)
    check("prepare_mask_for_mesh handles isolated voxel", iso_prep.max() == 255)
    # perimeter / max distance on a circle
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    circ = np.c_[5 * np.cos(th), 5 * np.sin(th), np.zeros_like(th)]
    per, mx = perimeter_and_max_dist(circ)
    check("perimeter_and_max_dist circle r=5", abs(per - 2 * np.pi * 5) / (2 * np.pi * 5) < 0.05 and abs(mx - 10) < 0.05,
          "perimeter=%.2f max=%.2f" % (per, mx))
    # topology on a Y: trunk of 10 pts then two branches of 10 pts each (branch points shared)
    trunk = [(0.0, 0.0, float(z)) for z in range(1, 11)]
    b1 = trunk + [(float(k), 0.0, 10.0 + k) for k in range(1, 11)]
    b2 = trunk + [(-float(k), 0.0, 10.0 + k) for k in range(1, 11)]
    topo = topology_metrics({0: b1, 1: b2})
    # original semantics: a segment sums the steps *inside* it (10 pts -> 9 steps); the connecting step between two
    # segments is not counted. Trunk (9 steps, shared by 2 cells -> weight 1/2 each) + two branches of 9 diagonal steps.
    exp_total = 9 + 2 * 9 * math.sqrt(2)
    check("topology_metrics total length of Y", abs(topo["total_length"] - exp_total) < 1e-6,
          "got %.3f expected %.3f" % (topo["total_length"], exp_total))
    check("topology_metrics end nodes / branches", topo["n_end_nodes"] == 3 and abs(topo["n_branches"] - 3.0) < 1e-9, str(topo))
    check("topology_metrics tortuosity of straight trunk ~ 0", abs(topo["tortuosity"]) < 1e-9, str(topo["tortuosity"]))
    # curved single line: tortuosity = L/d - 1 > 0
    arc = [(10 * math.sin(t), 0.0, 10 * (1 - math.cos(t))) for t in np.linspace(0.05, 1.5, 30)]
    tt = topology_metrics({0: arc})
    L = sum(np.linalg.norm(np.array(a) - np.array(b)) for a, b in zip(arc[:-1], arc[1:]))
    d = np.linalg.norm(np.array(arc[0]) - np.array(arc[-1]))
    check("topology_metrics tortuosity == L/d-1", abs(tt["tortuosity"] - (L / d - 1)) < 1e-9)
    # degenerate cell 0 -> tortuosity 0 (original fallback)
    tz = topology_metrics({0: [], 1: b1})
    check("topology_metrics degenerate cell 0 -> tortuosity 0", tz["tortuosity"] == 0.0)
    # derived features / ratios / gui names
    pv = {i: dict(vol=1.0, total_length=100.0, tortuosity=0.1 * i, curvature=0.01 * i, max_diameter=10.0 + i,
                  eq_diameter=9.0 + i, min_diameter=8.0 + i, circularity=0.9, n_end_nodes=3, n_branches=3)
          for i in (1, 2, 3)}
    row = derive_features(pv)
    check("area from diameter", abs(row["maximum area_3"] - math.pi * (13.0 / 2) ** 2) < 1e-9)
    check("MIA_12 == min area_1 / min area_2", abs(row["MIA_12"] - (9.0 / 10.0) ** 2) < 1e-9, "%.4f" % row["MIA_12"])
    check("all 5 model features present", all(k in row for k in MODEL_FEATURES))
    g = gui_fields(row)
    check("gui_fields mapping", abs(g["area_lpv"] - row["maximum area_3"]) < 1e-12 and g["curvature_sv"] == row["curvature_2"])
    row_nan = derive_features({1: pv[1], 2: None})
    check("failed vessel -> NaN ratios (no exception)", np.isnan(row_nan["MIA_12"]))
    # writers
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        write_case_outputs(td, "T1", row, {k: row[k] for k in MODEL_FEATURES}, g, {})
        j = json.load(open(os.path.join(td, "T1_vessel_features.json"), encoding="utf-8"))
        check("json written with model features", set(j["model_features"]) == set(MODEL_FEATURES))
        append_summary(os.path.join(td, "all.csv"), "T1", row)
        append_summary(os.path.join(td, "all.csv"), "T2", row_nan)
        rows = list(csv.DictReader(open(os.path.join(td, "all.csv"), encoding="utf-8")))
        check("summary csv has 2 rows", len(rows) == 2 and rows[0]["ID"] == "T1")
    print("\nCORE SELFTEST:", "ALL PASSED" if ok else "FAILED")
    return ok


def selftest_full(out_dir):
    nib = _need("nibabel")
    os.makedirs(out_dir, exist_ok=True)
    arr = make_phantom()
    path = os.path.join(out_dir, "phantom_vascular.nii.gz")
    nib.save(nib.Nifti1Image(arr, np.eye(4)), path)
    case, row, model, errors = process_case(path, out_dir, labels=(1, 2, 3), workers=1)
    print("\n=== phantom results (isotropic 1 mm; mm / mm^2) ===")
    for i in (1, 2, 3):
        print("vessel %d: eq diam %.2f, min diam %.2f, max diam %.2f, circ %.2f, curvature %.4f, tortuosity %.3f, len %.1f, vol %.2f ml"
              % (i, row[f"equivalent diameter_{i}"], row[f"minimum diameter_{i}"], row[f"maximum diameter_{i}"],
                 row[f"circularity_{i}"], row[f"curvature_{i}"], row[f"tortuosity_{i}"],
                 row[f"Total length_{i}"], row[f"vol_{i}"]))
    print("model features:", {k: (None if not np.isfinite(v) else round(v, 4)) for k, v in model.items()})
    bad = 0
    for i, e in PHANTOM_EXPECT.items():
        d_eq = row.get(f"equivalent diameter_{i}", NAN)
        c = row.get(f"curvature_{i}", NAN)
        if not np.isfinite(d_eq) or abs(d_eq - e["diam"]) / e["diam"] > 0.30:
            print("WARN  vessel %d equivalent diameter %.2f vs expected ~%.1f" % (i, d_eq, e["diam"]))
            bad += 1
        if not np.isfinite(c) or not (e["curv"][0] <= c <= e["curv"][1]):
            print("WARN  vessel %d curvature %.4f outside expected %s" % (i, c, e["curv"]))
            bad += 1
    if errors:
        print("ERRORS:", errors)
    ok = not errors and all(np.isfinite(v) for v in model.values())
    print("\nFULL SELFTEST:", "pipeline ran, all 5 model features finite" if ok else "FAILED",
          "(%d plausibility warning(s))" % bad)
    return ok


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Vessel features (centerline based) for the post-TIPS OHE model")
    ap.add_argument("--mask", help="multi-label vessel mask (.nii/.nii.gz) or a directory of masks")
    ap.add_argument("--out", default="vessel_out", help="output directory")
    ap.add_argument("--labels", type=int, nargs="+", default=[1, 2, 3],
                    help="vessel labels to process (default 1 2 3 = all that the model needs; use 1 2 3 4 5 for the full table)")
    ap.add_argument("--workers", type=int, default=1, help="parallel vessels per case (each is CPU/RAM heavy)")
    ap.add_argument("--reuse", action="store_true", help="reuse existing mesh/centerline files if present")
    ap.add_argument("--fix-inlet-target", action="store_true",
                    help="remove the inlet from the vmtk target list (NOT the original behaviour)")
    ap.add_argument("--selftest-core", action="store_true", help="test the pure-python parts (no vtk/vmtk needed)")
    ap.add_argument("--selftest", action="store_true", help="run the full pipeline on a synthetic phantom")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if a.selftest_core:
        return 0 if selftest_core() else 1
    if a.selftest:
        selftest_core()
        return 0 if selftest_full(a.out) else 1
    if not a.mask:
        ap.error("--mask is required (or use --selftest-core / --selftest)")

    if os.path.isdir(a.mask):
        files = sorted(os.path.join(a.mask, f) for f in os.listdir(a.mask) if f.endswith((".nii", ".nii.gz")))
    else:
        files = [a.mask]
    if not files:
        print("no NIfTI files found in", a.mask)
        return 2
    os.makedirs(a.out, exist_ok=True)
    summary = os.path.join(a.out, "vessel_features_all.csv")
    n_bad = 0
    for f in files:
        LOG.info("=== %s", f)
        try:
            case, row, model, errors = process_case(f, a.out, tuple(a.labels), a.workers, a.reuse, a.fix_inlet_target)
            append_summary(summary, case, row)
            print("%s: %s" % (case, {k: (None if not np.isfinite(v) else round(v, 4)) for k, v in model.items()}))
            if errors or not all(np.isfinite(v) for v in model.values()):
                n_bad += 1
                print("   incomplete:", errors)
        except Exception as e:
            n_bad += 1
            LOG.error("%s failed: %s", f, e)
            LOG.debug(traceback.format_exc())
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
