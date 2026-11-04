# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read and write SMAP _sml.mat localization files."""
import os
import warnings
import numpy as np


def _h5_to_py(f, obj):
    import h5py
    if isinstance(obj, h5py.Dataset):
        mclass = obj.attrs.get('MATLAB_class', b'').decode() if isinstance(obj.attrs.get('MATLAB_class', b''), bytes) else obj.attrs.get('MATLAB_class', '')
        if obj.attrs.get('MATLAB_empty', 0):
            return np.array([])
        data = obj[()]
        if mclass == 'char':
            try:
                return ''.join(chr(c) for c in np.asarray(data).ravel(order='F'))
            except Exception:
                return ''
        if mclass == 'cell':
            out = []
            for ref in np.asarray(data).ravel(order='F'):
                try:
                    out.append(_h5_to_py(f, f[ref]))
                except Exception:
                    out.append(None)
            return out
        if mclass in ('struct',) or data.dtype == object:
            return None
        if mclass == 'logical':
            data = data.astype(bool)
        arr = np.asarray(data)
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.ravel()
        elif arr.ndim >= 2:
            arr = arr.T  # MATLAB column-major
        if arr.size == 1 and arr.ndim <= 1:
            return arr.reshape(()).item() if arr.ndim == 0 else arr[0].item()
        return arr
    out = {}
    for k in obj.keys():
        try:
            out[k] = _h5_to_py(f, obj[k])
        except Exception:
            out[k] = None
    return out


def read_sml(path):
    """Read a SMAP *_sml.mat file."""
    with open(path, 'rb') as fh:
        header = fh.read(128)
    if b'MATLAB 7.3' in header:
        import h5py
        with h5py.File(path, 'r') as f:
            sl = f['saveloc']
            loc = {k: np.asarray(_h5_to_py(f, sl['loc'][k])).ravel() for k in sl['loc'].keys()}
            info = {}
            fitpar = None
            try:
                fileobj = sl['file']
                if isinstance(fileobj, __import__('h5py').Dataset):  # struct array -> refs
                    ref = fileobj[()].ravel()[0]
                    fileobj = f[ref]
                infog = fileobj['info']
                if isinstance(infog, __import__('h5py').Dataset):
                    infog = f[infog[()].ravel()[0]]
                info = _h5_to_py(f, infog)
            except Exception as e:  # pragma: no cover
                warnings.warn('could not read file info: %s' % e)
            try:
                fitpar = _h5_to_py(f, sl['fitparameters'])
            except Exception:
                pass
    else:
        from scipy.io import loadmat
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        sl = m['saveloc']
        loc = {k: np.asarray(getattr(sl.loc, k)).ravel() for k in sl.loc._fieldnames}
        info = {}
        try:
            fi = sl.file
            if isinstance(fi, np.ndarray):
                fi = fi.ravel()[0]
            info = {k: getattr(fi.info, k) for k in fi.info._fieldnames}
        except Exception:
            pass
        fitpar = None
    return {'loc': loc, 'info': info, 'fitparameters': fitpar}


def write_sml(path, loc, info=None, extra=None):
    """Write localizations as a SMAP-compatible *_sml.mat (MATLAB v7 format)."""
    from scipy.io import savemat
    if not path.endswith('_sml.mat'):
        path = os.path.splitext(path)[0] + '_sml.mat'
    keep_double = {'frame'}
    locout = {}
    for k, v in loc.items():
        v = np.asarray(v)
        if v.dtype == bool:
            v = v.astype(np.uint8)
        if k in keep_double:
            locout[k] = v.astype(np.float64).reshape(-1, 1)
        else:
            locout[k] = v.astype(np.float32).reshape(-1, 1) if v.dtype.kind == 'f' else v.reshape(-1, 1)
    if 'filenumber' not in locout:
        locout['filenumber'] = np.ones((len(locout['xnm']), 1), np.uint8)
    if 'channel' not in locout:
        locout['channel'] = np.zeros((len(locout['xnm']), 1), np.float32)
    info = dict(info or {})
    info.setdefault('basefile', path.replace('_sml.mat', ''))
    fileinfo = {'name': path, 'info': info, 'number': 1.0, 'numberOfTif': 0.0}
    saveloc = {'loc': locout, 'file': fileinfo, 'history': np.empty((0, 0))}
    if extra:
        saveloc.update(extra)
    savemat(path, {'saveloc': saveloc, 'fileformat': {'name': 'sml'}},
            do_compression=True, oned_as='column')
    return path
