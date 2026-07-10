#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, pickle, resource, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from nueronce.engine import functional as F
from nueronce.engine.nueronce_model import NueronceModel, preset_configs
from nueronce.engine.tensor import no_grad


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_pickle(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def canonical_state_hash(checkpoint: dict) -> str:
    h = hashlib.sha256()
    for array in checkpoint['params']:
        a = np.ascontiguousarray(array)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    for state in checkpoint['optimizer']['v']:
        parts = state if isinstance(state, tuple) else (state,)
        for array in parts:
            a = np.ascontiguousarray(array)
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    h.update(str(checkpoint['optimizer']['t']).encode())
    h.update(str(checkpoint['optimizer']['lr']).encode())
    h.update(str(checkpoint['meta']['step']).encode())
    return h.hexdigest()


def build_model(checkpoint: dict):
    cfg = preset_configs()['base_35m']
    model = NueronceModel(cfg)
    params = list(model.parameters())
    if len(params) != len(checkpoint['params']):
        raise RuntimeError('checkpoint parameter tensor count mismatch')
    for p, stored in zip(params, checkpoint['params']):
        p.data[...] = stored
    return model, params, cfg


def lm_loss(model, batch: np.ndarray):
    logits, _ = model.forward(batch)
    return F.cross_entropy(logits[:, :-1].reshape(-1, 256), batch[:, 1:].reshape(-1))


def finite_and_norm(params):
    total = 0.0; count = 0
    for index, p in enumerate(params):
        if p.grad is None:
            continue
        count += 1
        if not np.isfinite(p.grad).all():
            raise FloatingPointError(f'nonfinite gradient at parameter {index} shape={p.shape}')
        g = np.asarray(p.grad, np.float64)
        total += float(np.sum(g * g))
    return count, total ** 0.5


def update(params, state, step, lr, max_norm, beta2=0.999, eps=1e-8, tile=128):
    count, norm = finite_and_norm(params)
    scale = min(1.0, max_norm / (norm + 1e-6))
    correction = max(1.0 - beta2 ** step, eps)
    for i, p in enumerate(params):
        if p.grad is None:
            continue
        g = np.asarray(p.grad, np.float32) * scale
        if p.ndim >= 2:
            w = p.data.reshape(p.shape[0], -1); gg = g.reshape(w.shape)
            vr, vc = state[i]
            vr *= beta2; vc *= beta2
            vr += (1.0 - beta2) * np.mean(gg * gg, axis=1)
            vc += (1.0 - beta2) * np.mean(gg * gg, axis=0)
            rh, ch = vr / correction, vc / correction
            normalizer = max(float(rh.mean()), eps)
            for start in range(0, w.shape[0], tile):
                stop = min(start + tile, w.shape[0])
                u = gg[start:stop] / (np.sqrt(rh[start:stop, None] * ch[None, :] / normalizer) + eps)
                rms = float(np.sqrt(np.mean(u * u)))
                if rms > 1.0:
                    u *= 1.0 / (rms + eps)
                w[start:stop] -= lr * u
        else:
            v = state[i]
            v *= beta2; v += (1.0 - beta2) * g * g
            u = g / (np.sqrt(v / correction) + eps)
            rms = float(np.sqrt(np.mean(u * u))) if u.size else 0.0
            if rms > 1.0:
                u *= 1.0 / (rms + eps)
            p.data -= lr * u
        p.grad = None
    return count, norm, scale


def prepare(args):
    started = time.time()
    checkpoint_hash = sha256_file(args.checkpoint)
    with args.checkpoint.open('rb') as f:
        checkpoint = pickle.load(f)
    model, _, _ = build_model(checkpoint)
    document = args.document.resolve()
    raw = document.read_bytes()
    if len(raw) < args.seq_len:
        raise ValueError('document is shorter than seq_len')
    next_step = int(checkpoint['meta']['step']) + 1
    max_offset = len(raw) - args.seq_len
    offset = args.offset if args.offset is not None else ((next_step * 104729 + args.seed) % (max_offset + 1))
    if offset < 0 or offset > max_offset:
        raise ValueError(f'offset {offset} out of range 0..{max_offset}')
    sequence = raw[offset:offset + args.seq_len]
    batch = np.frombuffer(sequence, dtype=np.uint8).astype(np.int64)[None, :]
    t0 = time.time()
    with no_grad():
        loss = lm_loss(model, batch)
    forward_s = time.time() - t0
    plan = {
        'format': 'engine-base35m-pretrain-pending-v1', 'status': 'prepared',
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_canonical_hash': canonical_state_hash(checkpoint),
        'checkpoint_step': int(checkpoint['meta']['step']), 'next_step': next_step,
        'objective': 'document_byte_pretraining', 'document': str(document),
        'document_sha256': sha256_file(document), 'document_bytes': len(raw),
        'offset': offset, 'sequence': sequence, 'seq_len': args.seq_len,
        'expected_loss': float(loss.item()),
        'lr': args.lr if args.lr is not None else float(checkpoint['optimizer']['lr']),
        'max_grad_norm': args.max_grad_norm, 'seed': args.seed,
    }
    atomic_pickle(args.plan, plan)
    print(json.dumps({
        'status': 'prepared', 'step': next_step, 'objective': plan['objective'],
        'document': str(document), 'document_sha256': plan['document_sha256'],
        'offset': offset, 'seq_len': args.seq_len, 'loss': float(loss.item()),
        'forward_s': forward_s, 'plan': str(args.plan),
        'elapsed_s': time.time() - started,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }), flush=True)


def backward(args):
    started = time.time()
    with args.plan.open('rb') as f:
        plan = pickle.load(f)
    if plan.get('status') != 'prepared':
        raise RuntimeError('plan is not prepared')
    checkpoint_path = Path(plan['checkpoint'])
    if sha256_file(checkpoint_path) != plan['checkpoint_sha256']:
        raise RuntimeError('checkpoint file hash changed after prepare')
    document = Path(plan['document'])
    if sha256_file(document) != plan['document_sha256']:
        raise RuntimeError('document hash changed after prepare')
    with checkpoint_path.open('rb') as f:
        checkpoint = pickle.load(f)
    if canonical_state_hash(checkpoint) != plan['checkpoint_canonical_hash']:
        raise RuntimeError('checkpoint canonical state changed after prepare')
    if int(checkpoint['meta']['step']) != int(plan['checkpoint_step']):
        raise RuntimeError('checkpoint step changed after prepare')
    model, params, cfg = build_model(checkpoint)
    batch = np.frombuffer(plan['sequence'], dtype=np.uint8).astype(np.int64)[None, :]
    for p in params: p.grad = None
    t0 = time.time(); loss = lm_loss(model, batch); forward_s = time.time() - t0
    if abs(float(loss.item()) - float(plan['expected_loss'])) > 1e-5:
        raise RuntimeError('reconstructed loss mismatch')
    t0 = time.time(); loss.backward(); backward_s = time.time() - t0
    step = int(plan['next_step'])
    t0 = time.time()
    count, norm, scale = update(params, checkpoint['optimizer']['v'], step, float(plan['lr']), float(plan['max_grad_norm']))
    update_s = time.time() - t0
    record = {
        'step': step, 'phase': 'pretrain', 'objective': plan['objective'],
        'document': plan['document'], 'document_sha256': plan['document_sha256'],
        'offset': int(plan['offset']), 'sequence_bytes': int(plan['seq_len']),
        'supervised_target_bytes': int(plan['seq_len']) - 1,
        'loss': float(loss.item()), 'grad_norm': norm, 'clip_scale': scale,
        'grad_tensors': count, 'forward_s': forward_s,
        'backward_s': backward_s, 'update_s': update_s,
    }
    checkpoint['params'] = [p.data for p in params]
    checkpoint['optimizer']['t'] = step; checkpoint['optimizer']['lr'] = float(plan['lr'])
    checkpoint['meta']['step'] = step; checkpoint['meta']['phase'] = 'pretrain'
    checkpoint['meta']['training_objective'] = 'mixed_pretraining_and_response_only_sft'
    checkpoint['meta'].setdefault('history', []).append(record)
    canonical_hash = canonical_state_hash(checkpoint)
    t0 = time.time(); atomic_pickle(checkpoint_path, checkpoint); save_s = time.time() - t0
    plan['status'] = 'completed'; plan['result'] = record
    plan['completed_checkpoint_sha256'] = sha256_file(checkpoint_path)
    plan['completed_canonical_state_sha256'] = canonical_hash
    atomic_pickle(args.plan, plan)
    print(json.dumps({
        **record, 'status': 'completed', 'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': plan['completed_checkpoint_sha256'],
        'canonical_state_sha256': canonical_hash, 'save_s': save_s,
        'elapsed_s': time.time() - started,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--document', type=Path, required=True)
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--offset', type=int)
    p.add_argument('--seed', type=int, default=20260710)
    p.add_argument('--lr', type=float)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.set_defaults(func=prepare)
    p = sub.add_parser('backward')
    p.add_argument('--plan', type=Path, required=True)
    p.set_defaults(func=backward)
    args = parser.parse_args(); args.func(args)

if __name__ == '__main__':
    main()
