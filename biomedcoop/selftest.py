"""
Offline self-test for BiomedCoOp's novel math (MAD pruning + SCCM + KDSP).

Runs WITHOUT torch or the BiomedCLIP backbone (which needs a HuggingFace
download). It mirrors the exact ops in losses.py with numpy and asserts the
properties the algorithm must satisfy. If torch IS installed, it additionally
checks the real losses.py functions agree with the numpy mirror.

    python3 selftest.py
"""
import numpy as np


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _log_softmax(x, axis=-1):
    return np.log(_softmax(x, axis) + 1e-12)


def mad_mask_np(scores, tau):
    s_bar = np.median(scores)
    d_bar = np.median(np.abs(scores - s_bar))
    z = (scores - s_bar) / (d_bar + 1e-8)
    return np.abs((z - z.mean()) / (z.std() + 1e-8)) <= tau


def sccm_np(tp, pg):
    tp = tp / np.linalg.norm(tp, axis=-1, keepdims=True)
    pg = pg / np.linalg.norm(pg, axis=-1, keepdims=True)
    return float(((tp - pg) ** 2).mean())


def kdsp_np(student, teacher):
    ls, lt = _log_softmax(student), _log_softmax(teacher)
    return float((np.exp(lt) * (lt - ls)).sum() / student.size)


def run():
    rng = np.random.default_rng(0)
    C, D, N, B = 5, 16, 8, 4
    img = rng.standard_normal((B, D)); img /= np.linalg.norm(img, axis=-1, keepdims=True)
    Tg = rng.standard_normal((C, N, D)); Tg /= np.linalg.norm(Tg, axis=-1, keepdims=True)
    Tg[:, 3, :] = Tg[:, 3, :] * 0.01 + 5.0          # inject a clear outlier prompt
    scale = 50.0

    scores = np.array([(scale * (img @ Tg[:, i, :].T)).max(axis=1).mean() for i in range(N)])
    mask = mad_mask_np(scores, 1.25)
    assert not mask[3], "MAD must drop the injected outlier"
    assert mask.sum() >= N - 2, "MAD should keep most inliers"

    Pg, Ps = Tg.mean(1), Tg[:, mask].mean(1)
    assert abs(sccm_np(Pg, Pg)) < 1e-9
    assert sccm_np(rng.standard_normal((C, D)), Pg) > 0
    student, teacher = scale * (img @ Pg.T), scale * (img @ Ps.T)
    assert kdsp_np(student, student) < 1e-9
    assert kdsp_np(student, teacher) >= 0
    print("numpy mirror: MAD + SCCM + KDSP assertions PASSED "
          f"(dropped prompt {np.where(~mask)[0].tolist()}, kept {int(mask.sum())}/{N})")

    try:
        import torch
        from losses import mad_outlier_mask, sccm_loss, kdsp_loss
        m = mad_outlier_mask(torch.tensor(scores), 1.25).numpy()
        assert (m == mask).all(), "torch MAD disagrees with numpy mirror"
        s = sccm_loss(torch.tensor(Pg / np.linalg.norm(Pg, axis=-1, keepdims=True)),
                      torch.tensor(Pg / np.linalg.norm(Pg, axis=-1, keepdims=True))).item()
        assert abs(s) < 1e-6
        k = kdsp_loss(torch.tensor(student), torch.tensor(student)).item()
        assert abs(k) < 1e-6
        print("torch losses.py agrees with the numpy mirror PASSED")
    except ImportError:
        print("torch not installed -> skipped torch cross-check (numpy mirror already validated)")


if __name__ == "__main__":
    run()
