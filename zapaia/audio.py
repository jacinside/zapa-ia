"""Carga y ventaneo de audio."""
import numpy as np
from pydub import AudioSegment


def load_mono(path, sr):
    """MP3 -> ndarray float32 mono en [-1, 1]."""
    seg = AudioSegment.from_file(path)
    seg = seg.set_channels(1).set_frame_rate(sr)
    y = np.array(seg.get_array_of_samples()).astype(np.float32)
    full_scale = float(1 << (8 * seg.sample_width - 1))
    return y / full_scale


def windows(y, sr, win_s, hop_s, min_frac=0.5):
    """Yield (idx, start_seg, segmento). Descarta la cola si es < min_frac de ventana."""
    win, hop = int(win_s * sr), int(hop_s * sr)
    if len(y) < int(win * min_frac):
        return
    idx = 0
    for start in range(0, max(len(y) - int(win * min_frac) + 1, 1), hop):
        seg = y[start:start + win]
        if len(seg) < int(win * min_frac):
            break
        yield idx, start / sr, seg
        idx += 1
