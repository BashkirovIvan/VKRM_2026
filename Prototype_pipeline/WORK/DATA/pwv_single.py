# -*- coding: utf-8 -*-
"""
Пайплайн расчёта СРПВ (PWV) — одиночный файл.
МК: MSP430i2040  |  fs = 488.28 Гц (SMCLK=2.048 МГц, OSR=256, AVG=16)

Каналы в CSV:
  col 0 — счётчик (не используется для времени)
  col 1 — плетизмограмма грудь (Hall 1)
  col 2 — плетизмограмма рука  (Hall 2)
  col 3 — Hall 3, мусор (игнорируем)
  col 4 — ЭКГ
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.ndimage import binary_dilation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── ГЛОБАЛЬНЫЕ НАСТРОЙКИ (переопределяются через аргументы process_file) ────
DATA_FILE       = 'data4ch_1_2.csv'
DISTANCE_M      = 0.5
FS              = 488.28
TKEO_FACTOR     = 0.6
MIN_RR_SEC      = 0.4
FOOT_START_MS   = 50
FOOT_END_MS     = 400
PTT_MIN_MS      = 20
PTT_MAX_MS      = 200
PPG_CUTOFF_HZ   = 10.0   # верхняя граница bandpass для плетизмограмм
ARTIFACT_Z      = 6.0
ARTIFACT_EXP_MS = 150


# ═════════════════════════════════════════════════════════════════════════════
# 1. ФИЛЬТРЫ
# ═════════════════════════════════════════════════════════════════════════════

def butter_bandpass(data, low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, min(high / nyq, 0.99)], btype='band')
    return filtfilt(b, a, data)


def butter_bandpass_pleth(data, fs=FS, low=0.5, high=PPG_CUTOFF_HZ, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, min(high / nyq, 0.99)], btype='band')
    return filtfilt(b, a, data)


# ═════════════════════════════════════════════════════════════════════════════
# 2. УТИЛИТЫ
# ═════════════════════════════════════════════════════════════════════════════

def fix_zeros(sig):
    """Замена нулевых отсчётов линейной интерполяцией."""
    sig = sig.copy().astype(float)
    zero_mask = (sig == 0)
    if not zero_mask.any():
        return sig
    good_idx = np.where(~zero_mask)[0]
    if len(good_idx) == 0:
        return sig
    sig[zero_mask] = np.interp(np.where(zero_mask)[0], good_idx, sig[good_idx])
    return sig


def mask_artifacts(signal, fs=FS,
                   z_thresh=ARTIFACT_Z,
                   expand_ms=ARTIFACT_EXP_MS):
    """
    Spike-based маскирование артефактов плетизмограммы.
    Использует MAD вместо std — порог не «улетает» от самих выбросов.
    Расширяет маску на ±expand_ms (переходный процесс фильтра).
    """
    med = np.median(signal)
    mad = np.median(np.abs(signal - med)) * 1.4826
    if mad < 1e-10:
        return np.zeros(len(signal), dtype=bool)
    mask = np.abs(signal - med) > z_thresh * mad
    if mask.any():
        expand = int(expand_ms * fs / 1000)
        mask   = binary_dilation(mask, structure=np.ones(2 * expand + 1))
    return mask


def compute_hrv(rpeaks, fs):
    """
    HRV-метрики из NN-интервалов.

    Фильтрация RR:
      1. Физиологические границы: 300–2000 мс (30–200 уд/мин)
      2. Критерий Малика: исключаем интервалы, отличающиеся от предыдущего
         более чем на 20% — убирает артефакты пропущенных/двойных битов,
         которые иначе раздувают SDNN и RMSSD до нефизиологических значений.
    """
    if len(rpeaks) < 3:
        return dict(mean_hr=np.nan, sdnn_ms=np.nan,
                    rmssd_ms=np.nan, pnn50=np.nan)

    rr = np.diff(rpeaks) / fs * 1000          # мс
    rr = rr[(rr > 300) & (rr < 2000)]         # физиологические границы
    if len(rr) < 2:
        return dict(mean_hr=np.nan, sdnn_ms=np.nan,
                    rmssd_ms=np.nan, pnn50=np.nan)

    # Критерий Малика: |RR[i] - RR[i-1]| / RR[i-1] < 0.20
    malik_ok = np.ones(len(rr), dtype=bool)
    for i in range(1, len(rr)):
        if abs(rr[i] - rr[i - 1]) / rr[i - 1] > 0.20:
            malik_ok[i] = False
    nn = rr[malik_ok]

    if len(nn) < 2:
        return dict(mean_hr=np.nan, sdnn_ms=np.nan,
                    rmssd_ms=np.nan, pnn50=np.nan)

    diff_nn  = np.diff(nn)
    mean_hr  = 60000.0 / np.mean(nn)
    sdnn_ms  = float(np.std(nn, ddof=1))
    rmssd_ms = float(np.sqrt(np.mean(diff_nn ** 2)))
    pnn50    = float(np.mean(np.abs(diff_nn) > 50) * 100)
    return dict(mean_hr  = round(mean_hr,  1),
                sdnn_ms  = round(sdnn_ms,  1),
                rmssd_ms = round(rmssd_ms, 1),
                pnn50    = round(pnn50,    1))


# ═════════════════════════════════════════════════════════════════════════════
# 3. ЭКГ
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_ecg(raw, fs,
                   low=0.5, high=40.0):
    ecg = butter_bandpass(raw, low, min(high, fs * 0.45), fs)
    return (ecg - ecg.mean()) / (ecg.std() + 1e-8)


# ═════════════════════════════════════════════════════════════════════════════
# 4. R-ПИКИ (TKEO, робастный порог)
# ═════════════════════════════════════════════════════════════════════════════

def detect_rpeaks(ecg, fs,
                  min_rr_sec=MIN_RR_SEC,
                  tkeo_factor=TKEO_FACTOR):
    """
    Детекция R-пиков через TKEO.
    Порог = tkeo_factor × median(верхней половины TKEO):
    устойчив к единичным выбросам.
    ЭКГ не маскируется — R-пики находятся по всей записи.
    Артефактные биты исключаются позже при поиске foot.
    """
    pos = np.max(ecg) - np.median(ecg)
    neg = np.median(ecg) - np.min(ecg)
    inverted = neg > pos
    ecg_proc = -ecg if inverted else ecg

    tkeo = ecg_proc[1:-1] ** 2 - ecg_proc[:-2] * ecg_proc[2:]
    tkeo = np.insert(tkeo, 0, 0.0)
    win  = max(3, int(0.05 * fs) | 1)
    tkeo_sm = savgol_filter(tkeo, window_length=win, polyorder=2)
    tkeo_sm = np.clip(tkeo_sm, 0, None)

    upper    = tkeo_sm[tkeo_sm >= np.median(tkeo_sm)]
    thresh   = tkeo_factor * np.median(upper)
    min_dist = int(min_rr_sec * fs)
    candidates, _ = find_peaks(tkeo_sm, height=thresh, distance=min_dist)

    half_win = max(1, int(0.03 * fs))
    indices, heights = [], []
    for p in candidates:
        lo  = max(0, p - half_win)
        hi  = min(len(ecg), p + half_win + 1)
        loc = lo + (np.argmin(ecg[lo:hi]) if inverted else np.argmax(ecg[lo:hi]))
        win_bl   = int(1.0 * fs)
        baseline = np.median(ecg[max(0, loc - win_bl):min(len(ecg), loc + win_bl)])
        indices.append(loc)
        heights.append(abs(ecg[loc] - baseline))

    indices = np.array(indices)
    heights = np.array(heights)
    if len(heights) < 2:
        return indices
    keep = heights >= 0.5 * np.median(heights)
    return np.unique(indices[keep])


# ═════════════════════════════════════════════════════════════════════════════
# 5. ДЕТЕКЦИЯ FOOT
# ═════════════════════════════════════════════════════════════════════════════

def detect_feet_chest(signal, rpeaks, fs, artifact_mask=None,
                      start_ms=FOOT_START_MS, end_ms=FOOT_END_MS):
    start_dt = int(start_ms * fs / 1000)
    end_dt   = int(end_ms   * fs / 1000)
    feet = []
    for r in rpeaks:
        lo, hi = r + start_dt, r + end_dt
        if hi >= len(signal):
            continue
        if artifact_mask is not None and artifact_mask[lo:hi].any():
            continue
        seg = signal[lo:hi]
        if len(seg) < 3:
            continue
        peak_rel   = np.argmax(seg)
        search_end = max(1, int(peak_rel * 0.8))
        feet.append(lo + np.argmin(seg[:search_end]))
    return np.array(feet, dtype=int)


def detect_feet_arm(signal, feet_chest, fs, artifact_mask=None,
                    ptt_min_ms=PTT_MIN_MS, ptt_max_ms=PTT_MAX_MS):
    # round() вместо int() — иначе int(20*488.28/1000)=9 сэмплов=18.4мс≠20мс
    start_dt = max(1, round(ptt_min_ms * fs / 1000))
    end_dt   = max(start_dt + 1, round(ptt_max_ms * fs / 1000))
    feet_arm = []
    for fc in feet_chest:
        lo, hi = fc + start_dt, fc + end_dt
        if hi >= len(signal) or lo < 0:
            feet_arm.append(-1)
            continue
        if artifact_mask is not None and artifact_mask[lo:hi].any():
            feet_arm.append(-1)
            continue
        feet_arm.append(lo + np.argmin(signal[lo:hi]))
    return np.array(feet_arm, dtype=int)


# ═════════════════════════════════════════════════════════════════════════════
# 6. PTT / PWV
# ═════════════════════════════════════════════════════════════════════════════

def compute_pwv(feet_chest, feet_arm, fs, distance_m):
    valid = feet_arm >= 0
    fc_v, fa_v = feet_chest[valid], feet_arm[valid]
    if len(fc_v) == 0:
        return np.array([]), np.array([]), fc_v, fa_v

    ptt = (fa_v - fc_v) / fs
    # Фильтруем физиологически невозможные PTT
    # (> 0 уже гарантирует round(), но дублируем явно)
    ptt_min_s = PTT_MIN_MS / 1000
    pos = ptt >= ptt_min_s
    fc_v, fa_v, ptt = fc_v[pos], fa_v[pos], ptt[pos]
    if len(ptt) == 0:
        return np.array([]), np.array([]), fc_v, fa_v

    pwv  = distance_m / ptt
    med  = np.median(ptt)
    mad  = np.median(np.abs(ptt - med)) * 1.4826
    keep = np.abs(ptt - med) / max(mad, 1e-9) < 3.5
    return ptt[keep], pwv[keep], fc_v[keep], fa_v[keep]


# ═════════════════════════════════════════════════════════════════════════════
# 7. ВИЗУАЛИЗАЦИЯ
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(time, ecg, rpeaks, chest, feet_chest,
                 wrist, fc_final, fa_final, ptt_val, pwv_val,
                 artifact_mask=None, title_prefix=''):
    pwv_med = float(np.median(pwv_val)) if len(pwv_val) else float('nan')
    ptt_med = float(np.median(ptt_val)) if len(ptt_val) else float('nan')
    title   = (f'{title_prefix}СРПВ={pwv_med:.2f} м/с  |  '
               f'PTT={ptt_med*1000:.1f} мс  |  Пар:{len(ptt_val)}')

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.32, 0.32, 0.32, 0.04],
        subplot_titles=('ЭКГ + R-пики', 'Грудь + foot', 'Рука + foot', ''),
        vertical_spacing=0.05)

    if artifact_mask is not None and artifact_mask.any():
        diff   = np.diff(artifact_mask.astype(int))
        starts = list(np.where(diff ==  1)[0])
        ends   = list(np.where(diff == -1)[0])
        if artifact_mask[0]:  starts.insert(0, 0)
        if artifact_mask[-1]: ends.append(len(artifact_mask) - 1)
        for s, e in zip(starts, ends):
            for row in range(1, 4):
                fig.add_vrect(x0=time[s], x1=time[e],
                              fillcolor='rgba(220,80,80,0.18)',
                              line_width=0, row=row, col=1)

    fig.add_trace(go.Scattergl(x=time, y=ecg, name='ЭКГ',
        line=dict(color='royalblue', width=0.8)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=time[rpeaks], y=ecg[rpeaks],
        mode='markers', name='R-пики',
        marker=dict(color='red', size=7, symbol='x')), row=1, col=1)

    fig.add_trace(go.Scattergl(x=time, y=chest, name='Грудь',
        line=dict(color='seagreen', width=0.8)), row=2, col=1)
    if len(feet_chest):
        fig.add_trace(go.Scattergl(x=time[feet_chest], y=chest[feet_chest],
            mode='markers', name='Foot грудь',
            marker=dict(color='darkgreen', size=8, symbol='circle-open')), row=2, col=1)
    for i in range(min(5, len(ptt_val))):
        fc, fa = fc_final[i], fa_final[i]
        fig.add_trace(go.Scattergl(
            x=[time[fc], time[fa]], y=[chest[fc], wrist[fa]],
            mode='lines+markers', line=dict(color='black', dash='dot', width=1),
            marker=dict(size=5, color='black'), showlegend=(i == 0),
            name=f'PTT={ptt_val[i]*1000:.0f} мс'), row=2, col=1)

    fig.add_trace(go.Scattergl(x=time, y=wrist, name='Рука',
        line=dict(color='mediumpurple', width=0.8)), row=3, col=1)
    if len(fa_final):
        fa_plot = fa_final[fa_final >= 0]
        if len(fa_plot):
            fig.add_trace(go.Scattergl(x=time[fa_plot], y=wrist[fa_plot],
                mode='markers', name='Foot рука',
                marker=dict(color='darkviolet', size=8, symbol='circle-open')), row=3, col=1)

    fig.add_trace(go.Scattergl(x=time, y=ecg,
        line=dict(color='royalblue', width=0.5), showlegend=False), row=4, col=1)
    fig.update_layout(height=920, title=title, hovermode='x unified',
        xaxis4=dict(rangeslider=dict(visible=True, thickness=0.04),
                    title='Время, с', type='linear'))
    fig.update_yaxes(title_text='z-score',   row=1, col=1)
    fig.update_yaxes(title_text='Амплитуда', row=2, col=1)
    fig.update_yaxes(title_text='Амплитуда', row=3, col=1)
    fig.update_yaxes(visible=False,          row=4, col=1)
    fig.show()


# ═════════════════════════════════════════════════════════════════════════════
# 8. ОСНОВНАЯ ФУНКЦИЯ
# ═════════════════════════════════════════════════════════════════════════════

def process_file(filepath=DATA_FILE,
                 distance_m=None,
                 tkeo_factor=None,
                 ppg_cutoff_hz=None,
                 plot=True,
                 verbose=True):
    """
    Полный пайплайн для одного CSV-файла.
    Параметры distance_m, tkeo_factor, ppg_cutoff_hz берутся из аргументов
    (конфиг пациента) или из глобальных констант если не переданы.
    Возвращает dict с PWV, PTT, HRV и метаданными, или None при ошибке.
    """
    dist    = distance_m   if distance_m   is not None else DISTANCE_M
    tkeo    = tkeo_factor  if tkeo_factor  is not None else TKEO_FACTOR
    ppg_cut = ppg_cutoff_hz if ppg_cutoff_hz is not None else PPG_CUTOFF_HZ

    def log(msg):
        if verbose: print(msg)

    log(f'\n{"="*55}')
    log(f'Файл: {filepath}')

    try:
        df = pd.read_csv(filepath, header=None)
        # Приводим все столбцы к числам, нечисловые строки → NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        # Убираем строки где хотя бы один столбец не распознан
        # (типичная причина: первая строка вида "[0.2, 1, 2, 3, 4]")
        df = df.dropna().reset_index(drop=True)
    except Exception as e:
        log(f'[!] Ошибка чтения: {e}')
        return None

    log(f'Строк: {len(df)}, столбцов: {df.shape[1]}')
    if df.shape[1] < 5 or len(df) < int(FS * 5):
        log('[!] Слишком короткая запись или неверная структура.')
        return None

    time         = np.arange(len(df)) / FS
    log(f'Длина записи: {time[-1]:.1f} с')

    ch_chest_raw = df.iloc[:, 1].values.astype(float)
    ch_wrist_raw = df.iloc[:, 2].values.astype(float)
    ecg_raw      = df.iloc[:, 4].values.astype(float)

    ecg_filt    = preprocess_ecg(ecg_raw, FS)
    ch_chest_bp = butter_bandpass_pleth(fix_zeros(ch_chest_raw), fs=FS,
                                        high=ppg_cut)
    ch_wrist_bp = butter_bandpass_pleth(fix_zeros(ch_wrist_raw), fs=FS,
                                        high=ppg_cut)

    mask_chest    = mask_artifacts(ch_chest_bp)
    mask_wrist    = mask_artifacts(ch_wrist_bp)
    combined_mask = mask_chest | mask_wrist
    art_pct       = 100.0 * combined_mask.mean()
    log(f'Артефактов (маска): {art_pct:.1f}%')

    rpeaks = detect_rpeaks(ecg_filt, FS,
                           min_rr_sec=MIN_RR_SEC,
                           tkeo_factor=tkeo)
    log(f'R-пиков найдено: {len(rpeaks)}')
    if len(rpeaks) < 3:
        log('[!] Слишком мало R-пиков.')
        return None

    hrv = compute_hrv(rpeaks, FS)

    feet_chest = detect_feet_chest(ch_chest_bp, rpeaks, FS,
                                   artifact_mask=combined_mask)
    feet_arm   = detect_feet_arm(ch_wrist_bp, feet_chest, FS,
                                 artifact_mask=combined_mask)
    log(f'Foot груди: {len(feet_chest)},  пар с рукой: {(feet_arm >= 0).sum()}')

    ptt_val, pwv_val, fc_final, fa_final = compute_pwv(
        feet_chest, feet_arm, FS, dist)

    if len(ptt_val) == 0:
        log('[!] Валидных PTT не найдено.')
        return None

    pwv_med = float(np.median(pwv_val))
    ptt_med = float(np.median(ptt_val))

    log(f'\nРезультаты:')
    log(f'  Валидных пар: {len(ptt_val)}')
    log(f'  PTT медиана: {ptt_med*1000:.1f} мс  std: {np.std(ptt_val)*1000:.1f} мс')
    log(f'  PWV медиана: {pwv_med:.2f} м/с  std: {np.std(pwv_val):.2f} м/с')
    log(f'  HR: {hrv["mean_hr"]:.1f} уд/мин  '
        f'RMSSD: {hrv["rmssd_ms"]:.1f} мс  SDNN: {hrv["sdnn_ms"]:.1f} мс')

    result = dict(
        file         = str(filepath),
        duration_s   = round(float(time[-1]), 1),
        artifact_pct = round(art_pct, 1),
        n_rpeaks     = int(len(rpeaks)),
        n_valid      = int(len(ptt_val)),
        ptt_ms       = round(ptt_med * 1000, 1),
        ptt_std_ms   = round(float(np.std(ptt_val)) * 1000, 1),
        pwv_ms       = round(pwv_med, 2),
        pwv_std      = round(float(np.std(pwv_val)), 2),
        # Сырые массивы для CNN и feature-матрицы
        _ptt_arr     = ptt_val,
        _pwv_arr     = pwv_val,
        _rpeaks      = rpeaks,
        _fc_arr      = fc_final,   # foot-индексы груди (для CNN-окон)
        _fa_arr      = fa_final,   # foot-индексы руки
        **hrv,
    )

    if plot:
        plot_results(time, ecg_filt, rpeaks,
                     ch_chest_bp, feet_chest,
                     ch_wrist_bp, fc_final, fa_final,
                     ptt_val, pwv_val,
                     artifact_mask=combined_mask)
    return result


if __name__ == '__main__':
    res = process_file(DATA_FILE, plot=True)
    if res:
        print('\nИтог:', {k: v for k, v in res.items() if not k.startswith('_')})
