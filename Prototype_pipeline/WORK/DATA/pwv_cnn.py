# -*- coding: utf-8 -*-
"""
CNN для оценки PTT из сырых сигналов.

Подход: самообучение (self-supervised).
  - «Метки» PTT берутся из алгоритмического пайплайна (pwv_single.py)
    на качественных битах (quality_ok=True).
  - Вход модели: трёхканальное окно [ЭКГ, грудь, рука], 300 мс до и после
    foot-точки груди.
  - Выход: PTT в мс.
  - Применение: оценка PTT на зашумлённых сигналах, где foot-детекция
    нестабильна (P01 с аритмией).

Запуск в Colab:
    from pwv_cnn import build_dataset, train_cnn, evaluate_cnn, predict_pwv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt

# ─── ПАРАМЕТРЫ ───────────────────────────────────────────────────────────────
FS           = 488.28          # частота дискретизации
WIN_BEFORE_MS = 100            # окно до foot-точки груди, мс
WIN_AFTER_MS  = 500            # окно после foot-точки груди, мс
WIN_SAMPLES   = int((WIN_BEFORE_MS + WIN_AFTER_MS) * FS / 1000)

TRAIN_SPLIT  = 0.8             # доля обучения (остальное — тест)
EPOCHS       = 60
BATCH_SIZE   = 32
LEARNING_RATE = 1e-3


# ═════════════════════════════════════════════════════════════════════════════
# 1. СБОРКА ДАТАСЕТА
# ═════════════════════════════════════════════════════════════════════════════

def _bandpass(data, low=0.5, high=40.0, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, min(high/nyq, 0.99)], btype='band')
    return filtfilt(b, a, data)

def _bandpass_pleth(data, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [0.5/nyq, min(10.0/nyq, 0.99)], btype='band')
    return filtfilt(b, a, data)

def _fix_zeros(sig):
    sig = sig.copy().astype(float)
    mask = (sig == 0)
    if not mask.any(): return sig
    good = np.where(~mask)[0]
    if len(good) == 0: return sig
    sig[mask] = np.interp(np.where(mask)[0], good, sig[good])
    return sig

def _normalize(seg):
    """Min-max нормализация сегмента в [-1, 1]."""
    mn, mx = seg.min(), seg.max()
    if mx - mn < 1e-10:
        return np.zeros_like(seg)
    return 2 * (seg - mn) / (mx - mn) - 1


def extract_beats_from_file(filepath, distance_m, tkeo_factor,
                             win_before_ms=WIN_BEFORE_MS,
                             win_after_ms=WIN_AFTER_MS):
    """
    Извлекает обучающие примеры из одного CSV-файла.
    Использует process_file для получения валидных foot-пар и PTT.
    Возвращает (X, y):
        X — массив (N, WIN_SAMPLES, 3): [ЭКГ, грудь, рука]
        y — массив (N,): PTT в мс
    """
    from pwv_single import (process_file, butter_bandpass_pleth,
                             preprocess_ecg, FS as _FS)

    result = process_file(
        filepath=str(filepath),
        distance_m=distance_m,
        tkeo_factor=tkeo_factor,
        plot=False,
        verbose=False,
    )
    if result is None or len(result.get('_ptt_arr', [])) == 0:
        return np.empty((0, WIN_SAMPLES, 3)), np.empty(0)

    ptt_arr    = result['_ptt_arr']        # секунды
    fc_arr     = result.get('_fc_arr', np.array([], dtype=int))
    fa_arr     = result.get('_fa_arr', np.array([], dtype=int))

    if len(fc_arr) == 0:
        return np.empty((0, WIN_SAMPLES, 3)), np.empty(0)

    # Загружаем сигналы заново для формирования окон
    df = pd.read_csv(filepath, header=None)
    df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
    ecg_raw   = df.iloc[:, 4].values.astype(float)
    chest_raw = df.iloc[:, 1].values.astype(float)
    wrist_raw = df.iloc[:, 2].values.astype(float)

    ecg_f   = preprocess_ecg(ecg_raw, FS)
    chest_f = butter_bandpass_pleth(_fix_zeros(chest_raw))
    wrist_f = butter_bandpass_pleth(_fix_zeros(wrist_raw))

    wb = int(win_before_ms * FS / 1000)
    wa = int(win_after_ms  * FS / 1000)
    wlen = wb + wa

    X_list, y_list = [], []
    for fc, ptt_s in zip(fc_arr, ptt_arr):
        lo = fc - wb
        hi = fc + wa
        if lo < 0 or hi >= len(ecg_f):
            continue
        seg_ecg   = _normalize(ecg_f[lo:hi])
        seg_chest = _normalize(chest_f[lo:hi])
        seg_wrist = _normalize(wrist_f[lo:hi])
        X_list.append(np.stack([seg_ecg, seg_chest, seg_wrist], axis=-1))
        y_list.append(ptt_s * 1000)   # → мс

    if not X_list:
        return np.empty((0, wlen, 3)), np.empty(0)

    return np.array(X_list, dtype='float32'), np.array(y_list, dtype='float32')


def build_dataset(config_path, data_dir, quality_csv=None):
    """
    Собирает датасет из всех качественных сигналов.

    quality_csv: путь к features_quality_ok.csv (из batch_pwv).
                 Если задан — используем только quality_ok сигналы.
    Возвращает (X, y, meta) где meta — список dict с patient_id и file.
    """
    import json

    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    # Определяем какие файлы брать
    if quality_csv and Path(quality_csv).exists():
        df_ok = pd.read_csv(quality_csv)
        ok_files = set(df_ok['file'].apply(lambda p: Path(p).name))
        print(f'Качественных сигналов: {len(ok_files)}')
    else:
        ok_files = None
        print('quality_csv не задан — берём все сигналы')

    data_path = Path(data_dir)
    all_X, all_y, all_meta = [], [], []

    for patient in config['patients']:
        pid      = patient['id']
        dist     = patient['distance_m']
        tkeo     = patient['tkeo_factor']

        for fname in patient['files']:
            # Ищем файл
            fpath = None
            for candidate in [data_path/fname, data_path/(fname+'.csv')]:
                if candidate.exists():
                    fpath = candidate
                    break
            if fpath is None:
                continue

            if ok_files and fpath.name not in ok_files:
                continue

            print(f'  Извлечение битов: {fpath.name} ... ', end='', flush=True)
            X, y = extract_beats_from_file(fpath, dist, tkeo)
            print(f'{len(y)} битов')

            for i in range(len(y)):
                all_meta.append({'patient_id': pid, 'file': str(fpath)})
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        print('[!] Датасет пуст.')
        return None, None, None

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Фильтруем физически невозможные PTT
    valid = (y >= 20) & (y <= 250)
    X, y = X[valid], y[valid]
    all_meta = [m for m, v in zip(all_meta, valid) if v]

    print(f'\nИтого битов: {len(y)}')
    print(f'PTT: медиана={np.median(y):.1f} мс, '
          f'min={y.min():.1f}, max={y.max():.1f}')
    return X, y, all_meta


# ═════════════════════════════════════════════════════════════════════════════
# 2. МОДЕЛЬ
# ═════════════════════════════════════════════════════════════════════════════

def build_cnn(input_shape, name='PWV_CNN'):
    """
    1D CNN для регрессии PTT.

    Архитектура:
        Вход: (WIN_SAMPLES, 3) — три канала
        Conv1D блоки с уменьшающимся kernel (детекция грубых → тонких паттернов)
        Global Average Pooling → компактное представление
        Dense головы → PTT в мс
    """
    try:
        from tensorflow.keras import layers, Model, Input
        from tensorflow.keras.regularizers import l2
    except ImportError:
        print('[!] tensorflow не установлен: pip install tensorflow')
        return None

    inp = Input(shape=input_shape, name='signal_input')

    # Блок 1 — крупные паттерны (40–80 мс окно)
    x = layers.Conv1D(32, kernel_size=39, padding='same',
                      activation='relu', kernel_regularizer=l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Блок 2 — средние паттерны (~20 мс)
    x = layers.Conv1D(64, kernel_size=19, padding='same',
                      activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Блок 3 — тонкие паттерны (~5 мс)
    x = layers.Conv1D(128, kernel_size=9, padding='same',
                      activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Блок 4
    x = layers.Conv1D(64, kernel_size=5, padding='same',
                      activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Регрессионная голова
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, name='ptt_ms')(x)

    model = Model(inp, out, name=name)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 3. ОБУЧЕНИЕ
# ═════════════════════════════════════════════════════════════════════════════

def train_cnn(X, y, meta=None,
              test_split=1 - TRAIN_SPLIT,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              lr=LEARNING_RATE,
              save_path='pwv_cnn.h5'):
    """
    Обучает CNN на датасете (X, y).
    Стратегия разбивки: leave-one-patient-out если meta задан,
    иначе случайный split.
    Возвращает (model, history, X_test, y_test, meta_test).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import (EarlyStopping,
                                                ReduceLROnPlateau,
                                                ModelCheckpoint)
        from sklearn.model_selection import train_test_split
    except ImportError:
        print('[!] Установи: pip install tensorflow scikit-learn')
        return None, None, None, None, None

    tf.random.set_seed(42)
    np.random.seed(42)

    # ── Разбивка train/test ───────────────────────────────────────────────
    if meta:
        # Leave-one-patient-out: тест = пациент с наименьшим числом битов
        pids = [m['patient_id'] for m in meta]
        unique, counts = np.unique(pids, return_counts=True)
        test_pid = unique[np.argmin(counts)]
        print(f'Стратегия: leave-one-patient-out, тест = {test_pid}')
        test_mask  = np.array([m['patient_id'] == test_pid for m in meta])
        train_mask = ~test_mask
    else:
        idx = np.random.permutation(len(y))
        n_test = int(len(y) * test_split)
        test_mask  = np.zeros(len(y), dtype=bool)
        test_mask[idx[:n_test]] = True
        train_mask = ~test_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    meta_test = [m for m, t in zip(meta or [{}]*len(y), test_mask) if t]

    print(f'Обучение: {len(y_train)} битов  |  Тест: {len(y_test)} битов')

    # ── Модель ────────────────────────────────────────────────────────────
    model = build_cnn(input_shape=(X.shape[1], X.shape[2]))
    if model is None:
        return None, None, None, None, None

    model.compile(
        optimizer=Adam(lr),
        loss='huber',          # устойчив к выбросам PTT
        metrics=['mae'],
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True,
                      monitor='val_mae'),
        ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-5,
                          monitor='val_mae'),
        ModelCheckpoint(save_path, save_best_only=True, monitor='val_mae'),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print(f'\nМодель сохранена: {save_path}')
    return model, history, X_test, y_test, meta_test


# ═════════════════════════════════════════════════════════════════════════════
# 4. ОЦЕНКА
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_cnn(model, X_test, y_test, meta_test=None):
    """
    Оценка модели: MAE, RMSE, scatter-plot.
    Возвращает dict с метриками.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print('[!] plotly не установлен')
        return {}

    pred = model.predict(X_test, verbose=0).flatten()
    mae  = float(np.mean(np.abs(pred - y_test)))
    rmse = float(np.sqrt(np.mean((pred - y_test)**2)))
    r2   = float(1 - np.sum((pred - y_test)**2) /
                 np.sum((y_test - np.mean(y_test))**2))

    print(f'\nМетрики на тесте:')
    print(f'  MAE  = {mae:.2f} мс')
    print(f'  RMSE = {rmse:.2f} мс')
    print(f'  R²   = {r2:.3f}')

    # Bland-Altman + Scatter
    colors = {'P01': '#e74c3c', 'P02': '#2980b9', 'P03': '#27ae60'}
    pid_list = [m.get('patient_id','?') for m in (meta_test or [{}]*len(y_test))]

    fig = go.Figure()

    for pid in set(pid_list):
        mask = [p == pid for p in pid_list]
        yt = y_test[mask]; yp = pred[mask]
        fig.add_trace(go.Scatter(
            x=yt, y=yp, mode='markers', name=pid,
            marker=dict(color=colors.get(pid,'#888'), size=7, opacity=0.7),
            hovertemplate=f'{pid}: истинный=%{{x:.1f}}, предсказан=%{{y:.1f}}',
        ))

    lim = [y_test.min() - 5, y_test.max() + 5]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode='lines',
        line=dict(color='black', dash='dash'), name='Идеал', showlegend=True))
    fig.update_layout(
        title=f'CNN: предсказанный vs истинный PTT  |  MAE={mae:.1f} мс, R²={r2:.3f}',
        xaxis_title='Алгоритмический PTT, мс',
        yaxis_title='CNN PTT, мс',
        height=500,
    )
    fig.show()

    return dict(mae=mae, rmse=rmse, r2=r2)


def plot_training_history(history):
    """График потерь и MAE при обучении."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Loss (Huber)', 'MAE, мс'))
    for col, metric in enumerate(['loss', 'mae'], start=1):
        fig.add_trace(go.Scatter(y=history.history[metric],
            name='train', line=dict(color='royalblue')), row=1, col=col)
        fig.add_trace(go.Scatter(y=history.history[f'val_{metric}'],
            name='val',   line=dict(color='tomato', dash='dash')),
            row=1, col=col)
    fig.update_layout(height=380, title='История обучения CNN')
    fig.show()


# ═════════════════════════════════════════════════════════════════════════════
# 5. ИНФЕРЕНС
# ═════════════════════════════════════════════════════════════════════════════

def predict_pwv(model, filepath, distance_m, tkeo_factor,
                win_before_ms=WIN_BEFORE_MS, win_after_ms=WIN_AFTER_MS):
    """
    Предсказывает PTT и PWV для нового файла без алгоритмического пайплайна.
    Использует только R-пики для разметки окон, PTT берётся из CNN.
    """
    from pwv_single import (preprocess_ecg, detect_rpeaks,
                             butter_bandpass_pleth, FS as _FS)

    df = pd.read_csv(filepath, header=None)
    time      = np.arange(len(df)) / FS
    ecg_raw   = df.iloc[:, 4].values.astype(float)
    chest_raw = df.iloc[:, 1].values.astype(float)
    wrist_raw = df.iloc[:, 2].values.astype(float)

    ecg_f   = preprocess_ecg(ecg_raw, FS)
    chest_f = butter_bandpass_pleth(_fix_zeros(chest_raw))
    wrist_f = butter_bandpass_pleth(_fix_zeros(wrist_raw))

    rpeaks = detect_rpeaks(ecg_f, FS, tkeo_factor=tkeo_factor)

    wb   = int(win_before_ms * FS / 1000)
    wa   = int(win_after_ms  * FS / 1000)
    wlen = wb + wa

    # Строим окна вокруг каждого R-пика (вместо foot-точки)
    X_list, r_list = [], []
    for r in rpeaks:
        lo, hi = r - wb, r + wa
        if lo < 0 or hi >= len(ecg_f):
            continue
        seg_ecg   = _normalize(ecg_f[lo:hi])
        seg_chest = _normalize(chest_f[lo:hi])
        seg_wrist = _normalize(wrist_f[lo:hi])
        X_list.append(np.stack([seg_ecg, seg_chest, seg_wrist], axis=-1))
        r_list.append(r)

    if not X_list:
        print('[!] Нет окон для инференса.')
        return None

    X_inf = np.array(X_list, dtype='float32')
    ptt_pred = model.predict(X_inf, verbose=0).flatten()   # мс

    # Фильтруем невалидные
    valid = (ptt_pred >= 20) & (ptt_pred <= 250)
    ptt_valid = ptt_pred[valid]
    pwv_valid = distance_m / (ptt_valid / 1000)

    # MAD-фильтр
    if len(ptt_valid) > 3:
        med = np.median(ptt_valid)
        mad = np.median(np.abs(ptt_valid - med)) * 1.4826
        keep = np.abs(ptt_valid - med) / max(mad, 1e-9) < 3.5
        ptt_valid = ptt_valid[keep]
        pwv_valid = pwv_valid[keep]

    if len(pwv_valid) == 0:
        print('[!] Нет валидных предсказаний.')
        return None

    result = dict(
        ptt_ms     = round(float(np.median(ptt_valid)), 1),
        ptt_std_ms = round(float(np.std(ptt_valid)), 1),
        pwv_ms     = round(float(np.median(pwv_valid)), 2),
        pwv_std    = round(float(np.std(pwv_valid)), 2),
        n_beats    = int(len(pwv_valid)),
    )
    print(f'CNN инференс:')
    print(f'  PTT = {result["ptt_ms"]:.1f} ± {result["ptt_std_ms"]:.1f} мс')
    print(f'  PWV = {result["pwv_ms"]:.2f} ± {result["pwv_std"]:.2f} м/с')
    print(f'  Битов: {result["n_beats"]}')
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 6. ПОЛНЫЙ ИНФЕРЕНС С ВИЗУАЛИЗАЦИЕЙ
# ═════════════════════════════════════════════════════════════════════════════

def run_inference(model, filepath, distance_m, tkeo_factor,
                  patient_id='', show_plot=True):
    """
    Запускает оба метода оценки PTT/СРПВ на новом файле и сравнивает их:
      - Алгоритмический (foot-to-foot)
      - CNN (предсказание по трёхканальным окнам)

    Параметры:
        model       — обученная Keras-модель
        filepath    — путь к CSV-файлу
        distance_m  — расстояние грудь–запястье, м
        tkeo_factor — порог детекции R-пиков (из config.json пациента)
        patient_id  — строка-идентификатор (для заголовка графика)
        show_plot   — показывать ли сравнительный график

    Возвращает dict с полями:
        algo_ptt_ms, algo_pwv_ms  — алгоритмический результат
        cnn_ptt_ms,  cnn_pwv_ms   — результат CNN
        agreement_ms              — разница медиан (|algo − cnn|)
        n_algo, n_cnn             — число валидных пар/предсказаний
    """
    from pwv_single import (process_file, preprocess_ecg,
                             butter_bandpass_pleth, detect_rpeaks,
                             detect_feet_chest, mask_artifacts)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        show_plot = False

    filepath = str(filepath)
    print(f'\n{"="*55}')
    print(f'Инференс: {Path(filepath).name}  [{patient_id}]')

    # ── 1. Алгоритмический пайплайн ──────────────────────────────────────
    algo = process_file(filepath, distance_m=distance_m,
                        tkeo_factor=tkeo_factor, plot=False, verbose=False)

    algo_ptt = float(np.median(algo['_ptt_arr'])) * 1000 if algo else np.nan
    algo_pwv = float(np.median(algo['_pwv_arr']))          if algo else np.nan
    algo_n   = int(algo['n_valid'])                        if algo else 0
    fc_arr   = algo['_fc_arr']                             if algo else np.array([])

    print(f'  Алгоритм:  PTT={algo_ptt:.1f} мс  '
          f'PWV={algo_pwv:.2f} м/с  (n={algo_n})')

    # ── 2. Загрузка и предобработка сигналов ────────────────────────────
    df = pd.read_csv(filepath, header=None)
    df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
    time      = np.arange(len(df)) / FS
    ecg_raw   = df.iloc[:, 4].values.astype(float)
    chest_raw = df.iloc[:, 1].values.astype(float)
    wrist_raw = df.iloc[:, 2].values.astype(float)

    ecg_f   = preprocess_ecg(ecg_raw, FS)
    chest_f = butter_bandpass_pleth(_fix_zeros(chest_raw))
    wrist_f = butter_bandpass_pleth(_fix_zeros(wrist_raw))

    # Маска артефактов
    art_mask = mask_artifacts(chest_f) | mask_artifacts(wrist_f)

    # R-пики (независимо от алгоритмического результата)
    rpeaks = detect_rpeaks(ecg_f, FS, tkeo_factor=tkeo_factor)

    # Foot-точки груди (для центрирования CNN-окон)
    feet_c = detect_feet_chest(chest_f, rpeaks, FS, artifact_mask=art_mask)

    # ── 3. CNN-инференс по foot-окнам ────────────────────────────────────
    wb   = int(WIN_BEFORE_MS * FS / 1000)
    wa   = int(WIN_AFTER_MS  * FS / 1000)
    wlen = wb + wa

    X_inf, beat_times = [], []
    for fc in feet_c:
        lo, hi = fc - wb, fc + wa
        if lo < 0 or hi >= len(ecg_f):
            continue
        if art_mask[lo:hi].any():
            continue
        seg_e = _normalize(ecg_f[lo:hi])
        seg_c = _normalize(chest_f[lo:hi])
        seg_w = _normalize(wrist_f[lo:hi])
        X_inf.append(np.stack([seg_e, seg_c, seg_w], axis=-1))
        beat_times.append(time[fc])

    cnn_ptt_all  = np.array([])
    cnn_pwv_all  = np.array([])
    cnn_ptt      = np.nan
    cnn_pwv      = np.nan
    cnn_n        = 0

    if X_inf:
        X_inf = np.array(X_inf, dtype='float32')

        # Подгоняем размер если нужно (padding/truncation)
        target_len = model.input_shape[1]
        if X_inf.shape[1] != target_len:
            if X_inf.shape[1] > target_len:
                X_inf = X_inf[:, :target_len, :]
            else:
                pad = target_len - X_inf.shape[1]
                X_inf = np.pad(X_inf, ((0,0),(0,pad),(0,0)))

        ptt_pred = model.predict(X_inf, verbose=0).flatten()

        # Фильтр физиологических границ
        valid = (ptt_pred > 25) & (ptt_pred <= 250)
        ptt_pred = ptt_pred[valid]
        beat_times_valid = [t for t, v in zip(beat_times, valid) if v]

        if len(ptt_pred) > 3:
            # MAD-фильтр выбросов
            med = np.median(ptt_pred)
            mad = np.median(np.abs(ptt_pred - med)) * 1.4826
            keep = np.abs(ptt_pred - med) / max(mad, 1e-9) < 3.5
            ptt_pred = ptt_pred[keep]
            beat_times_valid = [t for t, k in zip(beat_times_valid, keep) if k]

        if len(ptt_pred) > 0:
            cnn_ptt_all = ptt_pred
            cnn_pwv_all = distance_m / (ptt_pred / 1000)
            cnn_ptt     = float(np.median(cnn_ptt_all))
            cnn_pwv     = float(np.median(cnn_pwv_all))
            cnn_n       = len(cnn_ptt_all)

    print(f'  CNN:       PTT={cnn_ptt:.1f} мс  '
          f'PWV={cnn_pwv:.2f} м/с  (n={cnn_n})')

    agreement = (abs(algo_ptt - cnn_ptt)
                 if (algo is not None) and not np.isnan(cnn_ptt)
                 else np.nan)
    print(f'  Расхождение алгоритм↔CNN: {agreement:.1f} мс' if not np.isnan(agreement)
          else '  Расхождение: н/д')

    # ── 4. Визуализация ──────────────────────────────────────────────────
    if show_plot and not np.isnan(cnn_ptt):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'PTT по ударам: алгоритм vs CNN',
                'СРПВ по ударам: алгоритм vs CNN',
                'Распределение PTT',
                'Распределение СРПВ',
            ),
            vertical_spacing=0.15, horizontal_spacing=0.12,
        )

        # PTT по времени
        if algo and len(algo['_ptt_arr']) > 0:
            algo_times = time[fc_arr] if len(fc_arr) else []
            fig.add_trace(go.Scatter(
                x=algo_times, y=algo['_ptt_arr'] * 1000,
                mode='markers', name='Алгоритм PTT',
                marker=dict(color='seagreen', size=7, symbol='circle'),
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=beat_times_valid if beat_times_valid else [],
            y=cnn_ptt_all,
            mode='markers', name='CNN PTT',
            marker=dict(color='royalblue', size=7, symbol='diamond'),
        ), row=1, col=1)

        # Медианы PTT
        fig.add_hline(y=algo_ptt, line=dict(color='seagreen', dash='dash', width=1.5),
                      annotation_text=f'Алг. медиана={algo_ptt:.0f}мс',
                      annotation_position='top left', row=1, col=1)
        fig.add_hline(y=cnn_ptt, line=dict(color='royalblue', dash='dot', width=1.5),
                      annotation_text=f'CNN медиана={cnn_ptt:.0f}мс',
                      annotation_position='bottom right', row=1, col=1)

        # СРПВ по времени
        if algo and len(algo['_pwv_arr']) > 0:
            fig.add_trace(go.Scatter(
                x=algo_times, y=algo['_pwv_arr'],
                mode='markers', name='Алгоритм СРПВ',
                marker=dict(color='seagreen', size=7),
                showlegend=False,
            ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=beat_times_valid if beat_times_valid else [],
            y=cnn_pwv_all,
            mode='markers', name='CNN СРПВ',
            marker=dict(color='royalblue', size=7, symbol='diamond'),
            showlegend=False,
        ), row=1, col=2)

        # Гистограммы PTT
        if algo and len(algo['_ptt_arr']) > 0:
            fig.add_trace(go.Histogram(
                x=algo['_ptt_arr'] * 1000, name='Алгоритм PTT',
                marker_color='seagreen', opacity=0.6, nbinsx=20,
                showlegend=False,
            ), row=2, col=1)

        fig.add_trace(go.Histogram(
            x=cnn_ptt_all, name='CNN PTT',
            marker_color='royalblue', opacity=0.6, nbinsx=20,
            showlegend=False,
        ), row=2, col=1)

        # Гистограммы СРПВ
        if algo and len(algo['_pwv_arr']) > 0:
            fig.add_trace(go.Histogram(
                x=algo['_pwv_arr'], name='Алгоритм СРПВ',
                marker_color='seagreen', opacity=0.6, nbinsx=20,
                showlegend=False,
            ), row=2, col=2)

        fig.add_trace(go.Histogram(
            x=cnn_pwv_all, name='CNN СРПВ',
            marker_color='royalblue', opacity=0.6, nbinsx=20,
            showlegend=False,
        ), row=2, col=2)

        title = (f'{patient_id} | {Path(filepath).name}<br>'
                 f'Алгоритм: PTT={algo_ptt:.0f}мс, СРПВ={algo_pwv:.2f}м/с  |  '
                 f'CNN: PTT={cnn_ptt:.0f}мс, СРПВ={cnn_pwv:.2f}м/с')
        fig.update_layout(height=620, title=title,
                          barmode='overlay', hovermode='x unified')
        fig.update_xaxes(title_text='Время, с', row=1, col=1)
        fig.update_xaxes(title_text='Время, с', row=1, col=2)
        fig.update_xaxes(title_text='PTT, мс',  row=2, col=1)
        fig.update_xaxes(title_text='СРПВ, м/с', row=2, col=2)
        fig.update_yaxes(title_text='PTT, мс',   row=1, col=1)
        fig.update_yaxes(title_text='СРПВ, м/с', row=1, col=2)
        fig.show()

    return dict(
        file        = filepath,
        patient_id  = patient_id,
        algo_ptt_ms = round(algo_ptt, 1),
        algo_pwv_ms = round(algo_pwv, 2),
        algo_n      = algo_n,
        cnn_ptt_ms  = round(cnn_ptt,  1),
        cnn_pwv_ms  = round(cnn_pwv,  2),
        cnn_n       = cnn_n,
        agreement_ms = round(agreement, 1) if not np.isnan(agreement) else None,
    )


def run_inference_batch(model, config_path, data_dir, show_plot=True):
    """
    Запускает инференс по всем файлам из config.json.
    Возвращает DataFrame со сравнением алгоритм vs CNN по каждому файлу.
    """
    import json

    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    data_path = Path(data_dir)
    results = []

    for patient in config['patients']:
        pid   = patient['id']
        dist  = patient['distance_m']
        tkeo  = patient['tkeo_factor']

        print(f'\n{"#"*55}')
        print(f'  ПАЦИЕНТ {pid}')

        for fname in patient['files']:
            fpath = None
            for c in [data_path/fname, data_path/(fname+'.csv')]:
                if c.exists():
                    fpath = c; break
            if fpath is None:
                continue

            r = run_inference(model, fpath, dist, tkeo,
                              patient_id=pid, show_plot=show_plot)
            results.append(r)

    df = pd.DataFrame(results)

    print(f'\n{"="*55}')
    print('Итоговая таблица (алгоритм vs CNN):')
    cols = ['patient_id','algo_ptt_ms','cnn_ptt_ms','agreement_ms',
            'algo_pwv_ms','cnn_pwv_ms','algo_n','cnn_n']
    print(df[cols].to_string(index=False))
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 7. ТОЧКА ВХОДА (COLAB)
# ═════════════════════════════════════════════════════════════════════════════
#
# ── Одиночный файл ──────────────────────────────────────────────────────────
# from pwv_cnn import run_inference
#
# result = run_inference(
#     model      = model2,           # обученная модель
#     filepath   = f'{DATA_DIR}/data4ch_0_7.csv',
#     distance_m = 0.57,             # из config.json для P01
#     tkeo_factor = 0.8,
#     patient_id  = 'P01',
#     show_plot   = True,
# )
#
# ── Все файлы из config.json ────────────────────────────────────────────────
# from pwv_cnn import run_inference_batch
#
# df_inference = run_inference_batch(
#     model       = model2,
#     config_path = CONFIG,
#     data_dir    = DATA_DIR,
#     show_plot   = False,   # True — график на каждый файл
# )
# display(df_inference)
