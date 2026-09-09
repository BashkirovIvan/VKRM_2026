# -*- coding: utf-8 -*-
"""
Батч-обработка всех пациентов из config.json.

Запуск:
    python batch_pwv.py                        # без графиков
    python batch_pwv.py --plot                 # с графиками
    python batch_pwv.py --data_dir /path/data  # другая папка с CSV

Выходные файлы (в output_dir из конфига):
    results_per_signal.csv   — одна строка на сигнал
    results_per_patient.csv  — агрегированные показатели по пациенту
    features_matrix.csv      — расширенная feature-матрица для ML
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pwv_single import process_file, FS, compute_hrv

warnings.filterwarnings('ignore')


# ─── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ─────────────────────────────────────────────────

def resolve_path(data_dir: Path, fname: str):
    """Находит файл: пробует имя как есть, затем с .csv."""
    for candidate in [data_dir / fname, data_dir / (fname + '.csv')]:
        if candidate.exists():
            return candidate
    return None


def extract_features(result: dict) -> dict:
    """
    Строит расширенный feature-вектор из результатов process_file.
    Добавляет флаг качества сигнала (quality_ok).
    """
    ptt = result.get('_ptt_arr', np.array([]))
    pwv = result.get('_pwv_arr', np.array([]))
    rpeaks = result.get('_rpeaks', np.array([]))

    def safe(fn, arr, default=np.nan):
        try:
            return float(fn(arr)) if len(arr) > 0 else default
        except Exception:
            return default

    rr_ms = np.diff(rpeaks) / FS * 1000 if len(rpeaks) > 1 else np.array([])
    rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)] if len(rr_ms) else rr_ms

    ptt_ms  = result.get('ptt_ms', np.nan)
    ptt_std = result.get('ptt_std_ms', np.nan)
    pwv_ms  = result.get('pwv_ms', np.nan)
    n_valid = result.get('n_valid', 0)
    ptt_cv  = ptt_std / ptt_ms if (ptt_ms and ptt_ms > 0) else np.nan

    # ── Флаг качества сигнала ─────────────────────────────────────────────
    # Сигнал считается валидным если:
    #   1. n_valid >= 15       — достаточная статистика
    #   2. PWV: 2.5–15 м/с    — физиологически допустимый диапазон
    #   3. CV_PTT < 0.60      — вариабельность < 60% медианы
    #                            (порог повышен с 0.45 до 0.60: у пациентов
    #                            с аритмией высокий CV — физиологическая
    #                            норма, а не признак плохого сигнала)
    quality_ok = (
        n_valid >= 15
        and (not np.isnan(pwv_ms)) and 2.5 < pwv_ms < 15.0
        and (not np.isnan(ptt_cv)) and ptt_cv < 0.60
    )

    feats = {
        'patient_id':    result.get('patient_id', ''),
        'signal_idx':    result.get('signal_idx', -1),
        'file':          result.get('file', ''),
        'quality_ok':    quality_ok,
        # Качество сигнала
        'duration_s':    result.get('duration_s', np.nan),
        'artifact_pct':  result.get('artifact_pct', np.nan),
        'n_rpeaks':      result.get('n_rpeaks', 0),
        'n_valid':       n_valid,
        'valid_ratio':   round(n_valid / max(result.get('n_rpeaks', 1), 1), 3),
        # PTT / PWV
        'ptt_median_ms': ptt_ms,
        'ptt_std_ms':    ptt_std,
        'ptt_cv':        round(ptt_cv, 3) if not np.isnan(ptt_cv) else np.nan,
        'ptt_p10_ms':    safe(lambda a: np.percentile(a, 10) * 1000, ptt),
        'ptt_p90_ms':    safe(lambda a: np.percentile(a, 90) * 1000, ptt),
        'ptt_iqr_ms':    safe(lambda a: (np.percentile(a,75) -
                                         np.percentile(a,25)) * 1000, ptt),
        'pwv_median_ms': pwv_ms,
        'pwv_std':       result.get('pwv_std', np.nan),
        'pwv_p10':       safe(lambda a: np.percentile(a, 10), pwv),
        'pwv_p90':       safe(lambda a: np.percentile(a, 90), pwv),
        # HRV
        'mean_hr':       result.get('mean_hr', np.nan),
        'sdnn_ms':       result.get('sdnn_ms', np.nan),
        'rmssd_ms':      result.get('rmssd_ms', np.nan),
        'pnn50':         result.get('pnn50', np.nan),
        'rr_cv':         round(float(np.std(rr_ms)/np.mean(rr_ms))
                               if len(rr_ms)>1 else np.nan, 4),
    }
    return feats


def patient_summary(df_signals: pd.DataFrame) -> dict:
    """
    Агрегированные показатели по пациенту из таблицы сигналов.
    Взвешиваем по числу валидных пар (n_valid) — больше данных → больший вес.
    """
    if df_signals.empty:
        return {}

    weights = df_signals['n_valid'].fillna(0).values
    w_sum   = weights.sum()

    def wavg(col):
        vals = df_signals[col].dropna()
        if len(vals) == 0 or w_sum == 0:
            return np.nan
        w = weights[df_signals[col].notna()]
        return float(np.average(vals, weights=w))

    return dict(
        patient_id      = df_signals['patient_id'].iloc[0],
        n_signals       = len(df_signals),
        n_signals_ok    = int((df_signals['n_valid'] > 0).sum()),
        # PWV / PTT
        pwv_mean        = round(wavg('pwv_median_ms'), 2),
        pwv_std_signals = round(float(df_signals['pwv_median_ms'].std()), 2),
        pwv_min         = round(float(df_signals['pwv_median_ms'].min()), 2),
        pwv_max         = round(float(df_signals['pwv_median_ms'].max()), 2),
        ptt_mean_ms     = round(wavg('ptt_median_ms'), 1),
        ptt_std_ms      = round(float(df_signals['ptt_median_ms'].std()), 1),
        # HRV
        mean_hr         = round(wavg('mean_hr'), 1),
        sdnn_ms         = round(wavg('sdnn_ms'), 1),
        rmssd_ms        = round(wavg('rmssd_ms'), 1),
        pnn50           = round(wavg('pnn50'), 1),
        # Качество
        avg_artifact_pct = round(float(df_signals['artifact_pct'].mean()), 1),
        avg_valid_ratio  = round(float(df_signals['valid_ratio'].mean()), 3),
    )


def print_patient_report(summary: dict):
    pid = summary.get('patient_id', '?')
    print(f"\n{'─'*55}")
    print(f"  ПАЦИЕНТ {pid}")
    print(f"{'─'*55}")
    print(f"  Записей: {summary['n_signals_ok']} / {summary['n_signals']} обработано")
    print(f"  PWV: {summary['pwv_mean']:.2f} ± {summary['pwv_std_signals']:.2f} м/с"
          f"  [{summary['pwv_min']:.2f} – {summary['pwv_max']:.2f}]")
    print(f"  PTT: {summary['ptt_mean_ms']:.1f} ± {summary['ptt_std_ms']:.1f} мс")
    print(f"  HR:  {summary['mean_hr']:.1f} уд/мин")
    print(f"  HRV SDNN:  {summary['sdnn_ms']:.1f} мс"
          f"   RMSSD: {summary['rmssd_ms']:.1f} мс"
          f"   pNN50: {summary['pnn50']:.1f}%")
    print(f"  Среднее артефактов: {summary['avg_artifact_pct']:.1f}%")


# ─── ОСНОВНАЯ ФУНКЦИЯ ────────────────────────────────────────────────────────

def run_batch(config_path='config.json',
              data_dir='.',
              plot=False,
              verbose=True):
    """
    Обрабатывает всех пациентов из config.json.
    Возвращает (df_signals, df_patients, df_features).
    """
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f'[!] config.json не найден: {cfg_path}')
        return None, None, None

    with open(cfg_path, encoding='utf-8') as f:
        config = json.load(f)

    global_cfg = config.get('global', {})
    output_dir = Path(global_cfg.get('output_dir', 'processed_data'))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path  = Path(data_dir)

    all_signals  = []
    all_features = []
    patient_summaries = []

    for patient in config.get('patients', []):
        pid      = patient['id']
        dist     = patient.get('distance_m', 0.5)
        tkeo     = patient.get('tkeo_factor', 0.6)
        ppg_cut  = patient.get('ppg_cutoff_hz', 10.0)
        files    = patient.get('files', [])

        print(f"\n{'#'*60}")
        print(f"  ПАЦИЕНТ {pid}   dist={dist} м   tkeo={tkeo}")
        print(f"  Файлов: {len(files)}")
        print(f"{'#'*60}")

        patient_feats = []

        for idx, fname in enumerate(files, start=1):
            fpath = resolve_path(data_path, fname)
            if fpath is None:
                print(f'  [!] Файл не найден: {fname}')
                continue

            result = process_file(
                filepath     = str(fpath),
                distance_m   = dist,
                tkeo_factor  = tkeo,
                ppg_cutoff_hz = min(ppg_cut, 10.0),   # не выше 10 Гц для PPG
                plot         = plot,
                verbose      = verbose,
            )

            if result is None:
                continue

            result['patient_id'] = pid
            result['signal_idx'] = idx

            # feature-вектор
            feats = extract_features(result)
            patient_feats.append(feats)
            all_features.append(feats)

            # скалярные поля для итоговой таблицы (без сырых массивов)
            scalar = {k: v for k, v in result.items() if not k.startswith('_')}
            scalar['patient_id'] = pid
            scalar['signal_idx'] = idx
            all_signals.append(scalar)

        if patient_feats:
            df_p  = pd.DataFrame(patient_feats)
            summ  = patient_summary(df_p)
            patient_summaries.append(summ)
            print_patient_report(summ)

    # ── Сборка итоговых таблиц ────────────────────────────────────────────
    df_signals  = pd.DataFrame(all_signals)
    df_features = pd.DataFrame(all_features)
    df_patients = pd.DataFrame(patient_summaries)

    if df_signals.empty:
        print('\n[!] Нет обработанных сигналов.')
        return df_signals, df_patients, df_features

    # ── Таблица только качественных сигналов ─────────────────────────────
    df_good = df_features[df_features['quality_ok'] == True].copy()
    n_total = len(df_features)
    n_good  = len(df_good)
    print(f'\n{"="*60}')
    print(f'Качество сигналов: {n_good}/{n_total} прошли фильтр '
          f'(PWV<15 м/с, CV_PTT<0.60, n_valid≥15)')
    if not df_good.empty:
        print('\nВалидные сигналы по пациентам:')
        print(df_good.groupby('patient_id')[
            ['ptt_median_ms','ptt_cv','pwv_median_ms','mean_hr',
             'sdnn_ms','rmssd_ms']
        ].agg(['mean','std']).round(2).to_string())

    # ── Сохранение ───────────────────────────────────────────────────────
    sig_path  = output_dir / 'results_per_signal.csv'
    pat_path  = output_dir / 'results_per_patient.csv'
    feat_path = output_dir / 'features_matrix.csv'
    good_path = output_dir / 'features_quality_ok.csv'

    df_signals.to_csv(sig_path,   index=False, encoding='utf-8-sig')
    df_patients.to_csv(pat_path,  index=False, encoding='utf-8-sig')
    df_features.to_csv(feat_path, index=False, encoding='utf-8-sig')
    df_good.to_csv(good_path,     index=False, encoding='utf-8-sig')

    print(f'\nФайлы сохранены в "{output_dir}":')
    print(f'  {sig_path.name}          — все результаты по сигналам')
    print(f'  {pat_path.name}        — сводка по пациентам')
    print(f'  {feat_path.name}        — полная feature-матрица')
    print(f'  {good_path.name}  — только качественные сигналы (для ML)')

    print(f'\nСводная таблица пациентов:')
    print(df_patients.to_string(index=False))

    return df_signals, df_patients, df_features, df_good


# ─── ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ ПАЦИЕНТОВ ────────────────────────────────────────

def plot_patient_comparison(df_features: pd.DataFrame,
                            patient_meta: dict = None):
    """
    Интерактивный дашборд сравнения пациентов.
    patient_meta: {patient_id: {'age': int, 'sex': str, 'notes': str}}
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print('[!] plotly не установлен: pip install plotly')
        return

    df = df_features.copy()
    df_ok = df[df['quality_ok'] == True]

    pid_all   = sorted(df['patient_id'].unique())
    pid_ok    = sorted(df_ok['patient_id'].unique())
    colors    = {'P01': '#e74c3c', 'P02': '#2980b9', 'P03': '#27ae60'}
    def col(pid): return colors.get(pid, '#888')

    def meta_label(pid):
        if patient_meta and pid in patient_meta:
            m = patient_meta[pid]
            return f"{pid} ({m.get('age','?')}л, {m.get('notes','')})"
        return pid

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'PWV по сигналам (качественные)',
            'Распределение PTT (качественные)',
            'ЧСС по пациентам (все сигналы)',
            'HRV: SDNN vs RMSSD',
            'pNN50 — маркер аритмии',
            'Качество записей',
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    # ── 1. PWV boxplot по пациентам (quality_ok) ─────────────────────────
    for pid in pid_ok:
        d = df_ok[df_ok['patient_id'] == pid]['pwv_median_ms'].dropna()
        fig.add_trace(go.Box(
            y=d, name=meta_label(pid),
            marker_color=col(pid), boxmean=True,
            legendgroup=pid, showlegend=True,
        ), row=1, col=1)

    # ── 2. PTT distribution (violin) ─────────────────────────────────────
    for pid in pid_ok:
        d = df_ok[df_ok['patient_id'] == pid]['ptt_median_ms'].dropna()
        fig.add_trace(go.Violin(
            y=d, name=meta_label(pid),
            marker_color=col(pid), box_visible=True,
            legendgroup=pid, showlegend=False,
        ), row=1, col=2)

    # ── 3. ЧСС scatter (все сигналы, по индексу) ─────────────────────────
    for pid in pid_all:
        d = df[df['patient_id'] == pid]
        fig.add_trace(go.Scatter(
            x=d['signal_idx'], y=d['mean_hr'],
            mode='markers+lines', name=meta_label(pid),
            marker=dict(color=col(pid), size=8),
            legendgroup=pid, showlegend=False,
        ), row=2, col=1)

    # ── 4. HRV scatter SDNN vs RMSSD ─────────────────────────────────────
    for pid in pid_all:
        d = df[df['patient_id'] == pid]
        fig.add_trace(go.Scatter(
            x=d['sdnn_ms'], y=d['rmssd_ms'],
            mode='markers', name=meta_label(pid),
            marker=dict(color=col(pid), size=9, opacity=0.8),
            text=d['signal_idx'].apply(lambda i: f'{pid} сигн.{i}'),
            hovertemplate='%{text}<br>SDNN=%{x:.1f}<br>RMSSD=%{y:.1f}',
            legendgroup=pid, showlegend=False,
        ), row=2, col=2)
    # Нормативные зоны
    fig.add_hrect(y0=0, y1=20, fillcolor='rgba(231,76,60,0.08)',
                  line_width=0, row=2, col=2)
    fig.add_hrect(y0=20, y1=50, fillcolor='rgba(243,156,18,0.08)',
                  line_width=0, row=2, col=2)

    # ── 5. pNN50 bar (среднее по пациенту) ───────────────────────────────
    for pid in pid_all:
        d = df[df['patient_id'] == pid]['pnn50'].dropna()
        fig.add_trace(go.Bar(
            x=[meta_label(pid)], y=[d.mean()],
            name=meta_label(pid), marker_color=col(pid),
            error_y=dict(type='data', array=[d.std()]),
            legendgroup=pid, showlegend=False,
        ), row=3, col=1)

    # ── 6. Качество: stacked bar ok/fail ─────────────────────────────────
    for pid in pid_all:
        d  = df[df['patient_id'] == pid]
        ok = d['quality_ok'].sum()
        total = len(d)
        fig.add_trace(go.Bar(
            x=[meta_label(pid)], y=[ok],
            name='Качественных', marker_color=col(pid),
            legendgroup=pid, showlegend=False,
        ), row=3, col=2)
        fig.add_trace(go.Bar(
            x=[meta_label(pid)], y=[total - ok],
            name='Отброшено', marker_color='rgba(180,180,180,0.6)',
            legendgroup=pid+'_bad', showlegend=False,
        ), row=3, col=2)

    fig.update_layout(
        height=950, title='Сравнительный анализ пациентов',
        barmode='stack',
        legend=dict(orientation='h', y=-0.05),
    )
    fig.update_yaxes(title_text='PWV, м/с',   row=1, col=1)
    fig.update_yaxes(title_text='PTT, мс',    row=1, col=2)
    fig.update_yaxes(title_text='ЧСС, уд/мин', row=2, col=1)
    fig.update_xaxes(title_text='Номер сигнала', row=2, col=1)
    fig.update_yaxes(title_text='RMSSD, мс',  row=2, col=2)
    fig.update_xaxes(title_text='SDNN, мс',   row=2, col=2)
    fig.update_yaxes(title_text='pNN50, %',   row=3, col=1)
    fig.update_yaxes(title_text='Сигналов',   row=3, col=2)
    fig.show()
    return fig


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Батч-обработка СРПВ')
    parser.add_argument('--config',   default='config.json',
                        help='Путь к config.json')
    parser.add_argument('--data_dir', default='.',
                        help='Папка с CSV-файлами')
    parser.add_argument('--plot',     action='store_true',
                        help='Показывать графики для каждого файла')
    args = parser.parse_args()

    run_batch(config_path=args.config,
              data_dir=args.data_dir,
              plot=args.plot)


# ─── COLAB — запуск напрямую из ячейки ───────────────────────────────────────
# DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/WORK/DATA'
# CONFIG   = f'{DATA_DIR}/config.json'
#
# from batch_pwv import run_batch
# df_signals, df_patients, df_features, df_good = run_batch(
#     config_path = CONFIG,
#     data_dir    = DATA_DIR,
#     plot        = False,
# )
# display(df_good)   # только качественные сигналы
