# -*- coding: utf-8 -*-
"""
Калибровка СРПВ → Артериальное давление (АД).

Физиологическая основа:
    У одного пациента в ограниченном диапазоне давлений связь
    СРПВ и АД линейна: BP = a × PWV + b.
    Коэффициенты a и b определяются из парных измерений
    манжетного АД и СРПВ, сделанных в одно время.

Ограничения:
    - Калибровка индивидуальна: коэффициенты нельзя переносить
      между пациентами.
    - Для надёжной регрессии нужно ≥ 5 пар измерений.
    - Работает только в диапазоне давлений, покрытом калибровкой.
    - Используется среднее АД (MAP) как наиболее физиологически
      значимая величина: MAP = DBP + (SBP − DBP) / 3.

Запуск в Colab:
    from pwv_bp_calibration import run_calibration, estimate_bp
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════════════
# 1. РАСЧЁТ СРПВ ИЗ ФАЙЛОВ (опционально)
# ═════════════════════════════════════════════════════════════════════════════

def extract_pwv_from_files(file_pwv_list, data_dir,
                            distance_m, tkeo_factor):
    """
    Автоматически извлекает СРПВ из списка файлов.

    file_pwv_list — список имён файлов (без пути)
    Возвращает list[float] — медианные СРПВ для каждого файла.
    """
    from pwv_single import process_file

    data_path = Path(data_dir)
    pwv_list  = []

    for fname in file_pwv_list:
        fpath = None
        for c in [data_path / fname, data_path / (fname + '.csv')]:
            if c.exists():
                fpath = c; break

        if fpath is None:
            print(f'  [!] Файл не найден: {fname} → пропускаем')
            pwv_list.append(np.nan)
            continue

        res = process_file(str(fpath), distance_m=distance_m,
                           tkeo_factor=tkeo_factor, plot=False, verbose=False)
        if res and not np.isnan(res['pwv_ms']):
            pwv = res['pwv_ms']
            print(f'  {fname}: СРПВ = {pwv:.2f} м/с')
        else:
            pwv = np.nan
            print(f'  {fname}: не удалось извлечь СРПВ')
        pwv_list.append(pwv)

    return pwv_list


# ═════════════════════════════════════════════════════════════════════════════
# 2. ОСНОВНАЯ ФУНКЦИЯ КАЛИБРОВКИ
# ═════════════════════════════════════════════════════════════════════════════

def run_calibration(pwv_values, sbp_values, dbp_values,
                    patient_id='', show_plot=True):
    """
    Строит калибровочную кривую СРПВ → АД по парным измерениям.

    Параметры:
        pwv_values — list[float]: СРПВ, м/с
        sbp_values — list[float]: систолическое АД, мм рт. ст.
        dbp_values — list[float]: диастолическое АД, мм рт. ст.
        patient_id — строка-идентификатор пациента
        show_plot  — показывать ли график

    Возвращает dict с коэффициентами регрессии и метриками.
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
    except ImportError:
        print('[!] pip install scikit-learn')
        return None
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        show_plot = False

    # ── Подготовка данных ─────────────────────────────────────────────────
    pwv = np.array(pwv_values, dtype=float)
    sbp = np.array(sbp_values, dtype=float)
    dbp = np.array(dbp_values, dtype=float)

    # Среднее АД: MAP = DBP + (SBP − DBP) / 3
    map_ = dbp + (sbp - dbp) / 3.0

    # Убираем строки с NaN
    valid = ~(np.isnan(pwv) | np.isnan(sbp) | np.isnan(dbp))
    pwv_v, sbp_v, dbp_v, map_v = pwv[valid], sbp[valid], dbp[valid], map_[valid]

    n = valid.sum()
    print(f'\n{"="*55}')
    print(f'Калибровка СРПВ → АД  [{patient_id}]')
    print(f'Точек измерений: {n}')

    if n < 3:
        print('[!] Недостаточно точек для регрессии (нужно ≥ 3).')
        return None

    # ── Линейная регрессия для каждого вида АД ───────────────────────────
    def fit(x, y, label):
        reg = LinearRegression().fit(x.reshape(-1,1), y)
        a, b = float(reg.coef_[0]), float(reg.intercept_)
        pred = reg.predict(x.reshape(-1,1))
        r2   = r2_score(y, pred)
        mae  = np.mean(np.abs(y - pred))
        print(f'  {label}: BP = {a:.2f}×СРПВ + {b:.2f}  '
              f'R²={r2:.3f}  MAE={mae:.1f} мм рт.ст.')
        return dict(a=round(a,4), b=round(b,4), r2=round(r2,3),
                    mae=round(mae,2), label=label)

    res_sbp = fit(pwv_v, sbp_v, 'СД (SBP)')
    res_dbp = fit(pwv_v, dbp_v, 'ДД (DBP)')
    res_map = fit(pwv_v, map_v, 'MAP')

    # ── Корреляция Пирсона ────────────────────────────────────────────────
    r_sbp = float(np.corrcoef(pwv_v, sbp_v)[0, 1])
    r_dbp = float(np.corrcoef(pwv_v, dbp_v)[0, 1])
    r_map = float(np.corrcoef(pwv_v, map_v)[0, 1])
    print(f'\n  Корреляция Пирсона:')
    print(f'    СРПВ ↔ СД:  r = {r_sbp:.3f}')
    print(f'    СРПВ ↔ ДД:  r = {r_dbp:.3f}')
    print(f'    СРПВ ↔ MAP: r = {r_map:.3f}')

    # ── Визуализация ──────────────────────────────────────────────────────
    if show_plot:
        fig = make_subplots(rows=1, cols=3,
            subplot_titles=('СРПВ vs СД (SBP)',
                            'СРПВ vs ДД (DBP)',
                            'СРПВ vs MAP'),
            horizontal_spacing=0.10)

        pwv_line = np.linspace(pwv_v.min() - 0.3, pwv_v.max() + 0.3, 100)

        for col, (y_vals, res, r_val, color) in enumerate([
            (sbp_v, res_sbp, r_sbp, '#e74c3c'),
            (dbp_v, res_dbp, r_dbp, '#2980b9'),
            (map_v, res_map, r_map, '#27ae60'),
        ], start=1):
            # Точки измерений
            fig.add_trace(go.Scatter(
                x=pwv_v, y=y_vals, mode='markers',
                marker=dict(color=color, size=10, line=dict(width=1,color='white')),
                name=res['label'], showlegend=(col==1),
            ), row=1, col=col)

            # Линия регрессии
            y_line = res['a'] * pwv_line + res['b']
            fig.add_trace(go.Scatter(
                x=pwv_line, y=y_line, mode='lines',
                line=dict(color=color, dash='dash', width=2),
                name=f'Регрессия R²={res["r2"]:.2f}',
                showlegend=(col==1),
            ), row=1, col=col)

            # Аннотация
            fig.add_annotation(
                x=pwv_v.mean(), y=y_vals.max(),
                text=f'r={r_val:.3f}, R²={res["r2"]:.3f}',
                showarrow=False, font=dict(size=11, color=color),
                row=1, col=col,
            )

        fig.update_xaxes(title_text='СРПВ, м/с')
        fig.update_yaxes(title_text='АД, мм рт.ст.', col=1)
        fig.update_layout(
            height=420,
            title=f'Калибровочная кривая СРПВ → АД  [{patient_id}]  '
                  f'n={n} измерений',
        )
        fig.show()

    return dict(
        patient_id = patient_id,
        n          = int(n),
        sbp        = res_sbp,
        dbp        = res_dbp,
        map        = res_map,
        r_sbp      = round(r_sbp, 3),
        r_dbp      = round(r_dbp, 3),
        r_map      = round(r_map, 3),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3. ОЦЕНКА АД ПО НОВОМУ СРПВ
# ═════════════════════════════════════════════════════════════════════════════

def estimate_bp(calibration, pwv_new):
    """
    Оценивает СД, ДД и MAP по новому значению СРПВ.

    calibration — результат run_calibration()
    pwv_new     — float или list[float]: новые значения СРПВ, м/с

    Возвращает DataFrame с оценками АД.
    """
    if calibration is None:
        print('[!] Калибровка не проведена.')
        return None

    pwv_arr = np.atleast_1d(np.array(pwv_new, dtype=float))
    rows = []

    for pwv in pwv_arr:
        sbp_est = calibration['sbp']['a'] * pwv + calibration['sbp']['b']
        dbp_est = calibration['dbp']['a'] * pwv + calibration['dbp']['b']
        map_est = calibration['map']['a'] * pwv + calibration['map']['b']
        # Проверяем физиологические границы
        sbp_ok  = 80 <= sbp_est <= 200
        dbp_ok  = 50 <= dbp_est <= 130
        rows.append(dict(
            pwv_ms     = round(float(pwv), 2),
            sbp_est    = round(sbp_est, 1),
            dbp_est    = round(dbp_est, 1),
            map_est    = round(map_est, 1),
            bp_str     = f'{sbp_est:.0f}/{dbp_est:.0f}',
            in_range   = sbp_ok and dbp_ok,
        ))

    df = pd.DataFrame(rows)

    print(f'\nОценка АД по СРПВ [{calibration["patient_id"]}]:')
    print(df[['pwv_ms','bp_str','map_est','in_range']].to_string(index=False))
    if not df['in_range'].all():
        print('[!] Часть значений вышла за физиологические границы — '
              'экстраполяция за пределы калибровки ненадёжна.')
    return df


def estimate_bp_from_files(calibration, file_list, data_dir,
                            distance_m, tkeo_factor, show_plot=True):
    """
    Полный пайплайн: новые файлы → СРПВ → оценка АД.

    calibration — результат run_calibration()
    file_list   — list[str]: имена новых CSV-файлов
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        show_plot = False

    print(f'\nИзвлечение СРПВ из {len(file_list)} файлов...')
    pwv_list = extract_pwv_from_files(file_list, data_dir,
                                       distance_m, tkeo_factor)

    df = estimate_bp(calibration, pwv_list)
    if df is None:
        return None

    df['file'] = file_list

    if show_plot and not df.empty:
        valid = df[df['in_range']]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=valid['pwv_ms'], y=valid['sbp_est'],
            mode='markers+lines', name='СД (оценка)',
            marker=dict(color='#e74c3c', size=9),
        ))
        fig.add_trace(go.Scatter(
            x=valid['pwv_ms'], y=valid['dbp_est'],
            mode='markers+lines', name='ДД (оценка)',
            marker=dict(color='#2980b9', size=9),
        ))
        fig.add_trace(go.Scatter(
            x=valid['pwv_ms'], y=valid['map_est'],
            mode='markers+lines', name='MAP (оценка)',
            marker=dict(color='#27ae60', size=9),
        ))
        fig.update_layout(
            title=f'Оценка АД по СРПВ [{calibration["patient_id"]}]',
            xaxis_title='СРПВ, м/с',
            yaxis_title='АД, мм рт. ст.',
            height=420,
        )
        fig.show()

    return df


# ═════════════════════════════════════════════════════════════════════════════
# 4. ШАБЛОН ДЛЯ COLAB
# ═════════════════════════════════════════════════════════════════════════════
#
# ── Шаг 1: вводим парные измерения (СРПВ + манжетное АД) ─────────────────
#
# Вариант А — СРПВ уже посчитана (из df_signals или вручную):
#
# pwv_values = [5.80, 5.46, 3.89, 4.97, 5.25, 3.90, 4.22, 7.32, 5.35, 3.20]
# sbp_values = [132,  128,  118,  125,  130,  115,  122,  138,  127,  112]
# dbp_values = [ 84,   80,   74,   78,   82,   72,   76,   88,   80,   70]
#
# from pwv_bp_calibration import run_calibration, estimate_bp
#
# calib = run_calibration(
#     pwv_values = pwv_values,
#     sbp_values = sbp_values,
#     dbp_values = dbp_values,
#     patient_id = 'P01',
#     show_plot  = True,
# )
#
# ── Шаг 2: оцениваем АД по новым СРПВ ────────────────────────────────────
#
# # Вручную:
# df_bp = estimate_bp(calib, pwv_new=[4.5, 5.0, 5.5, 6.0])
#
# # Или автоматически из новых файлов:
# from pwv_bp_calibration import estimate_bp_from_files
#
# df_bp = estimate_bp_from_files(
#     calibration = calib,
#     file_list   = ['new_signal_1.csv', 'new_signal_2.csv'],
#     data_dir    = DATA_DIR,
#     distance_m  = 0.57,
#     tkeo_factor = 0.8,
#     show_plot   = True,
# )
# display(df_bp)
#
# ── Вариант Б — автоматически извлечь СРПВ из файлов P01 ─────────────────
#
# from pwv_bp_calibration import extract_pwv_from_files, run_calibration
#
# files_p01 = [
#     'data4ch_0_1.csv', 'data4ch_0_6.csv', 'data4ch_0_7.csv',
#     'data4ch_0_9.csv', 'data4ch_0_10.csv',  # ... все файлы с замерами АД
# ]
# sbp_p01 = [132, 128, 130, 127, 118]   # введи реальные значения
# dbp_p01 = [ 84,  80,  82,  80,  74]
#
# pwv_p01 = extract_pwv_from_files(files_p01, DATA_DIR,
#                                   distance_m=0.57, tkeo_factor=0.8)
# calib = run_calibration(pwv_p01, sbp_p01, dbp_p01,
#                          patient_id='P01', show_plot=True)
