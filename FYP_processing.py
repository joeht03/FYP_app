# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 17:44:49 2026

@author: Joseph Hilton-Tapp
"""

# -*- coding: utf-8 -*-
"""
THz processing and plotting core.

Updated with:
- Global Matplotlib defaults (sans-serif, white background)
- Shared styling helper for axes
- Styled evolution plots (returning fig, no plt.show)
- Styled regional occlusion comparison
- Filename cleaner for participant/area/force/scan
"""

import json
import ast
import os

import numpy as np
from numpy import load
from numpy.lib.scimath import sqrt

import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events

from scipy import interpolate
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d, make_interp_spline

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import streamlit as st

# -------------------------------------------------------------------------
# GLOBAL MATPLOTLIB DEFAULTS
# -------------------------------------------------------------------------

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


# -------------------------------------------------------------------------
# BASIC HELPERS
# -------------------------------------------------------------------------


def select_waveform_index_from_time(waveforms, default="2:22"):
    waveforms_per_minute = 239
    waveforms_per_second = waveforms_per_minute / 60
    total_waveforms = len(waveforms)
    total_seconds = int(total_waveforms / waveforms_per_second)

    time_labels = [f"{s // 60}:{s % 60:02d}" for s in range(total_seconds + 1)]
    label_to_index = {label: int(s * waveforms_per_second)
                      for s, label in enumerate(time_labels)}

    selected_label = st.select_slider(
        "Select Time (mm:ss)",
        options=time_labels,
        value=default
    )

    i_th_wf = label_to_index[selected_label]
    st.write(f"📈 Using waveform index: **{i_th_wf}**")
    return i_th_wf


def render_1d_analysis(view, waveforms, reference, time, window):
    t = time[window]
    index = select_waveform_index_from_time(waveforms)

    if view == "FFT of windowed waveform":
        freqs, X = calculate_fft(waveforms[index][window], t)
        X_mag = np.abs(X)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs, y=X_mag, mode='lines', name='FFT',
            hovertemplate='Frequency: %{x:.3f} THz<br>'
                          'Magnitude: %{y:.3f}<extra></extra>'
        ))
        fig.update_layout(
            title="1D THz Signal",
            xaxis_title="Frequency (THz)",
            yaxis_title="FFT Magnitude",
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    elif view == "M":
        wf_w = waveforms[index][window]
        ref_w = reference[window]
        M, freqs = calculate_sample_response(wf_w, t, ref_w)
        mag = np.abs(M)

        hover_x = st.slider("Select frequency (THz)", 0.2, 2.0, 0.5,
                            step=0.01)
        y_min, y_max = st.slider("Y-axis range (Magnitude)",
                                 0.0, 5.0, (0.0, 2.0))
        y_interp = np.interp(hover_x, freqs, mag)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs, y=mag,
                                 mode='lines+markers',
                                 name='Sample Response'))
        fig.add_trace(go.Scatter(
            x=[hover_x], y=[y_interp],
            mode='markers+text',
            marker=dict(color='red', size=10),
            text=[f"{y_interp:.3f}"],
            textposition='top center',
            name='Interpolated'
        ))
        fig.update_layout(
            title="M (Sample / Reference)",
            xaxis_title="Frequency (THz)",
            yaxis_title="Magnitude",
            xaxis=dict(range=[0.2, 2]),
            yaxis=dict(range=[y_min, y_max]),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Interpolated magnitude at {hover_x:.3f} THz: "
                 f"**{y_interp:.4f}**")

    elif view == "Refractive Index":
        wf_w = waveforms[index][window]
        ref_w = reference[window]
        M, freqs = calculate_sample_response(wf_w, t, ref_w)
        M_mag = np.abs(M)
        RI = calculate_n_sample(M_mag)

        hover_x = st.slider("Select frequency (THz)", 0.2, 2.0, 0.5,
                            step=0.01)
        y_min, y_max = st.slider("Y-axis range (RI)",
                                 0.0, 5.0, (0.0, 2.5))
        y_interp = np.interp(hover_x, freqs, RI)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs, y=RI,
                                 mode='lines+markers',
                                 name='Refractive Index'))
        fig.add_trace(go.Scatter(
            x=[hover_x], y=[y_interp],
            mode='markers+text',
            marker=dict(color='red', size=10),
            text=[f"{y_interp:.3f}"],
            textposition='top center',
            name='Interpolated'
        ))
        fig.update_layout(
            title="Refractive Index",
            xaxis_title="Frequency (THz)",
            yaxis_title="RI",
            xaxis=dict(range=[0.2, 2]),
            yaxis=dict(range=[y_min, y_max]),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Interpolated RI at {hover_x:.3f} THz: "
                 f"**{y_interp:.4f}**")

    elif view == "Occlusion Curve":
        sf = st.slider("Sampling Freq (defaulted to 12.5)",
                       0.0, 30.0, 12.5, step=0.5)
        start, stop = st.slider("Time range plotted on X-axis (s)",
                                0.0, 90.0, (0.0, 60.0))

        B = int(round(start * sf))
        A = int(round(stop * sf))
        waveforms_half = waveforms[B:A, :]
        peak_time = np.linspace(start, stop, A - B)

        window = find_common_second_pulse_window_1d(waveforms, time)
        time_windowed = time[window]
        reference_windowed = reference[window]

        wf_half_windowed = waveforms_half[:, window]

        pt, pm = occlusion_curve(wf_half_windowed,
                                 peak_time,
                                 time_windowed,
                                 reference_windowed)
        pm = clean_occlusion_curve(pm)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pt, y=pm,
                                 mode='lines+markers',
                                 name='Occlusion Curve'))
        fig.update_layout(
            title="Occlusion Curve",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)




def try_parse_array(s, idx):
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(s)
    except Exception as e:
        print(f"[Error] Failed to parse array at index {idx}:\n{s[:100]}...\nReason: {e}")
        return None


def load_txt_file(sample_file, reference_file):
    content = sample_file.read().decode("utf-8")
    array_strings = content.replace("]\n[", "]SPLIT[").split("SPLIT")

    arrays = []
    for idx, s in enumerate(array_strings):
        parsed = try_parse_array(s, idx)
        if parsed is not None:
            arrays.append(parsed)

    if not arrays or len(arrays) < 2:
        raise ValueError("Not enough valid arrays parsed from sample file.")

    time = np.array(arrays[0])
    waveforms = [np.array(w) for w in arrays[1:]]

    reference = np.loadtxt(reference_file, usecols=1)

    return time, waveforms, reference


def load_data_all(sample_file):
    data = np.load(sample_file)
    return data['coords'], data['wfs'], data['ref_start'], data['time']


def load_data_2(sample_file, time_file):
    data = np.load(sample_file)
    time_data = np.load(time_file)
    return data['coords'], data['wfs'], data['ref_start'], time_data['time']


def load_data_1d(sample_file):
    data = np.load(sample_file)
    return data['time'], data['wfs'], data['ref']


def clean_filename(name: str) -> str:
    """
    Extract participant, area, force, scan number from filenames like:
    P001_Forearm_5N_1_2024-11-12_14-33-22.npz -> P001_Forearm_5N_1
    """
    base = name.replace(".npz", "")
    parts = base.split("_")
    if len(parts) >= 4:
        return "_".join(parts[:4])
    return base


# -------------------------------------------------------------------------
# STYLING HELPER
# -------------------------------------------------------------------------

def style_axes(
    ax,
    title=None,
    xlabel=None,
    ylabel=None,
    title_size=20,
    axis_label_size=14,
    tick_size=12,
    grid_on=True
):
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=axis_label_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=axis_label_size)
    ax.tick_params(axis='both', labelsize=tick_size)
    if grid_on:
        ax.grid(True, alpha=0.3)


###############################################################################
# FFT / FREQUENCY HELPERS
###############################################################################

def calculate_fft(waveform, t):
    """
    t in ps -> freqs in THz (rfftfreq with ps spacing).
    """
    N = len(waveform)
    dt = (t[-1] - t[0]) / (N - 1)
    freqs = np.fft.rfftfreq(N, d=dt)
    X = np.fft.rfft(waveform)
    return freqs, X


def fft_and_freq(windowed_waveform, t):
    """
    Backwards-compatible wrapper around calculate_fft for existing code.
    Returns freqs and |X|.
    """
    freqs, X = calculate_fft(windowed_waveform, t)
    return X, freqs, np.abs(X)


###############################################################################
# 2D INTERPOLATION + MAP HELPERS
###############################################################################

def interpolate_grid(coords, values, interp_method, res=100):
    x, y = coords[:, 0], coords[:, 1]
    xi = np.linspace(np.min(x), np.max(x), res)
    yi = np.linspace(np.min(y), np.max(y), res)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_values = interpolate.griddata(coords, values, (grid_x, grid_y),
                                       method=interp_method)
    return grid_x, grid_y, grid_values, xi, yi


def calculate_phase(waveforms, window, index):
    phases = []
    for wf in waveforms:
        X = np.fft.rfft(wf[window])
        phases.append(np.angle(X[index]))
    return phases


def calculate_max_amplitude(waveforms, window):
    return [np.max(np.abs(np.fft.rfft(wf[window]))) for wf in waveforms]


def unwrap_phase(grid_phase):
    mask = np.isnan(grid_phase)
    gv = np.copy(grid_phase)
    gv[mask] = 0
    gv_unwrapped = np.unwrap(np.unwrap(gv, axis=1), axis=0)
    gv_unwrapped[mask] = np.nan
    return gv_unwrapped


def calculate_peak_to_peak(waveforms, window):
    return [np.max(wf[window]) - np.min(wf[window]) for wf in waveforms]


###############################################################################
# SAMPLE RESPONSE + REFRACTIVE INDEX CORE
###############################################################################

def calculate_sample_response(waveform, t, reference):
    """
    Amplitude-only sample response M = X/Y.
    """
    freqs, X = calculate_fft(waveform, t)
    Y = np.fft.rfft(reference)
    M = X / Y
    return M, freqs


def calculate_n_sample(M_meas, nq=2.12, na=1.00, theta_a_deg=9):
    """
    Amplitude-only Fresnel inversion for refractive index.
    """
    theta_a = np.deg2rad(theta_a_deg)
    theta_q = np.arcsin((na / nq) * np.sin(theta_a))
    cos_theta_q = np.cos(theta_q)
    sin_theta_q = np.sin(theta_q)
    cos_theta_a = np.cos(theta_a)

    n_s = []
    for M_val in M_meas:
        numerator = (nq**2 * cos_theta_q**2) * (1 - M_val) + \
                    (na * nq * cos_theta_a * cos_theta_q) * (1 + M_val)
        denominator = (nq * cos_theta_q) * (1 + M_val) + \
                      (na * cos_theta_a) * (1 - M_val)
        X = numerator / denominator
        n_s.append(np.real(sqrt(X**2 + (nq**2 * sin_theta_q**2))))

    return np.array(n_s)


###############################################################################
# 2D CLICK HANDLER
###############################################################################

def handle_waveform_click(fig, coords, waveforms, time):
    clicked_points = plotly_events(fig, click_event=True, hover_event=False)

    if clicked_points:
        x_click = clicked_points[0]['x']
        y_click = clicked_points[0]['y']

        distances = np.sqrt((coords[:, 0] - x_click) ** 2 +
                            (coords[:, 1] - y_click) ** 2)
        nearest_idx = np.argmin(distances)

        st.markdown("---")
        st.subheader("📈 Waveform at Selected Location")
        st.write(f"**Nearest coordinate:** "
                 f"({coords[nearest_idx][0]:.2f}, {coords[nearest_idx][1]:.2f})")

        wf = waveforms[nearest_idx]
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=time, y=wf, mode='lines',
                                      name='Waveform'))
        fig_wave.update_layout(
            title='Full Waveform at Selected Coordinate',
            xaxis_title='Time (ps)',
            yaxis_title='Amplitude',
            template='plotly_white'
        )
        st.plotly_chart(fig_wave, use_container_width=True)


###############################################################################
# 2D VIEW
###############################################################################

def render_2d_view(view, interp_method, coords, waveforms, reference, time, window):
    t = time[window]
    _, freqs, _ = fft_and_freq(waveforms[0][window], t)

    label = ""
    data_vals = []
    freq_target = None

    if view == "Max Amplitude":
        data_vals = calculate_max_amplitude(waveforms, window)
        label = "Max Amplitude"

    elif view == "Peak-to-Peak":
        data_vals = calculate_peak_to_peak(waveforms, window)
        label = "Peak-to-Peak"

    elif view == "Phase":
        freq_target = st.slider("Target frequency (THz)", 0.1, 2.0, 0.5)
        index = np.argmin(np.abs(freqs - freq_target))
        data_vals = calculate_phase(waveforms, window, index)
        label = f"Phase at {freq_target:.2f} THz"

    elif view == "Refractive Index":
        freq_target = st.slider("Target frequency (THz)", 0.1, 2.0, 0.5)
        index = np.argmin(np.abs(freqs - freq_target))

        nq, na, theta_a = 2.12, 1.00, 9
        data_vals = []
        ref_w = reference[window]

        for wf in waveforms:
            wf_w = wf[window]
            M, freqs_local = calculate_sample_response(wf_w, t, ref_w)
            M_mag = np.abs(M)
            n_full = calculate_n_sample(M_mag, nq=nq, na=na, theta_a_deg=theta_a)
            data_vals.append(n_full[index])

        label = f"Refractive Index at {freq_target:.2f} THz"

    elif view == "Time Delay":
        full_time = time
        st.markdown("⚠️ Early part (e.g. before –220 ps) may be redundant.")
        selected_time = st.slider("Select Time Delay (ps)",
                                  -250.0, -200.0, -217.0, step=0.1)
        time_idx = np.argmin(np.abs(full_time - selected_time))
        data_vals = [wf[time_idx] for wf in waveforms]
        label = f"Amplitude at {selected_time:.1f} ps"

    else:
        st.warning("Unsupported view type selected.")
        return

    gx, gy, gv, xi, yi = interpolate_grid(coords, data_vals, interp_method)

    if view == "Phase":
        gv = unwrap_phase(gv)

    fig = go.Figure(data=go.Heatmap(
        z=gv,
        x=xi,
        y=yi,
        colorscale='Viridis',
        colorbar=dict(title=label),
        hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=label,
        xaxis_title='X',
        yaxis_title='Y',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template='plotly_white'
    )

    average_value = st.checkbox("🟢 Average value area selection?")
    if average_value:
        st.markdown("### 📏 Select Region for Averaging")
        x0, x1 = st.slider("X Range",
                           float(np.min(xi)), float(np.max(xi)),
                           (float(np.min(xi)), float(np.max(xi))))
        y0, y1 = st.slider("Y Range",
                           float(np.min(yi)), float(np.max(yi)),
                           (float(np.min(yi)), float(np.max(yi))))

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=y0, y1=y1,
            line=dict(color="red", width=2),
            fillcolor="rgba(255,0,0,0)",
            layer="above"
        )

        grid_x, grid_y = np.meshgrid(xi, yi)
        mask = (grid_x >= x0) & (grid_x <= x1) & \
               (grid_y >= y0) & (grid_y <= y1)
        selected_vals = gv[mask]

        if selected_vals.size > 0:
            avg_val = np.nanmean(selected_vals)
            st.success(f"Average value in selected region: {avg_val:.4f}")
        else:
            st.warning("No data points in selected region.")

    st.caption("Click on the heatmap to display the waveform at chosen location.")
    handle_waveform_click(fig, coords, waveforms, time)


###############################################################################
# GEOMETRIC HELPERS
###############################################################################

def round_cutter(wf, coords, R, center=(0, 0)):
    xc, yc = center
    xy_ind = np.where((coords[:, 0] - xc)**2 +
                      (coords[:, 1] - yc)**2 < R**2)[0]
    XY_new = coords[xy_ind, :]
    wf_cut = wf[:, xy_ind]
    return wf_cut, XY_new


###############################################################################
# WINDOW FINDING – 1D / REGIONAL
###############################################################################

def auto_find_second_pulse_window_1d(wf, time, prominence=0.1, width_ps=2):
    envelope = np.abs(hilbert(wf))
    peaks, props = find_peaks(envelope,
                              prominence=prominence * np.max(envelope))

    if len(peaks) < 2:
        target_peak = peaks[0] if len(peaks) > 0 else np.argmax(envelope)
    else:
        prominences = props["prominences"]
        sorted_peaks = [x for _, x in
                        sorted(zip(prominences, peaks), reverse=True)]
        target_peak = sorted_peaks[1]

    dt = np.mean(np.diff(time))
    half_width_idx = int((width_ps / 2) / dt)
    start = max(0, target_peak - half_width_idx)
    end = min(len(wf), target_peak + half_width_idx)
    return slice(start, end)


def find_common_second_pulse_window_1d(waveforms, time,
                                       prominence=0.1, width_ps=2,
                                       pad_ps=0.5, buffer=25):
    slices = [auto_find_second_pulse_window_1d(wf, time,
                                               prominence, width_ps)
              for wf in waveforms]
    starts = np.array([s.start for s in slices])
    stops = np.array([s.stop for s in slices])

    start_idx = int(np.median(starts))
    stop_idx = int(np.median(stops))

    dt = np.mean(np.diff(time))
    pad_idx = int(pad_ps / dt)

    start_idx = max(0, start_idx - pad_idx)
    stop_idx = min(len(time), stop_idx + pad_idx)

    return slice(start_idx - buffer, stop_idx + buffer)


###############################################################################
# WINDOW FINDING – 2D VERSION
###############################################################################

def auto_find_second_pulse_window_2d(wf, t, threshold=0.1, min_width=100):
    N = len(wf)
    mid = N // 2
    wf2 = wf[mid:]

    envelope = np.abs(hilbert(wf2))
    env_norm = envelope / np.max(envelope)
    indices = np.where(env_norm > threshold)[0]

    if len(indices) == 0:
        return slice(mid, mid + 50)

    start = indices[0]
    end = indices[-1]

    if end - start < min_width:
        pad = (min_width - (end - start)) // 2
        start = max(0, start - pad)
        end = min(len(wf2), end + pad)

    return slice(mid + start, mid + end)


def find_common_second_pulse_window_2d(waveforms, coords, time,
                                       R=10, center=(0, 0),
                                       threshold=0.1, pad_ps=1):
    wf_stack = np.stack(waveforms).T
    xc, yc = center
    wf_cut, _ = round_cutter(wf_stack, coords, R, center=(xc, yc))
    waveforms_cut = [wf_cut[:, i] for i in range(wf_cut.shape[1])]

    pulse_windows = [auto_find_second_pulse_window_2d(wf, time,
                                                      threshold=threshold)
                     for wf in waveforms_cut]

    starts = [w.start for w in pulse_windows]
    stops = [w.stop for w in pulse_windows]

    start_idx = int(np.median(starts))
    stop_idx = int(np.median(stops))

    start_time = time[start_idx] - pad_ps
    stop_time = time[stop_idx] + pad_ps + 1

    start_common = int(np.argmin(np.abs(time - start_time)))
    stop_common = int(np.argmin(np.abs(time - stop_time)))

    return slice(start_common, stop_common), start_common, stop_common


###############################################################################
# OCCLUSION CURVE – IMPULSE RESPONSE VERSION
###############################################################################

def filtered_impulse_response(M, freqs, low_f=0.1, high_f=1, t=None):
    dg = np.exp(-(freqs / high_f) ** 2) - np.exp(-(freqs / low_f) ** 2)
    filtered_fft_sample = M * dg
    h = np.fft.irfft(filtered_fft_sample)
    peak_idx = np.argmax(np.abs(h))
    h = np.roll(h, len(h)//2 - peak_idx)
    return h


def impulse_response_peak(wf, t, reference, window_ps=2.0):
    M, freqs = calculate_sample_response(wf, t, reference)
    h = filtered_impulse_response(M, freqs, t=t)
    dt = np.mean(np.diff(t))
    half_w = int((window_ps / 2) / dt)
    center = np.argmax(np.abs(h))
    s = max(0, center - half_w)
    e = min(len(h), center + half_w)
    h_win = h[s:e]
    return np.max(h_win) - np.min(h_win)


def occlusion_curve(waveforms, peak_time, time, reference):
    peak_mag = []
    for wf in waveforms:
        amp = impulse_response_peak(wf, time, reference)
        peak_mag.append(amp)
    return np.array(peak_time), np.array(peak_mag)


def clean_occlusion_curve(pm, median_size=15, outlier_thresh=4, eps=1e-6):
    pm = np.asarray(pm)
    pm_med = median_filter(pm, size=median_size)
    diff = np.abs(pm - pm_med)
    mad = median_filter(diff, size=median_size)
    mad = np.maximum(mad, eps)
    mask = diff > (outlier_thresh * mad)
    pm_clean = pm.copy()
    pm_clean[mask] = pm_med[mask]
    return pm_clean


def align_occlusion_time(t, occl):
    occl_s = median_filter(occl, size=9)
    deriv = np.gradient(occl_s)
    drop_idx = np.argmax(np.abs(deriv))
    return t - t[drop_idx]


###############################################################################
# RAW / PROCESSED EVOLUTION PLOTS (STREAMLIT-FRIENDLY)
###############################################################################

def raw_waveform_evolution_full(
    waveforms,
    sf,
    patient="Unknown",
    title_size=20,
    axis_label_size=14,
    tick_size=12,
    line_width=2.0,
    alpha_value=0.8,
    grid_on=True
):
    snapshot_times = np.arange(0, 20.01, 1)
    snapshot_indices = [int(round(t * sf)) for t in snapshot_times]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=20)

    for ts, idx in zip(snapshot_times, snapshot_indices):
        if idx >= waveforms.shape[0]:
            continue
        wf = waveforms[idx]
        ax.plot(wf, color=cmap(norm(ts)), alpha=alpha_value, linewidth=line_width)

    style_axes(
        ax,
        title=f"RAW Full Waveform Evolution (0–20s): {patient}",
        xlabel="Sample Index",
        ylabel="Amplitude",
        title_size=title_size,
        axis_label_size=axis_label_size,
        tick_size=tick_size,
        grid_on=grid_on
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Time (s)")

    fig.tight_layout()
    return fig


def raw_waveform_evolution_windowed(
    waveforms,
    window,
    sf,
    patient="Unknown",
    title_size=20,
    axis_label_size=14,
    tick_size=12,
    line_width=2.0,
    alpha_value=0.8,
    grid_on=True
):
    snapshot_times = np.arange(0, 20.01, 1)
    snapshot_indices = [int(round(t * sf)) for t in snapshot_times]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.cividis
    norm = plt.Normalize(vmin=0, vmax=20)

    for ts, idx in zip(snapshot_times, snapshot_indices):
        if idx >= waveforms.shape[0]:
            continue
        wf = waveforms[idx][window]
        ax.plot(wf, color=cmap(norm(ts)), alpha=alpha_value, linewidth=line_width)

    style_axes(
        ax,
        title=f"RAW Windowed Waveform Evolution (0–20s): {patient}",
        xlabel="Window Sample Index",
        ylabel="Amplitude",
        title_size=title_size,
        axis_label_size=axis_label_size,
        tick_size=tick_size,
        grid_on=grid_on
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Time (s)")

    fig.tight_layout()
    return fig


def waveform_evolution(
    waveforms,
    window,
    scanning_freq,
    t,
    start_time,
    end_time,
    reference,
    patient="Unknown",
    title_size=20,
    axis_label_size=14,
    tick_size=12,
    line_width=2.0,
    alpha_value=0.6,
    grid_on=True
):
    skip_seconds = 5
    start_idx = int(round(skip_seconds * scanning_freq))
    waveforms_skipped = waveforms[start_idx:, :]
    sf2 = int(round(scanning_freq * 2))
    waveforms_ds = waveforms_skipped[::sf2, :]

    time_windowed = t[window]
    reference_windowed = reference[window]

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])
    colors = cmap(np.linspace(0, 1, len(waveforms_ds)))

    for i, wf in enumerate(waveforms_ds):
        wf_windowed = wf[window]
        M, freqs = calculate_sample_response(wf_windowed, time_windowed, reference_windowed)
        h = filtered_impulse_response(M, freqs, t=time_windowed)
        ax.plot(time_windowed, h, color=colors[i], alpha=alpha_value, linewidth=line_width)

    style_axes(
        ax,
        title=f'Evolution of Sample Impulse Response (5s–{int(end_time)}s): {patient}',
        xlabel="Time (ps)",
        ylabel="Impulse Response Amplitude",
        title_size=title_size,
        axis_label_size=axis_label_size,
        tick_size=tick_size,
        grid_on=grid_on
    )

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=start_time + 5, vmax=end_time)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Scan Time (s)")
    cbar.set_ticks([start_time + 5, end_time])
    cbar.set_ticklabels([f"{int(start_time + 5)}s", f"{int(end_time)}s"])

    fig.tight_layout()
    return fig


def waveform_evolution_offset(
    waveforms,
    window,
    scanning_freq,
    t,
    start_time,
    end_time,
    reference,
    patient="Unknown",
    n_snapshots=12,
    pulse_width_visual_width=20,
    title_size=20,
    axis_label_size=14,
    tick_size=12,
    line_width=1.5,
    alpha_value=0.85,
    grid_on=True
):
    snapshot_times = np.linspace(start_time + 5, end_time, n_snapshots, endpoint=True)
    snapshot_indices = [int(round(ts * scanning_freq)) for ts in snapshot_times]

    time_windowed = t[window]
    reference_windowed = reference[window]
    time_centered = time_windowed - np.mean(time_windowed)
    ps_range = time_centered[-1] - time_centered[0]
    scaling_factor = pulse_width_visual_width / ps_range

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=start_time, vmax=end_time)

    peak_x_coords = []
    peak_y_coords = []
    peak_x_coords_b = []
    peak_y_coords_b = []

    for ts, idx in zip(snapshot_times, snapshot_indices):
        if idx >= waveforms.shape[0]:
            continue

        wf = waveforms[idx][window]
        M, freqs = calculate_sample_response(wf, time_windowed, reference_windowed)
        h = filtered_impulse_response(M, freqs, t=time_windowed)
        h = h * 100

        x_values = ts + (time_centered * scaling_factor)
        ax.plot(
            x_values,
            h,
            color=cmap(norm(ts)),
            lw=line_width,
            alpha=alpha_value
        )

        peak_idx = np.argmax(h)
        peak_x_coords.append(x_values[peak_idx])
        peak_y_coords.append(h[peak_idx])

        peak_idx_b = np.argmin(h)
        peak_x_coords_b.append(x_values[peak_idx_b])
        peak_y_coords_b.append(h[peak_idx_b])

    if len(peak_x_coords) > 2:
        spline = make_interp_spline(np.array(peak_x_coords), np.array(peak_y_coords), k=3)
        x_smooth = np.linspace(min(peak_x_coords), max(peak_x_coords), 300)
        y_smooth = spline(x_smooth)

        spline_b = make_interp_spline(np.array(peak_x_coords_b), np.array(peak_y_coords_b), k=3)
        x_smooth_b = np.linspace(min(peak_x_coords_b), max(peak_x_coords_b), 300)
        y_smooth_b = spline_b(x_smooth_b)

        ax.plot(x_smooth, y_smooth, color='red', linewidth=2)
        ax.plot(x_smooth_b, y_smooth_b, color='red', linewidth=2)

    style_axes(
        ax,
        title=f"Waveform Evolution (5s–60s): {patient}",
        xlabel="Scan Time (s) [Pulses Magnified]",
        ylabel="Amplitude [*10^-2]",
        title_size=title_size,
        axis_label_size=axis_label_size,
        tick_size=tick_size,
        grid_on=grid_on
    )

    ax.set_xlim(
        (start_time + 5) - (pulse_width_visual_width / 2),
        end_time + (pulse_width_visual_width / 2)
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Scan Time (s)")

    fig.tight_layout()
    return fig


###############################################################################
# REGIONAL OCCLUSION ENGINE
###############################################################################

def process_files(
    file_list,
    sf=12.5,
    start_seconds=0,
    end_seconds=60,
    show_evolution=False,
    evolution_indices=None
):
    results = []
    file_details = []

    for file_path in file_list:
        data = load(file_path)
        waveforms = data['wfs']
        time = data['time']

        if 'ref' in data:
            reference = data['ref']
        else:
            print(f"⚠️ WARNING: File {file_path} has no 'ref'. Skipping.")
            continue

        patient_name = "_".join(os.path.splitext(
            os.path.basename(file_path))[0].split('_')[:4])

        window = find_common_second_pulse_window_1d(waveforms, time)

        B = int(round(start_seconds * sf))
        A = int(round(end_seconds * sf))

        waveforms_half = waveforms[B:A, window]
        peak_time = np.linspace(start_seconds, end_seconds, A - B)

        time_windowed = time[window]
        reference_windowed = reference[window]

        pt, pm = occlusion_curve(
            waveforms_half,
            peak_time,
            time_windowed,
            reference_windowed
        )

        pm = clean_occlusion_curve(pm)

        results.append({
            'patient': patient_name,
            'time': pt,
            'occlusion': pm
        })

        file_details.append({
            'waveforms': waveforms,
            'window': window,
            'reference': reference,
            'time': time,
            'patient': patient_name
        })

    aligned_occlusions, aligned_times = [], []
    for r in results:
        occl = r['occlusion']
        t = r['time']
        t_aligned = align_occlusion_time(t, occl)
        aligned_occlusions.append(occl)
        aligned_times.append(t_aligned)

    if len(aligned_times) == 0:
        return {
            'common_time': np.array([]),
            'interpolated_curves': np.array([]),
            'mean_curve': np.array([]),
            'results': results
        }

    t_min = max(t[0] for t in aligned_times)
    t_max = min(t[-1] for t in aligned_times)
    common_time = np.linspace(t_min, t_max, 1000)

    interpolated_curves = []
    for t_aligned, occl in zip(aligned_times, aligned_occlusions):
        f = interp1d(t_aligned, occl, kind='linear',
                     bounds_error=False, fill_value=np.nan)
        interpolated_curves.append(f(common_time))

    interpolated_curves = np.array(interpolated_curves)
    mean_curve = np.nanmean(interpolated_curves, axis=0)

    if show_evolution and evolution_indices is not None:
        for idx in evolution_indices:
            if 0 <= idx < len(file_details):
                fd = file_details[idx]

                raw_waveform_evolution_full(
                    fd['waveforms'],
                    sf,
                    patient=fd['patient']
                )

                raw_waveform_evolution_windowed(
                    fd['waveforms'],
                    fd['window'],
                    sf,
                    patient=fd['patient']
                )

                waveform_evolution(
                    fd['waveforms'],
                    fd['window'],
                    sf,
                    fd['time'],
                    start_seconds,
                    end_seconds,
                    fd['reference'],
                    patient=fd['patient']
                )

                waveform_evolution_offset(
                    fd['waveforms'],
                    fd['window'],
                    sf,
                    fd['time'],
                    start_seconds,
                    end_seconds,
                    fd['reference'],
                    patient=fd['patient']
                )
            else:
                print(f"⚠️ evolution index {idx} is out of range for provided files.")

    return {
        'common_time': common_time,
        'interpolated_curves': interpolated_curves,
        'mean_curve': mean_curve,
        'results': results
    }


def compare_regions_dynamic(
    pools,
    sf=12.5,
    duration=60,
    show_confidence=True,
    title_size=20,
    axis_label_size=14,
    tick_size=12,
    legend_size=12,
    legend_loc="best",
    line_width=2.0,
    alpha_value=0.4,
    grid_on=True,
    return_data=False
):
    processed = {}
    colors = {}

    # ------------------------------------------------------------
    # Load and process all pools
    # ------------------------------------------------------------
    for pool in pools:
        name = pool["name"]
        files = pool["paths"]
        color = pool["color"]
        colors[name] = color

        if files and len(files) > 0:
            processed[name] = process_files(
                files,
                sf=sf,
                start_seconds=0,
                end_seconds=duration,
                show_evolution=False,
                evolution_indices=None
            )
        else:
            processed[name] = None

    # Find reference region with valid time axis
    reference_region = next(
        (processed[r] for r in processed
         if processed[r] is not None and processed[r]["common_time"].size > 0),
        None
    )

    if reference_region is None:
        st.error("No valid data provided for any pool.")
        return

    common_time = reference_region["common_time"]

    # ------------------------------------------------------------
    # FIGURE 1 — All curves + mean + CI
    # ------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    for region_name, data in processed.items():
        if data is None or data["common_time"].size == 0:
            continue

        curves = data["interpolated_curves"]
        mean_curve = data["mean_curve"]
        std_curve = np.nanstd(curves, axis=0)

        base_color = colors[region_name]

        # Plot all curves
        for occl in curves:
            ax1.plot(common_time, occl, color=base_color,
                     alpha=alpha_value, linewidth=line_width)

        # Plot mean
        ax1.plot(common_time, mean_curve, color=base_color,
                 lw=line_width + 1, label=f"{region_name} Mean")

        # Confidence shading
        if show_confidence:
            lower = mean_curve - 1.96 * std_curve
            upper = mean_curve + 1.96 * std_curve
            ax1.fill_between(common_time, lower, upper,
                             color=base_color, alpha=0.15)

    style_axes(
        ax1,
        title="Occlusion Curves: All Pools (All Curves + Means)",
        xlabel="Time since occlusion onset (s)",
        ylabel="SIR Peak-to-Peak Magnitude",
        title_size=title_size,
        axis_label_size=axis_label_size,
        tick_size=tick_size,
        grid_on=grid_on
    )
    ax1.legend(fontsize=legend_size, loc=legend_loc)
    st.pyplot(fig1)

    # ------------------------------------------------------------
    # FIGURE 2 — Means only + CI
    # ------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    for region_name, data in processed.items():
        if data is None or data["common_time"].size == 0:
            continue

        curves = data["interpolated_curves"]
        mean_curve = data["mean_curve"]
        std_curve = np.nanstd(curves, axis=0)

        base_color = colors[region_name]

        ax2.plot(common_time, mean_curve, color=base_color,
                 lw=line_width + 1, label=f"{region_name} Mean")

        if show_confidence:
            lower = mean_curve - 1.96 * std_curve
            upper = mean_curve + 1.96 * std_curve
            ax2.fill_between(common_time, lower, upper,
                             color=base_color, alpha=0.15)

    style_axes(
        ax2,
        title="Occlusion Curves: Pool Means Only",
        xlabel="Time since occlusion onset (s)",
        ylabel="SIR Peak-to-Peak Magnitude",
        title_size=title_size,
        axis_label_size=axis_label_size,
        tick_size=tick_size,
        grid_on=grid_on
    )
    ax2.legend(fontsize=legend_size, loc=legend_loc)
    st.pyplot(fig2)

    # ------------------------------------------------------------
    # Return data for numerical analysis
    # ------------------------------------------------------------
    if return_data:
        return {
            "fig1": fig1,
            "fig2": fig2,
            "common_time": common_time,
            "regions": {
                region_name: {
                    "mean": data["mean_curve"],
                    "std": np.nanstd(data["interpolated_curves"], axis=0)
                }
                for region_name, data in processed.items()
                if data is not None
            }
        }


###############################################################################
# REGIONAL REFRACTIVE INDEX COMPARISON (SIMPLE VERSION)
###############################################################################

def auto_find_second_pulse_window(wf, time, prominence=0.1, width_ps=2):
    envelope = np.abs(hilbert(wf))
    peaks, props = find_peaks(envelope, prominence=prominence * np.max(envelope))

    if len(peaks) < 2:
        target_peak = peaks[0] if len(peaks) > 0 else np.argmax(envelope)
    else:
        prominences = props["prominences"]
        sorted_peaks = [x for _, x in sorted(zip(prominences, peaks), reverse=True)]
        target_peak = sorted_peaks[1]

    dt = np.mean(np.diff(time))
    half_width_idx = int((width_ps / 2) / dt)
    start = max(0, target_peak - half_width_idx)
    end = min(len(wf), target_peak + half_width_idx)
    return slice(start, end)

def find_common_second_pulse_window(waveforms, time, prominence=0.1, width_ps=2, pad_ps=0.5, buffer=25):
    slices = [auto_find_second_pulse_window(wf, time, prominence, width_ps) for wf in waveforms]
    starts = np.array([s.start for s in slices])
    stops = np.array([s.stop for s in slices])

    start_idx = int(np.median(starts))
    stop_idx = int(np.median(stops))

    dt = np.mean(np.diff(time))
    pad_idx = int(pad_ps / dt)

    start_idx = max(0, start_idx - pad_idx)
    stop_idx = min(len(time), stop_idx + pad_idx)

    return slice(start_idx-buffer, stop_idx+buffer)

def compare_regions_refractive_index_dynamic(
    pools,
    fmin=0.2,
    fmax=2.0,
    show_confidence=True,
    title_size=20,
    axis_label_size=14,
    tick_size=12,
    legend_size=12,
    legend_loc="best",
    line_width=2.0,
    alpha_value=0.4,
    grid_on=True,
    analysis_time=20.0,
    return_data=False
):
    sf = 12.5
    region_freqs = {}
    region_curves = {}

    # ------------------------------------------------------------
    # Compute n(f) for each pool
    # ------------------------------------------------------------
    for pool in pools:
        region_name = pool["name"]
        color = pool["color"]
        files = pool["paths"]

        if not files:
            continue

        n_list = []
        freq_list = []

        for fp in files:
            data = load(fp)
            wfs = data["wfs"]
            time = data["time"]
            ref = data["ref"]

            idx = int(round(analysis_time * sf))
            idx = min(idx, wfs.shape[0] - 1)
            wf = wfs[idx]

            window = find_common_second_pulse_window(wfs, time)
            t_w = time[window]
            wf_w = wf[window]
            ref_w = ref[window]

            wf_half = wf_w[:len(wf_w)//2]
            ref_half = ref_w[:len(ref_w)//2]
            shift = np.argmax(wf_half) - np.argmax(ref_half)
            ref_aligned = np.roll(ref_w, shift)

            freqs, X = calculate_fft(wf_w, t_w)
            Y = np.fft.rfft(ref_aligned, n=len(wf_w))

            M = X / Y
            M_mag = np.abs(M)

            n_full = calculate_n_sample(M_mag)

            freq_list.append(freqs)
            n_list.append(n_full)

        max_len_idx = np.argmax([len(f) for f in freq_list])
        freqs_master = freq_list[max_len_idx]

        n_interp_list = []
        for freqs_i, n_i in zip(freq_list, n_list):
            n_interp = np.interp(freqs_master, freqs_i, n_i)
            n_interp_list.append(n_interp)

        region_freqs[region_name] = (freqs_master, color)
        region_curves[region_name] = np.vstack(n_interp_list)

    # ------------------------------------------------------------
    # Frequency mask
    # ------------------------------------------------------------
    any_region = next(iter(region_freqs.keys()))
    freqs_any, _ = region_freqs[any_region]
    mask = (freqs_any >= fmin) & (freqs_any <= fmax)
    freqs_plot = freqs_any[mask]

    # ------------------------------------------------------------
    # Compute global y-limits
    # ------------------------------------------------------------
    global_min = np.inf
    global_max = -np.inf

    for region_name, curves_full in region_curves.items():
        curves = curves_full[:, mask]
        mean_n = np.nanmean(curves, axis=0)
        std_n = np.nanstd(curves, axis=0)

        lower = np.maximum(mean_n - 1.96 * std_n, 0)
        upper = mean_n + 1.96 * std_n

        global_min = min(global_min, np.min(lower))
        global_max = max(global_max, np.max(upper))

    global_min -= 0.02
    global_max += 0.02

    # ------------------------------------------------------------
    # FIGURE 1 — All curves + mean + CI
    # ------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    for region_name, curves_full in region_curves.items():
        freqs_region, color = region_freqs[region_name]
        curves = curves_full[:, mask]

        mean_n = np.nanmean(curves, axis=0)
        std_n = np.nanstd(curves, axis=0)

        for c in curves:
            ax1.plot(freqs_plot, c, color=color,
                     alpha=alpha_value, linewidth=line_width)

        ax1.plot(freqs_plot, mean_n, color=color,
                 lw=line_width + 1, label=f"{region_name} Mean")

        if show_confidence:
            lower = np.maximum(mean_n - 1.96 * std_n, 0)
            upper = mean_n + 1.96 * std_n
            ax1.fill_between(freqs_plot, lower, upper,
                             color=color, alpha=0.15)

    ax1.set_ylim(global_min, global_max)

    style_axes(
        ax1,
        title=f"Refractive Index vs Frequency by Pool (t={analysis_time:.1f}s)",
        xlabel="Frequency (THz)",
        ylabel="Refractive Index n(f)",
        title_size=title_size,
        axis_label_size=axis_label_size,
        tick_size=tick_size,
        grid_on=grid_on
    )
    ax1.legend(fontsize=legend_size, loc=legend_loc)
    st.pyplot(fig1)

    # ------------------------------------------------------------
    # FIGURE 2 — Means only + CI
    # ------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    for region_name, curves_full in region_curves.items():
        freqs_region, color = region_freqs[region_name]
        curves = curves_full[:, mask]

        mean_n = np.nanmean(curves, axis=0)
        std_n = np.nanstd(curves, axis=0)

        ax2.plot(freqs_plot, mean_n, color=color,
                 lw=line_width + 1, label=f"{region_name} Mean")

        if show_confidence:
            lower = np.maximum(mean_n - 1.96 * std_n, 0)
            upper = mean_n + 1.96 * std_n
            ax2.fill_between(freqs_plot, lower, upper,
                             color=color, alpha=0.15)

    ax2.set_ylim(global_min, global_max)

    style_axes(
        ax2,
        title=f"Refractive Index Spectra: Pool Means Only (t={analysis_time:.1f}s)",
        xlabel="Frequency (THz)",
        ylabel="Refractive Index n(f)",
        title_size=title_size,
        axis_label_size=axis_label_size,
        tick_size=tick_size,
        grid_on=grid_on
    )
    ax2.legend(fontsize=legend_size, loc=legend_loc)
    st.pyplot(fig2)

    # ------------------------------------------------------------
    # Return data for numerical analysis
    # ------------------------------------------------------------
    if return_data:
        return {
            "fig1": fig1,
            "fig2": fig2,
            "freqs": freqs_plot,
            "regions": {
                region_name: {
                    "mean": np.nanmean(curves_full[:, mask], axis=0),
                    "std": np.nanstd(curves_full[:, mask], axis=0)
                }
                for region_name, curves_full in region_curves.items()
            }
        }