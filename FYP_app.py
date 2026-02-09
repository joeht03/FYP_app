# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 17:47:07 2026

@author: Joseph Hilton-Tapp
"""

import streamlit as st
import numpy as np
import tempfile
import os

from FYP_processing import (
    load_txt_file,
    load_data_1d,
    load_data_2,
    load_data_all,

    # 2D view
    render_2d_view,
    find_common_second_pulse_window_2d,

    # 1D analysis
    find_common_second_pulse_window_1d,
    
    auto_find_second_pulse_window,
    
    find_common_second_pulse_window,

    # Regional occlusion
    compare_regions,

    # Regional refractive index
    compare_regions_refractive_index,

    # Evolution plots + helpers
    raw_waveform_evolution_full,
    raw_waveform_evolution_windowed,
    waveform_evolution,
    waveform_evolution_offset,
    clean_filename
)

# -------------------------------------------------------------------------
# APP HEADER
# -------------------------------------------------------------------------

st.markdown("## 🔬 THz Imaging & Spectroscopy Analysis Suite")
st.write("Upload your THz `.txt` or `.npz` files to analyse 1D, 2D, and multi-region data.")

# -------------------------------------------------------------------------
# GLOBAL STYLING PANEL (SIDEBAR)
# -------------------------------------------------------------------------

with st.sidebar:
    file_type = st.radio("File Type", [".npz", ".txt"])

    st.header("📂 Section")
    section = st.radio(
        "Choose Section",
        [
            "📈 1D Point Analysis",
            "📷 2D Image View",
            "📊 Regional Comparison — Occlusion Curve",
            "🔬 Regional Comparison — Refractive Index"
        ]
    )

    st.markdown("---")
    st.header("🎨 Plot Styling")

    title_size = st.slider("Title Font Size", 8, 40, 20)
    axis_label_size = st.slider("Axis Label Font Size", 8, 30, 14)
    tick_size = st.slider("Tick Label Font Size", 6, 24, 12)
    legend_size = st.slider("Legend Font Size", 6, 24, 12)

    legend_position = st.selectbox(
        "Legend Position",
        ["best", "upper right", "upper left", "lower right", "lower left",
         "center right", "center left", "upper center", "lower center", "center"]
    )

    line_width = st.slider("Line Width", 1.0, 6.0, 2.0)
    alpha_value = st.slider("Line Transparency", 0.1, 1.0, 0.8)

    grid_on = st.checkbox("Show Grid", value=True)

    st.markdown("### Region Colours")
    forearm_color = st.color_picker("Forearm Colour", "#1f77b4")
    cheek_color = st.color_picker("Cheek Colour", "#d62728")
    shoulder_color = st.color_picker("Shoulder Colour", "#2ca02c")
    foot_color = st.color_picker("Foot Colour", "#9467bd")

# -------------------------------------------------------------------------
# 2D IMAGE VIEW
# -------------------------------------------------------------------------

if section == "📷 2D Image View":

    if file_type != ".npz":
        st.warning("2D imaging requires `.npz` files.")
        st.stop()

    sample_file = st.file_uploader("Sample .npz", type="npz")
    time_file = st.file_uploader("Time .npz (optional)", type="npz")
    ref_file = st.file_uploader("Reference .npz (optional)", type="npz")

    if not sample_file:
        st.info("📤 Please upload a sample `.npz` file.")
        st.stop()

    try:
        if sample_file and time_file and ref_file:
            coords, waveforms, _, time = load_data_2(sample_file, time_file)
            _, _, reference, _ = load_data_all(ref_file)

        elif sample_file and time_file:
            coords, waveforms, reference, time = load_data_2(sample_file, time_file)

        elif sample_file and ref_file:
            coords, waveforms, _, time = load_data_all(sample_file)
            _, _, reference, _ = load_data_all(ref_file)

        else:
            coords, waveforms, reference, time = load_data_all(sample_file)

    except KeyError as e:
        missing_key = str(e).strip("'")
        st.warning(f"❗ Missing key `{missing_key}`. Please upload a separate time `.npz` file.")
        st.stop()

    window, tA, tB = find_common_second_pulse_window_2d(
        waveforms, coords, time, R=10
    )
    st.info(f"Window selected: {time[tA]:.2f} ps → {time[tB]:.2f} ps")

    st.markdown("---")
    st.header("⚙️ Settings")

    view = st.selectbox(
        "Choose View",
        ["Max Amplitude", "Phase", "Peak-to-Peak", "Refractive Index", "Time Delay"]
    )
    interp = st.selectbox("Interpolation Method", ["linear", "nearest"])

    apply_crop = st.checkbox("🟢 Apply Circular Crop?")
    if apply_crop:
        from FYP_processing import round_cutter
        R = st.slider("Crop Radius", 1.0, 12.0, 11.0, step=0.5)
        wf_array = np.stack(waveforms).T
        wf_cut, coords_cut = round_cutter(wf_array, coords, R, center=(0, -1.5))
        waveforms = [wf_cut[:, i] for i in range(wf_cut.shape[1])]
        coords = coords_cut

    render_2d_view(view, interp, coords, waveforms, reference, time, window)

# -------------------------------------------------------------------------
# 1D POINT ANALYSIS
# -------------------------------------------------------------------------

elif section == "📈 1D Point Analysis":

    from FYP_processing import render_1d_analysis  # imported here to avoid circulars

    if file_type == ".npz":
        sample_file = st.file_uploader("Sample .npz", type="npz")
        if not sample_file:
            st.info("📤 Please upload a sample `.npz` file.")
            st.stop()

        time, waveforms, reference = load_data_1d(sample_file)
        window = find_common_second_pulse_window_1d(waveforms, time)

        st.markdown("---")
        st.header("⚙️ Settings")

        view = st.selectbox(
            "Choose View",
            ["FFT of windowed waveform", "M", "Refractive Index", "Occlusion Curve"]
        )

        render_1d_analysis(view, waveforms, reference, time, window)

    elif file_type == ".txt":
        sample_file = st.file_uploader("Sample + Time (.txt)", type="txt")
        ref_file = st.file_uploader("Reference (.txt)", type="txt")

        if not sample_file or not ref_file:
            st.info("📤 Please upload both sample and reference `.txt` files.")
            st.stop()

        time, waveforms, reference = load_txt_file(sample_file, ref_file)
        window = find_common_second_pulse_window_1d(waveforms, time)

        st.markdown("---")
        st.header("⚙️ Settings")

        view = st.selectbox(
            "Choose View",
            ["FFT of windowed waveform", "M", "Refractive Index"]
        )

        render_1d_analysis(view, waveforms, reference, time, window)

# -------------------------------------------------------------------------
# REGIONAL OCCLUSION CURVE COMPARISON
# -------------------------------------------------------------------------

elif section == "📊 Regional Comparison — Occlusion Curve":

    st.markdown("### Upload your repeat measurement `.npz` files for each region")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        [
            "Occlusion Curve",
            "Sample Impulse Response Evolution",
            "Base Waveform Evolution",
            "Windowed Waveform Evolution",
            "Sample Impulse Response Offset"
        ]
    )

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        vf_files = st.file_uploader("Volar Forearm", type="npz", accept_multiple_files=True)
    with col2:
        cheek_files = st.file_uploader("Cheek", type="npz", accept_multiple_files=True)
    with col3:
        shoulder_files = st.file_uploader("Shoulder", type="npz", accept_multiple_files=True)
    with col4:
        foot_files = st.file_uploader("Foot", type="npz", accept_multiple_files=True)

    if st.button("Process Regional Analysis"):

        def pair_files(uploaded_files):
            pairs = []
            for f in uploaded_files or []:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
                tmp.write(f.read())
                tmp.close()
                pairs.append((f, tmp.name))
            return pairs

        vf_pairs = pair_files(vf_files)
        cheek_pairs = pair_files(cheek_files)
        shoulder_pairs = pair_files(shoulder_files)
        foot_pairs = pair_files(foot_files)

        all_regions = {
            "Forearm": vf_pairs,
            "Cheek": cheek_pairs,
            "Shoulder": shoulder_pairs,
            "Foot": foot_pairs
        }

        # 1. Occlusion Curve
        if analysis_type == "Occlusion Curve":
            compare_regions(
                forearm_files=[p[1] for p in vf_pairs],
                cheek_files=[p[1] for p in cheek_pairs],
                shoulder_files=[p[1] for p in shoulder_pairs],
                foot_files=[p[1] for p in foot_pairs],
                sf=12.5,
                duration=60,
                show_confidence=True,
                title_size=title_size,
                axis_label_size=axis_label_size,
                tick_size=tick_size,
                legend_size=legend_size,
                legend_loc=legend_position,
                line_width=line_width,
                alpha_value=alpha_value,
                grid_on=grid_on,
                forearm_color=forearm_color,
                cheek_color=cheek_color,
                shoulder_color=shoulder_color,
                foot_color=foot_color
            )

        # 2. Evolution Plots (one plot per file)
        else:
            for region_name, file_pairs in all_regions.items():
                if not file_pairs:
                    continue
            
                st.markdown(f"## 📍 {region_name}")
            
                for uploaded_file, file_path in file_pairs:
                    clean_name = clean_filename(uploaded_file.name)
                    st.markdown(f"### File: `{clean_name}`")
            
                    data = np.load(file_path)
                    waveforms = data["wfs"]
                    time = data["time"]
                    reference = data["ref"]
            
                    window = find_common_second_pulse_window_1d(waveforms, time)
            
                    if analysis_type == "Base Waveform Evolution":
                        fig = raw_waveform_evolution_full(
                            waveforms,
                            12.5,
                            patient=clean_name,
                            title_size=title_size,
                            axis_label_size=axis_label_size,
                            tick_size=tick_size,
                            line_width=line_width,
                            alpha_value=alpha_value,
                            grid_on=grid_on
                        )
                        st.pyplot(fig)
            
                    elif analysis_type == "Windowed Waveform Evolution":
                        fig = raw_waveform_evolution_windowed(
                            waveforms,
                            window,
                            12.5,
                            patient=clean_name,
                            title_size=title_size,
                            axis_label_size=axis_label_size,
                            tick_size=tick_size,
                            line_width=line_width,
                            alpha_value=alpha_value,
                            grid_on=grid_on
                        )
                        st.pyplot(fig)
            
                    elif analysis_type == "Sample Impulse Response Evolution":
                        fig = waveform_evolution(
                            waveforms,
                            window,
                            12.5,
                            time,
                            0,
                            60,
                            reference,
                            patient=clean_name,
                            title_size=title_size,
                            axis_label_size=axis_label_size,
                            tick_size=tick_size,
                            line_width=line_width,
                            alpha_value=alpha_value,
                            grid_on=grid_on
                        )
                        st.pyplot(fig)
            
                    elif analysis_type == "Sample Impulse Response Offset":
                        fig = waveform_evolution_offset(
                            waveforms,
                            window,
                            12.5,
                            time,
                            0,
                            60,
                            reference,
                            patient=clean_name,
                            title_size=title_size,
                            axis_label_size=axis_label_size,
                            tick_size=tick_size,
                            line_width=line_width,
                            alpha_value=alpha_value,
                            grid_on=grid_on
                        )
                        st.pyplot(fig)

# -------------------------------------------------------------------------
# REGIONAL REFRACTIVE INDEX COMPARISON
# -------------------------------------------------------------------------

elif section == "🔬 Regional Comparison — Refractive Index":

    st.markdown("### Upload your repeat measurement `.npz` files for each region")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        vf_files = st.file_uploader("Volar Forearm", type="npz", accept_multiple_files=True)
    with col2:
        cheek_files = st.file_uploader("Cheek", type="npz", accept_multiple_files=True)
    with col3:
        shoulder_files = st.file_uploader("Shoulder", type="npz", accept_multiple_files=True)
    with col4:
        foot_files = st.file_uploader("Foot", type="npz", accept_multiple_files=True)

    if st.button("Process Regional Refractive Index"):

        def save_uploaded(files):
            paths = []
            for f in files or []:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
                tmp.write(f.read())
                tmp.close()
                paths.append(tmp.name)
            return paths

        vf_paths = save_uploaded(vf_files)
        cheek_paths = save_uploaded(cheek_files)
        shoulder_paths = save_uploaded(shoulder_files)
        foot_paths = save_uploaded(foot_files)

        compare_regions_refractive_index(
            forearm_files=vf_paths,
            cheek_files=cheek_paths,
            shoulder_files=shoulder_paths,
            foot_files=foot_paths,
            fmin=0.2,
            fmax=2.0,
            show_confidence=True,
            title_size=title_size,
            axis_label_size=axis_label_size,
            tick_size=tick_size,
            legend_size=legend_size,
            legend_loc=legend_position,
            line_width=line_width,
            alpha_value=alpha_value,
            grid_on=grid_on,
            forearm_color=forearm_color,
            cheek_color=cheek_color,
            shoulder_color=shoulder_color,
            foot_color=foot_color
        )
