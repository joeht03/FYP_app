import streamlit as st
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt

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

    # Dynamic regional occlusion
    compare_regions_dynamic,

    # Dynamic regional refractive index
    compare_regions_refractive_index_dynamic,

    # Evolution plots + helpers
    raw_waveform_evolution_full,
    raw_waveform_evolution_windowed,
    waveform_evolution,
    waveform_evolution_offset,
    clean_filename
)

# -------------------------------------------------------------------------
# POOL MANAGEMENT
# -------------------------------------------------------------------------

def init_pools():
    if "pools" not in st.session_state:
        st.session_state.pools = []

def add_pool():
    init_pools()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    next_color = color_cycle[len(st.session_state.pools) % len(color_cycle)]
    st.session_state.pools.append({
        "name": f"Pool {len(st.session_state.pools)+1}",
        "color": next_color,
        "files": None
    })

def render_pool_editor(section_key_prefix="pool"):
    init_pools()

    if st.button("➕ Add Pool", key=f"{section_key_prefix}_add_pool"):
        add_pool()

    for i, pool in enumerate(st.session_state.pools):
        st.markdown(f"#### Pool {i+1}")

        pool["name"] = st.text_input(
            f"Name for Pool {i+1}",
            pool["name"],
            key=f"{section_key_prefix}_name_{i}"
        )

        pool["color"] = st.color_picker(
            f"Colour for Pool {i+1}",
            pool["color"],
            key=f"{section_key_prefix}_color_{i}"
        )

        uploaded = st.file_uploader(
            f"Upload `.npz` files for {pool['name']}",
            type="npz",
            accept_multiple_files=True,
            key=f"pool_files_{i}"
        )

        pool["files"] = uploaded

        if st.button(f"🗑️ Remove Pool {i+1}", key=f"{section_key_prefix}_remove_{i}"):
            st.session_state.pools.pop(i)
            st.rerun()

# -------------------------------------------------------------------------
# APP HEADER
# -------------------------------------------------------------------------

st.markdown("## 🔬 THz Imaging & Spectroscopy Analysis Suite")
st.write("Upload your THz `.txt` or `.npz` files to analyse 1D, 2D, and multi-region data.")

# -------------------------------------------------------------------------
# SIDEBAR
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

    st.markdown("### Upload your repeat measurement `.npz` files for each region / participant")
    render_pool_editor(section_key_prefix="occl")

    st.markdown("---")
    st.header("⚙️ Occlusion Settings")

    sf = st.number_input("Sampling Frequency (Hz)", value=12.5, step=0.5)
    duration = st.number_input("Occlusion Duration (s)", value=60.0, step=5.0)

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

    # PROCESS BUTTON
    if st.button("Process Regional Analysis"):

        pool_paths = []
        for pool in st.session_state.pools:
            paths = []
            if pool["files"] is not None:
                for f in pool["files"]:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
                    tmp.write(f.read())
                    tmp.close()
                    paths.append(tmp.name)

            if paths:
                pool_paths.append({
                    "name": pool["name"],
                    "color": pool["color"],
                    "paths": paths
                })

        if not pool_paths:
            st.warning("Please add at least one pool with uploaded `.npz` files.")
            st.stop()

        if analysis_type == "Occlusion Curve":
            st.session_state.occlusion_results = compare_regions_dynamic(
                pools=pool_paths,
                sf=sf,
                duration=duration,
                show_confidence=True,
                title_size=title_size,
                axis_label_size=axis_label_size,
                tick_size=tick_size,
                legend_size=legend_size,
                legend_loc=legend_position,
                line_width=line_width,
                alpha_value=alpha_value,
                grid_on=grid_on,
                return_data=True
            )

    # DISPLAY RESULTS IF THEY EXIST
    if "occlusion_results" in st.session_state:

        results = st.session_state.occlusion_results


        # Numerical analysis
        st.markdown("## 🔢 Numerical Analysis Across All Pools")

        common_time = results["common_time"]

        time_choice = st.number_input(
            "Choose a time (s) for analysis",
            min_value=float(common_time.min()),
            max_value=float(common_time.max()),
            value=float(common_time[len(common_time)//2]),
            step=0.1,
            format="%.2f"
        )

        idx = (np.abs(common_time - time_choice)).argmin()
        st.write(f"Nearest available time: **{common_time[idx]:.2f} s**")

        table = []
        for region_name, data in results["regions"].items():
            mean_n = data["mean"]
            std_n = data["std"]
            ci95 = 1.96 * std_n
        
            # Number of files in this pool
            N = len([p for p in st.session_state.pools if p["name"] == region_name][0]["files"])
        
            # Standard error of the mean
            sem = std_n[idx] / np.sqrt(N)
        
            # 95% CI on the mean (uncertainty)
            ci95_mean = 1.96 * sem
        
            table.append({
                "Pool": region_name,
                "Mean n(f)": float(mean_n[idx]),
                "Std": float(std_n[idx]),
                "95% CI (data spread) ±": float(ci95[idx]),
                "Uncertainty on Mean (95% CI) ±": float(ci95_mean),
                "Reported Value": f"{mean_n[idx]:.4f} ± {ci95_mean:.4f}"
            })


        st.dataframe(table)

# -------------------------------------------------------------------------
# REGIONAL REFRACTIVE INDEX COMPARISON
# -------------------------------------------------------------------------

elif section == "🔬 Regional Comparison — Refractive Index":

    st.markdown("### Upload your repeat measurement `.npz` files for each region / participant")
    render_pool_editor(section_key_prefix="ri")

    st.markdown("---")
    st.header("⚙️ Refractive Index Settings")

    fmin = st.number_input("Minimum Frequency (THz)", value=0.2, step=0.05)
    fmax = st.number_input("Maximum Frequency (THz)", value=2.0, step=0.05)
    analysis_time = st.number_input("Analysis Time (s)", value=20.0, step=1.0)

    # PROCESS BUTTON
    if st.button("Process Regional Refractive Index"):

        pool_paths = []
        for pool in st.session_state.pools:
            paths = []
            if pool["files"] is not None:
                for f in pool["files"]:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
                    tmp.write(f.read())
                    tmp.close()
                    paths.append(tmp.name)

            if paths:
                pool_paths.append({
                    "name": pool["name"],
                    "color": pool["color"],
                    "paths": paths
                })

        if not pool_paths:
            st.warning("Please add at least one pool with uploaded `.npz` files.")
            st.stop()

        st.session_state.ri_results = compare_regions_refractive_index_dynamic(
            pools=pool_paths,
            fmin=fmin,
            fmax=fmax,
            show_confidence=True,
            title_size=title_size,
            axis_label_size=axis_label_size,
            tick_size=tick_size,
            legend_size=legend_size,
            legend_loc=legend_position,
            line_width=line_width,
            alpha_value=alpha_value,
            grid_on=grid_on,
            analysis_time=analysis_time,
            return_data=True
        )

    # DISPLAY RESULTS IF THEY EXIST
    if "ri_results" in st.session_state:

        results = st.session_state.ri_results

        # Numerical analysis
        st.markdown("## 🔢 Numerical Analysis Across All Pools")

        freqs_plot = results["freqs"]

        freq_choice = st.number_input(
            "Choose a frequency (THz) for analysis",
            min_value=float(freqs_plot.min()),
            max_value=float(freqs_plot.max()),
            value=float(freqs_plot[len(freqs_plot)//2]),
            step=0.01,
            format="%.2f"
        )

        idx = (np.abs(freqs_plot - freq_choice)).argmin()
        st.write(f"Nearest available frequency: **{freqs_plot[idx]:.3f} THz**")

        table = []
        for region_name, data in results["regions"].items():
            mean_n = data["mean"]
            std_n = data["std"]
            ci95 = 1.96 * std_n
        
            # Number of files in this pool
            N = len([p for p in st.session_state.pools if p["name"] == region_name][0]["files"])
        
            # Standard error of the mean
            sem = std_n[idx] / np.sqrt(N)
        
            # 95% CI on the mean (uncertainty)
            ci95_mean = 1.96 * sem
        
            table.append({
                "Pool": region_name,
                "Mean n(f)": float(mean_n[idx]),
                "Std": float(std_n[idx]),
                "95% CI (data spread) ±": float(ci95[idx]),
                "Uncertainty on Mean (95% CI) ±": float(ci95_mean),
                "Reported Value": f"{mean_n[idx]:.4f} ± {ci95_mean:.4f}"
            })


        st.dataframe(table)
