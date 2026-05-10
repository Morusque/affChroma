
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

A4_HZ = 440.0
C0_MIDI = 12
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_cyclic_octave_position(freq):
    midi = 69 + 12 * np.log2(freq / A4_HZ)
    pitch_class = np.mod(midi, 12)
    return pitch_class / 12.0


def analyze_file(path, bins=240, fft_size=4096, hop_size=1024, min_freq=40, max_freq=8000, attenuation_exponent=0.5, smoothing=1.0):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    histogram = np.zeros(bins)

    window = np.hanning(fft_size)
    freqs = np.fft.rfftfreq(fft_size, 1 / sr)

    valid = (freqs >= min_freq) & (freqs <= max_freq)
    valid_freqs = freqs[valid]

    positions = freq_to_cyclic_octave_position(valid_freqs)
    bin_positions = positions * bins

    for start in range(0, len(audio) - fft_size, hop_size):
        frame = audio[start:start + fft_size] * window
        spectrum = np.abs(np.fft.rfft(frame))
        spectrum = spectrum[valid]

        for i, (pos, amp) in enumerate(zip(bin_positions, spectrum)):
            freq = valid_freqs[i]
            if attenuation_exponent > 0:
                weight = (min_freq / freq) ** attenuation_exponent
            else:
                weight = 1.0
            bin_index = int(np.round(pos)) % bins
            histogram[bin_index] += amp * weight

    if np.max(histogram) > 0:
        histogram /= np.max(histogram)

    if smoothing > 0 and bins > 12:
        sigma = smoothing * (bins / 12.0)
        x = np.arange(bins)
        dist = np.minimum(x, bins - x)
        kernel = np.exp(-0.5 * (dist / sigma)**2)
        kernel /= np.sum(kernel)
        hist_fft = np.fft.fft(histogram)
        kernel_fft = np.fft.fft(kernel)
        histogram = np.fft.ifft(hist_fft * kernel_fft).real

    return histogram


def merge_to_12_notes(histogram):
    bins = len(histogram)
    note_values = []

    for note in range(12):
        center = note / 12.0
        center_bin = center * bins

        total = 0.0
        weight_sum = 0.0

        for i, value in enumerate(histogram):
            dist = abs(i - center_bin)
            dist = min(dist, bins - dist)

            # Gaussian-ish soft note region
            sigma = bins / 48.0
            weight = np.exp(-(dist * dist) / (2 * sigma * sigma))

            total += value * weight
            weight_sum += weight

        note_values.append(total / weight_sum)

    return np.array(note_values)


def plot_continuous_histogram(histogram, title="Cyclic pitch histogram"):
    bins = len(histogram)
    x = np.arange(bins)

    plt.figure(figsize=(14, 4))
    plt.bar(x, histogram, width=1.0)

    note_positions = [i * bins / 12 for i in range(12)]
    plt.xticks(note_positions, NOTE_NAMES)

    plt.title(title)
    plt.xlabel("Pitch class cycle")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.show()


def plot_note_histogram(note_values, title="Notes from least used to most used"):
    order = np.argsort(note_values)
    sorted_names = [NOTE_NAMES[i] for i in order]
    sorted_values = note_values[order]

    plt.figure(figsize=(10, 4))
    plt.bar(sorted_names, sorted_values)
    plt.title(title)
    plt.xlabel("Notes, least used to most used")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.show()

    print("Least used to most used:")
    for name, value in zip(sorted_names, sorted_values):
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    print("Starting analysis...")
    folder = os.path.join(os.path.dirname(__file__), "audio")
    extensions = (".wav", ".flac", ".aiff", ".aif", ".mp3")
    
    files = [f for f in os.listdir(folder) if f.lower().endswith(extensions)]
    histograms = {}
    combined = None
    initial_file = files[0] if files else None

    if initial_file:
        path = os.path.join(folder, initial_file)
        try:
            histograms[initial_file] = analyze_file(path, bins=12, fft_size=8192, hop_size=1000, min_freq=20, max_freq=10000, attenuation_exponent=0.5, smoothing=1.0)
        except Exception as e:
            print(f"Error analyzing {initial_file}: {e}")

    # GUI
    root = tk.Tk()
    root.title("Audio Chromagram Analyzer")
    
    # File selection
    tk.Label(root, text="Select File:").pack()
    file_var = tk.StringVar()
    file_combo = ttk.Combobox(root, textvariable=file_var, values=["Combined"] + files)
    if initial_file:
        file_combo.set(initial_file)
    else:
        file_combo.set("Combined")
    file_combo.pack()
    
    # Parameters
    params_frame = tk.Frame(root)
    params_frame.pack(side=tk.TOP)
    
    # Bins
    bins_frame = tk.Frame(params_frame)
    bins_frame.pack(side=tk.LEFT, padx=5)
    tk.Label(bins_frame, text="Bins:").pack()
    bins_var = tk.IntVar(value=12)
    bins_scale = tk.Scale(bins_frame, from_=12, to=480, orient=tk.VERTICAL, variable=bins_var)
    bins_scale.pack()
    
    # FFT Size
    fft_frame = tk.Frame(params_frame)
    fft_frame.pack(side=tk.LEFT, padx=5)
    tk.Label(fft_frame, text="FFT Size:").pack()
    fft_var = tk.IntVar(value=8192)
    fft_scale = tk.Scale(fft_frame, from_=2048, to=8192, orient=tk.VERTICAL, variable=fft_var)
    fft_scale.pack()
    
    # Hop Size
    hop_frame = tk.Frame(params_frame)
    hop_frame.pack(side=tk.LEFT, padx=5)
    tk.Label(hop_frame, text="Hop Size:").pack()
    hop_var = tk.IntVar(value=1000)
    hop_scale = tk.Scale(hop_frame, from_=256, to=2048, orient=tk.VERTICAL, variable=hop_var)
    hop_scale.pack()
    
    # Min Freq
    minf_frame = tk.Frame(params_frame)
    minf_frame.pack(side=tk.LEFT, padx=5)
    tk.Label(minf_frame, text="Min Freq:").pack()
    minf_var = tk.IntVar(value=20)
    minf_scale = tk.Scale(minf_frame, from_=20, to=200, orient=tk.VERTICAL, variable=minf_var)
    minf_scale.pack()
    
    # Max Freq
    maxf_frame = tk.Frame(params_frame)
    maxf_frame.pack(side=tk.LEFT, padx=5)
    tk.Label(maxf_frame, text="Max Freq:").pack()
    maxf_var = tk.IntVar(value=10000)
    maxf_scale = tk.Scale(maxf_frame, from_=4000, to=16000, orient=tk.VERTICAL, variable=maxf_var)
    maxf_scale.pack()
    
    # Freq Atten Exp
    atten_frame = tk.Frame(params_frame)
    atten_frame.pack(side=tk.LEFT, padx=5)
    tk.Label(atten_frame, text="Freq Atten Exp:").pack()
    atten_var = tk.DoubleVar(value=0.5)
    atten_scale = tk.Scale(atten_frame, from_=0.0, to=2.0, resolution=0.1, orient=tk.VERTICAL, variable=atten_var)
    atten_scale.pack()
    
    # Smoothing
    smooth_frame = tk.Frame(params_frame)
    smooth_frame.pack(side=tk.LEFT, padx=5)
    tk.Label(smooth_frame, text="Smoothing:").pack()
    smooth_var = tk.DoubleVar(value=1.0)
    smooth_scale = tk.Scale(smooth_frame, from_=0.0, to=3.0, resolution=0.1, orient=tk.VERTICAL, variable=smooth_var)
    smooth_scale.pack()
    
    # Plot canvas
    fig = Figure(figsize=(14, 4))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()
    
    def plot_hist(hist, title):
        fig.clear()
        ax = fig.add_subplot(111)
        bins = len(hist)
        x = np.arange(bins)
        ax.bar(x, hist, width=1.0)
        note_positions = [i * bins / 12 for i in range(12)]
        ax.set_xticks(note_positions)
        ax.set_xticklabels(NOTE_NAMES)
        ax.set_title(title)
        ax.set_xlabel("Pitch class cycle")
        ax.set_ylabel("Energy")
        canvas.draw()
    
    def update_plot():
        selected = file_var.get()
        if selected == "Combined":
            if combined is None:
                start_reanalyze_all()
                return
            hist = combined
            title = "Combined cyclic pitch histogram"
        else:
            # Re-analyze with current params
            path = os.path.join(folder, selected)
            try:
                hist = analyze_file(path, bins=bins_var.get(), fft_size=fft_var.get(), hop_size=hop_var.get(), min_freq=minf_var.get(), max_freq=maxf_var.get(), attenuation_exponent=atten_var.get(), smoothing=smooth_var.get())
                title = selected
            except Exception as e:
                print(f"Error analyzing {selected}: {e}")
                return
        plot_hist(hist, title)
    
    def reanalyze_all():
        start_reanalyze_all()
    
    process_state = {
        "active": False,
        "idx": 0,
        "total": 0,
    }

    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack()
    update_button = tk.Button(button_frame, text="Update Plot", command=update_plot)
    update_button.pack(side=tk.LEFT)
    reanalyze_button = tk.Button(button_frame, text="Re-analyze All", command=lambda: start_reanalyze_all())
    reanalyze_button.pack(side=tk.LEFT)
    tk.Button(button_frame, text="Set 12 Bins", command=lambda: bins_var.set(12)).pack(side=tk.LEFT)
    
    status_var = tk.StringVar(value="0/0 files processed")
    status_label = tk.Label(root, textvariable=status_var)
    status_label.pack()

    def start_reanalyze_all():
        process_state["active"] = True
        process_state["idx"] = 0
        process_state["total"] = len(files)
        process_state["combined"] = None
        process_state["histograms"] = {}
        status_var.set(f"0/{process_state['total']} files processed")
        update_button.configure(state="disabled")
        reanalyze_button.configure(state="disabled")
        root.after(10, process_next_file)

    def process_next_file():
        global combined
        if process_state["idx"] >= process_state["total"]:
            process_state["active"] = False
            if process_state.get("combined") is not None:
                process_state["combined"] /= np.max(process_state["combined"])
                combined = process_state["combined"]
                combined_val = combined
            else:
                combined_val = None
            status_var.set(f"{process_state['total']}/{process_state['total']} files processed")
            update_button.configure(state="normal")
            reanalyze_button.configure(state="normal")
            if file_var.get() == "Combined" and combined_val is not None:
                plot_hist(combined_val, "Combined cyclic pitch histogram")
            return

        filename = files[process_state["idx"]]
        path = os.path.join(folder, filename)
        try:
            hist = analyze_file(path, bins=bins_var.get(), fft_size=fft_var.get(), hop_size=hop_var.get(), min_freq=minf_var.get(), max_freq=maxf_var.get(), attenuation_exponent=atten_var.get(), smoothing=smooth_var.get())
            process_state["histograms"][filename] = hist
            if process_state.get("combined") is None:
                process_state["combined"] = hist.copy()
            else:
                process_state["combined"] += hist
            process_state["idx"] += 1
            status_var.set(f"{process_state['idx']}/{process_state['total']} files processed")
            if file_var.get() == "Combined":
                combined_norm = process_state["combined"] / np.max(process_state["combined"])
                plot_hist(combined_norm, f"Combined cyclic pitch histogram ({process_state['idx']}/{process_state['total']})")
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            process_state["idx"] += 1
            status_var.set(f"{process_state['idx']}/{process_state['total']} files processed")
        root.after(10, process_next_file)

    # Initial plot
    update_plot()
    
    root.mainloop()