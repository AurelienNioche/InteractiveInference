import expyriment
from expyriment import stimuli, misc
import time
import numpy as np
from scipy.special import expit

from pathlib import Path
from scipy.signal import butter, sosfiltfilt


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    flt = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    y = sosfiltfilt(flt, data)
    return y


def main():

    base_radius = 10
    var_radius = 20

    max_radius = base_radius + var_radius

    alpha, beta = 0.01, 0.01

    v = 0.99  # Threshold

    time_window_sec = 1.0
    samples_noise_sec = 3.0
    mvt_size_noise = 0.05
    mvt_size_mouse = 0.5

    fs = 10.0  # sample rate, Hz
    cutoff = 1.0  # desired cutoff frequency of the filter, Hz
    order = 2  # sin wave can be approx represented as quadratic

    time_window = int(time_window_sec * fs)
    n_sample_noise = int(samples_noise_sec*fs)

    # expyriment.control.defaults.initialize_delay = 0
    expyriment.control.set_develop_mode(True)

    exp = expyriment.design.Experiment(name="Text Experiment")

    # exp.data_variable_names = ["trial", "timestamp", "img"]

    expyriment.control.initialize(exp)

    txt = stimuli.TextLine("yo")
    dot = stimuli.Circle(radius=40)

    expyriment.control.start(experiment=exp, subject_id=0, skip_ready_screen=True)

    rng = np.random.default_rng()

    x_noise, y_noise = None, None

    fx, fy = 0.1, 0.1

    gamma = 0

    hist = {k: np.random.uniform(0, 1, size=time_window) for k in ('x', 'y', 'fx', 'fy')}

    i = 0
    while True:
        try:
            if i == 0:
                # smp = rng.normal(size=(2, n_sample_noise))
                # x_noise = butter_lowpass_filter(smp[0], cutoff=cutoff, fs=fs, order=order)
                # y_noise = butter_lowpass_filter(smp[1], cutoff=cutoff, fs=fs, order=order)
                x_noise = np.ones(n_sample_noise)
                x_noise[:] = 0.1  # 0.0001
                x_noise[int(len(x_noise)/2):] = - 0.1  # -0.0001
                print(x_noise)
                y_noise = np.zeros(n_sample_noise)

            # Reference position is middle
            # Direction is left => right, bottom => top
            x_max, y_max = exp.screen.window_size

            mouse_x, mouse_y = exp.mouse.position
            mouse_x /= x_max
            mouse_y /= y_max

            fx = fx + x_noise[i] * mvt_size_noise
            fy = fy + y_noise[i] * mvt_size_noise

            print("fx", fx)

            x = fx + mouse_x * mvt_size_mouse
            y = fy + mouse_y * mvt_size_mouse

            x = min(max(x, 0), 1)
            y = min(max(y, 0), 1)

            values = {
                'x': x,
                'y': y,
                'fx': fx,
                'fy': fy
            }

            for k, val in hist.items():
                hist[k] = np.roll(val, -1)
                hist[k][-1] = values[k]

            print()
            print(hist["fx"])

            mean = {k: hist[k].mean() for k in hist.keys()}
            rms_pos = np.sqrt(((x - mean['x'])**2 + (y - mean['y'])**2).mean())
            rms_f = np.sqrt(((fx - mean['fx'])**2 + (fy - mean['fy'])**2).mean())

            print("rms_f", rms_f)

            r = rms_pos / rms_f
            print("r", r)

            need_increase = r < v
            print(need_increase)
            if need_increase:
                gamma += alpha * r
            else:
                gamma -= beta * r

            min_gamma = -10
            max_gamma = +10
            gamma = min(max_gamma, gamma)
            gamma = max(min_gamma, gamma)
            base_gamma = 1
            p = expit(gamma+base_gamma)
            print(f"gamma = {gamma:.2f} p= {p:.2f}")
            x_scaled = x - 0.5
            y_scaled = y - 0.5
            x_scaled *= x_max
            y_scaled *= y_max

            radius = base_radius + var_radius * p

            if p > 0.99:
                color = misc.constants.C_RED
            else:
                color = misc.constants.C_YELLOW

            txt.text = f"r={r:.2f}"
            txt.present(clear=False)

            # exp.clock.wait(1000/fs)

            dot.radius = radius,
            dot.colour = color,
            dot.position = (x_scaled, y_scaled)
            dot.present(clear=False)
            # dot.present()

            exp.clock.wait(1000/fs)
            print("window size", exp.screen.window_size)

            # exp.data.add([i, time.time(), img])
            # To convert back ts: datetime.utcfromtimestamp(ts)
            i += 1
            i %= n_sample_noise

        except KeyboardInterrupt:
            break

    expyriment.control.end()

    # expyriment.misc.data_preprocessing.write_concatenated_data(
    #     data_folder="data", file_name="main",
    #     output_file="results.csv")


if __name__ == '__main__':
    main()
