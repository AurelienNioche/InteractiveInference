# label_position_mouse = TextLine(window, position=(0.5, 0.4))

# x_noise_maker = LossPassFilteredWhiteNoise(
#     window_size_sec=noise_window_size_sec,
#     fps=window.fps,
#     rng=rng,
#     sd_noise=sd_noise)
# y_noise_maker = LossPassFilteredWhiteNoise(
#     window_size_sec=noise_window_size_sec,
#     fps=window.fps,
#     rng=rng,
#     sd_noise=sd_noise)

# mvt_size_noise = 0.01

    # cutoff = 0.1  # desired cutoff frequency of the filter, Hz
    # order = 2  # sin wave can be approx represented as quadratic
    #
    # sd_noise = 2.0
    # noise_window_size_sec = 2.0
    # rng = np.random.default_rng(123)

# n_sample_noise = int(samples_noise_sec * window.fps)

# samples_noise_sec = 6.0


# x_noise, y_noise = low_pass_filtered_white_noise(
#     sd_noise=sd_noise,
#     rng=rng,
#     n_sample_noise=n_sample_noise,
#     cutoff=cutoff,
#     fs=window.fps,
#     order=order)
# x_noise = np.ones(n_sample_noise)
# x_noise[:] = 0.01  # 0.0001
# x_noise[int(len(x_noise) / 2):] = - 0.01  # -0.0001
# y_noise = np.zeros(n_sample_noise)


# fx = x_center  # + x_noise # * mvt_size_noise
# fy = y_center  # + y_noise # * mvt_size_noise

# print("x_noise", x_noise, "y_noise", y_noise)


# fx = x
# fy = y


# label_position_mouse.update(f"mouse position = {mouse_x, mouse_y}")





# def low_pass_filtered_white_noise(
#         rng,
#         n_sample_noise,
#         sd_noise,
#         cutoff, fs, order):
#
#     def butter_lowpass_filter(data):
#         nyq = 0.5 * fs  # Nyquist Frequency
#         normal_cutoff = cutoff / nyq
#         # Get the filter coefficients
#         flt = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
#         y = sosfilt(flt, data)
#         return y
#
#     x_smp = rng.normal(scale=sd_noise, size=n_sample_noise)
#     y_smp = rng.normal(scale=sd_noise, size=n_sample_noise)
#     x_noise = butter_lowpass_filter(x_smp)
#     y_noise = butter_lowpass_filter(y_smp)
#     return x_noise, y_noise


# class LossPassFilteredWhiteNoise:
#
#     def __init__(self, window_size_sec, fps, rng, sd_noise):
#         self.buffer = np.zeros(int(window_size_sec*fps))
#         self.rng = rng
#         self.sd_noise = sd_noise
#
#     def new(self):
#         smp_i = self.rng.normal(scale=self.sd_noise)
#         self.buffer = np.roll(self.buffer, -1)
#         self.buffer[-1] = smp_i
#         return self.buffer.mean()
#
#
# class SinusoidalNoise:
#
#     def __init__(self, fps, f, a=0.1):
#         self.fps = fps
#         self.f = f
#         self.t = 0
#         self.a = a
#
#     def new(self):
#         v = np.sin(2*np.pi*self.f*self.t)
#         self.t += 1/self.fps
#         return v*self.a


# text.update(text=f"r={r:.2f}\ngamma={gamma}\np={p:.2f}", position=(0.7, 0.7))