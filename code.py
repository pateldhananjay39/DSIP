import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import convolve

# Load audio file
audio, sample_rate = sf.read("song.wav")

# Convert stereo to mono
if len(audio.shape) == 2:
    audio = np.mean(audio, axis=1)

print("Sample Rate =", sample_rate)
print("Total Samples =", len(audio))

# Three Impulse Responses
ir1 = np.array([1])
ir2 = np.array([1, 0.5])
ir3 = np.array([1, -1])

impulse_responses = [ir1, ir2, ir3]

# Plot original signal
plt.figure(figsize=(10,3))
plt.plot(audio)
plt.title("Original Audio")
plt.show()

# Apply each impulse response
for i in range(len(impulse_responses)):

    print("\nProcessing IR", i + 1)

    kernel = impulse_responses[i]

    # Convolution
    convoluted = convolve(audio, kernel, mode="same")

    # Normalize
    convoluted = convoluted / np.max(np.abs(convoluted))

    # Save convoluted audio
    sf.write("IR" + str(i+1) + "_Convolution.wav", convoluted, sample_rate)

    # Simple inverse filter
    inverse_kernel = kernel[::-1]

    restored = convolve(convoluted, inverse_kernel, mode="same")

    restored = restored / np.max(np.abs(restored))

    # Save restored audio
    sf.write("IR" + str(i+1) + "_Inverse.wav", restored, sample_rate)

    # Plot graphs
    plt.figure(figsize=(10,6))

    plt.subplot(3,1,1)
    plt.plot(audio)
    plt.title("Original Audio")

    plt.subplot(3,1,2)
    plt.plot(convoluted)
    plt.title("Convolution Output")

    plt.subplot(3,1,3)
    plt.plot(restored)
    plt.title("Inverse Filter Output")

    plt.tight_layout()
    plt.show()

print("\nProgram Completed Successfully.")
