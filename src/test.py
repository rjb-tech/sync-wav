from os.path import dirname, join as pjoin
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip
import os
from scipy.io import wavfile
import numpy as np
import scipy.io
import subprocess as sp
from PIL import Image
import torch
import math
from typing import List
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from model import Generator

DESIRED_FRAME_RATE = 10


class WavFile:
    def __init__(self, filename: str) -> None:
        self.sample_rate, self.audio_data = wavfile.read(filename)

        self.duration_in_seconds = math.floor(len(self.audio_data) / self.sample_rate)
        self.fps = DESIRED_FRAME_RATE
        self.bits_per_frame = math.floor(self.sample_rate / self.fps)

        # This is how much data we will have to vectorize for each video frame within a second of audio
        #   One second of audio has (sample_rate) values that need to be divided
        self.data_point_per_frame = math.floor(self.sample_rate / self.fps)

    def __get_data_for_each_second(self):
        scanner_length = self.sample_rate
        scanner_start_cursor = 0
        scanner_end_cursor = scanner_start_cursor + scanner_length
        data_by_second = []
        for seconds_index in range(0, self.duration_in_seconds):
            data_by_second.append(
                [*self.audio_data[scanner_start_cursor:scanner_end_cursor]]
            )

            scanner_start_cursor += scanner_length
            scanner_end_cursor += scanner_length

        return data_by_second

    def __get_frames_data_for_one_second(self, second_queried):
        scanner_begin = second_queried * self.bits_per_frame
        scanner_end = scanner_begin + self.bits_per_frame

        frames_data = []
        for frame_index in range(0, self.fps):
            frames_data.append(self.audio_data[scanner_begin:scanner_end])

            scanner_begin += self.bits_per_frame
            scanner_end += self.bits_per_frame

        return frames_data

    def get_formatted_audio_data(self):
        formatted_data = []
        for second_index in range(0, self.duration_in_seconds):
            formatted_data.append(self.__get_frames_data_for_one_second(second_index))

        return np.array(formatted_data)


def main():
    project_dir = os.getcwd()
    src_folder = os.path.join(project_dir, "src")
    data_folder = os.path.join(project_dir, "data")
    models_folder = os.path.join(project_dir, "models")
    outputs_folder = os.path.join(project_dir, "outputs")
    videos_folder = os.path.join(outputs_folder, "videos")
    audio_folder = os.path.join(data_folder, "audio")
    test_file = os.path.join(audio_folder, "cosmic_girl.wav")
    frames_folder = os.path.join(outputs_folder, "frames")
    model_file = os.path.join(models_folder, "009_10fps")
    model_state = torch.load(model_file)["generator"]["state_dict"]

    file = WavFile(test_file)
    data = file.get_formatted_audio_data()

    scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)

    # This has 735 elements. This is the data we will use to create the visualization
    # Create a model that has

    model = Generator()
    model.load_state_dict(model_state)
    summed_data = np.array(np.apply_along_axis(lambda x: np.sum(x), 3, data))

    for second in tqdm(range(file.duration_in_seconds)):
        for frame in range(file.fps):
            scaled_data = scaler.fit_transform(
                np.reshape(summed_data[second][frame], (-1, 1))
            )

            zoop = torch.from_numpy(np.reshape(scaled_data.astype("float32"), (1, -1)))

            output_image_data = model.forward(zoop)
            image = transforms.ToPILImage()(output_image_data[0])
            image.save(os.path.join(frames_folder, f"{second}-{frame}.jpg"), "JPEG")

            # plt.imshow(
            #     transforms.ToPILImage()(output_image[0]), interpolation="bilinear"
            # )
            # plt.show()

    clip = ImageSequenceClip(frames_folder, fps=DESIRED_FRAME_RATE)
    audio = AudioFileClip(test_file)
    actual_audio = CompositeAudioClip([audio])

    clip.audio = actual_audio
    clip.write_videofile(
        os.path.join(videos_folder, "output.mp4"),
        codec="libx264",
        audio=True,
        ffmpeg_params=[
            "-vf",
            f"minterpolate=fps={DESIRED_FRAME_RATE*3}:mi_mode=blend",
        ],
    )

    command = ["rm", "-rf", "outputs/frames/*.jpg"]
    sp.run(command, check=True)


if __name__ == "__main__":
    main()
