import subprocess
from tqdm import tqdm

from src.utils.get_data import import_data


def convert_and_split(filename):
    # command_webm = ['ffmpeg', '-i', f'audio_data/{filename}.webm', '-c:a', 'pcm_f32le',
    #            f'wav_data/{filename}.wav']
    # subprocess.run(command_webm, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    command_ogg = ['ffmpeg', '-i', f'audio_data/{filename}.ogg', f'wav_data/{filename}.wav']
    subprocess.run(command_ogg, stdout=subprocess.PIPE, stdin=subprocess.PIPE)


DATA_PATH = '../../data'

if __name__ == '__main__':
    X, y = import_data(DATA_PATH, segmentation_type='no', drop_user_features=True, return_type='pd')
    for subject in tqdm(X.index.get_level_values(0)):
        convert_and_split(subject)
