from pydub import AudioSegment, effects

import sys,os
import shutil
from tqdm import tqdm
from pathlib import Path
from timeit import default_timer as timer


from utils.cli_parser import parse_cli_overides
from utils.config import get_dataset


def main(args, cfg):
    """Execute a task based on the given command-line arguments.

    This function is the main entry-point of the program. It allows the
    user to extract features, train a model, infer predictions, and
    evaluate predictions using the command-line interface.

    Args:
        args: command line arguments.
    Return:
        0: successful termination
        'any nonzero value': abnormal termination
    """

    # Create workspace
    Path(cfg['workspace_dir']).mkdir(parents=True, exist_ok=True)

    # Dataset initialization
    dataset = get_dataset(dataset_name=cfg['dataset'], root_dir=cfg['dataset_dir'])


    # Norm the audio and save it to proper space
    data_dir_list = [
        dataset.dataset_dir['eval']['foa']
    ]
    print(data_dir_list)

    for h5_dir in data_dir_list:
        h5_dir=Path(os.path.join(h5_dir,'normed'))
        if h5_dir.is_dir():
            # flag = input("norm folder {} is already existed, delete it? (y/n)".format(h5_dir)).lower()
            # if flag == 'y':
            shutil.rmtree(h5_dir)
            # elif flag == 'n':
            #     print("User select not to remove the norm folder {}. The process will quit.\n".format(h5_dir))
            #     return
        h5_dir.mkdir(parents=True)

    for idx, data_dir in enumerate(data_dir_list):
        begin_time = timer()
        data_list = [path for path in sorted(data_dir.glob('*.wav')) if not path.name.startswith('.')]
        iterator = tqdm(data_list, total=len(data_list), unit='it')
        audio_count = 1
        for data_path in iterator:

            print(data_path)
            #file_name=str(data_path).split('\\')[-1]
            #print(file_name)
            rawsound = AudioSegment.from_file(data_path, "wav")
            normalizedsound = effects.normalize(rawsound)
            normalizedsound -= 2
            normalizedsound = normalizedsound.low_pass_filter(5000)
            normalizedsound = normalizedsound.high_pass_filter(200)
            normalizedsound.export(os.path.join(data_dir,'normed',"{}_normed.wav".format(audio_count)), format="wav")
            #pass
            audio_count+=1

        iterator.close()
        print("Norm Audio finished! Time spent: {:.3f} s".format(timer() - begin_time))


    return 0


if __name__ == '__main__':
    args, cfg = parse_cli_overides()
    sys.exit(main(args, cfg))
