import argparse
import subprocess
from mnist_to_bigquery import download_data_from_bigquery, run_mnist_to_bigquery
from compare_performance import find_best_model

parser = argparse.ArgumentParser()

parser.add_argument("--mode", help="[train], [serving], [upload_mnist], [download_mnist], [find_best_model]",
                    type=str, default='train')

flag = parser.parse_args()


if __name__ == '__main__':
    if flag.mode == 'train':
        # 8080이 사용되고 있으면 삭제
        subprocess.call('kill -9 $(lsof -t -i:8080)', shell=True)
        subprocess.call('nnictl create --config nni_config_torch.yml', shell=True)
    elif flag.mode == 'find_best_model':
        find_best_model()
    elif flag.mode == 'serving':
        subprocess.call("./cloud_functions/deploy.sh", shell=True)
    elif flag.mode == 'upload_mnist':
        run_mnist_to_bigquery()
    elif flag.mode == 'donwload_mnist':
        download_data_from_bigquery()
    else:
        print('not supported')
        raise Exception

