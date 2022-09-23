import datasets

def load_with_datasets():
  datasets.disable_progress_bar()
  datasets.logging.set_verbosity_error()
  return datasets.load.load_dataset('csv', data_files={"train": './data/cleaned/train.csv', "test": './data/cleaned/test.csv'})