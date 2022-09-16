import datasets

def load_with_datasets():
  return datasets.load.load_dataset('csv', data_files={"train": './data/cleaned/train.csv', "test": './data/cleaned/test.csv'})