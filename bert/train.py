from data import load_with_datasets

def start_train():
  data = load_with_datasets()
  print(data['train'][0])