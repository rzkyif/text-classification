import json

MODELS = ['1-16', '4-16', '4-32']

def initialize_data_files(model_name):
  loss_data = open(f'./bert/results/{model_name}_loss.csv', 'w')
  loss_data.write('loss,epoch\n')
  accuracy_data = open(f'./bert/results/{model_name}_accuracy.csv', 'w')
  accuracy_data.write('loss,accuracy,runtime,samples_per_second,steps_per_second,epoch\n')
  return (loss_data, accuracy_data)

with open('./bert/results/training.log', 'r') as f:
  i = 0
  loss_data, accuracy_data = initialize_data_files(MODELS[i])

  for line in f:
    thing = json.loads(line)

    if 'train_runtime' in thing:
      i += 1
      if i < len(MODELS):
        loss_data.close()
        accuracy_data.close()
        loss_data, accuracy_data = initialize_data_files(MODELS[i])
    elif 'eval_accuracy' in thing:
      accuracy_data.write(','.join([str(thing[x]) for x in thing]) + '\n')
    elif 'loss' in thing:
      loss_data.write(','.join([str(thing[x]) for x in ['loss', 'epoch']]) + '\n')

  loss_data.close()
  accuracy_data.close()
