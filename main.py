import argparse
import importlib

SUPPORTED_COMMANDS = ['train', 'test']
SUPPORTED_MODELS = ['xgboost', 'word2vec', 'bert']

def main(args):
  model = importlib.import_module(args.model)
  function = getattr(model, f'start_{args.command}')
  function()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'command', metavar='command', 
    choices=SUPPORTED_COMMANDS, 
    help=f'Whether to start training or to test a model. Options: {", ".join(SUPPORTED_COMMANDS)}'
  )
  parser.add_argument(
    'model', metavar='model', 
    choices=SUPPORTED_MODELS, 
    help=f'Which model to train or test. Options: {", ".join(SUPPORTED_MODELS)}'
  )
  args = parser.parse_args()
  main(args)