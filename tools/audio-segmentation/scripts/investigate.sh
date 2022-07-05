#!/bin/bash

set -e

CONFIG_FILE='./configs/ein_seld/seld.yaml'

## investigate with sigmoid
#python investigate.py --filename='result_sigmoid'

# investigate with no sigmoid
python investigate.py --filename='result_string'

### investigate with sigmoid and last layer 14 class
#python investigate.py --filename='result_brassband_with_last'
#
## investigate with no sigmoid and last layer 14 class
#python investigate.py --filename='result_brassband_sigmoid_with_last'

