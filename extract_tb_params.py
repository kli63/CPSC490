import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import sys

log_file = "/home/kli63/dev/CPSC490/nglod/sdf-net/_results/logs/runs/test_a_shark/events.out.tfevents.1745169702.ThisIsLaptop.39546.0"

try:
    for e in summary_iterator(log_file):
        for v in e.summary.value:
            if v.tag == "Parameters/text_summary":
                print(v.tensor.string_val[0].decode('utf-8'))
                sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
