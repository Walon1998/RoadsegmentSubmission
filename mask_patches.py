#!/usr/bin/env python

import glob
import os
import re

# Mask to submission
from mask_to_submission import masks_to_submission
from submission_to_mask import reconstruct_from_labels

submission_filename = 'temp_submission.csv'
image_filenames = glob.glob("./Data/Additional_Data/Masks/*.png")
masks_to_submission(submission_filename, *image_filenames)

for f in image_filenames:
    i = re.findall(r'\d+', os.path.basename(f))[0]
    print(i)
    reconstruct_from_labels(int(i))

# Submission to mask
