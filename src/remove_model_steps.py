import glob
import os
for root, dirs, files in os.walk("/disk1/sajad/sci-trained-models/presum/", topdown=True):

    for dir in dirs:
        for f in glob.glob(os.path.join(root, dir) + '/*.pt'):
            if f.split('/')[-1].startswith('model_step'):
                os.remove(f)
