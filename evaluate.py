import torch
import torchvision.transforms as T
from pathlib import Path
from ssim import SSIM
import pickle
from PIL import Image
import pandas as pd

def evaluate_dataset(datapath, modelpath):
    hipath = datapath/'high'
    lopath = datapath/'low'
    outpath = datapath/modelpath.name
    if outpath.exists() != True:
        outpath.mkdir(parents=True)
    with open(modelpath/'model.pkl', 'rb') as f:
        model = pickle.load(f)
        model.to('cpu')
        print('Model loaded succesfully')
        f.close()
    ssim = SSIM()
    result = pd.DataFrame(columns=['filename', 'luminance', 'structure', 'contrast', 'ssim'])

    for file in hipath.iterdir():
        original = Image.open(file)
        low = Image.open(lopath/file.name)
        low = T.ToTensor()(low).unsqueeze(0)
        recons = model.get_clean_image(low)[0]
        recons = T.ToPILImage()(recons)
        luminance, contrast, structure = ssim.calculate_ssim(original, recons)
        l, c, s = torch.mean(luminance), torch.mean(contrast), torch.mean(structure)
        overall = torch.mean(luminance*contrast*structure)
        entry = {
            'filename': file.name,
            'luminance': l.item(),
            'structure': c.item(),
            'contrast': s.item(),
            'ssim': overall.item()    
        }
        result = pd.concat([result, pd.DataFrame([entry])], ignore_index=True)
        recons.save(outpath/file.name)
        print(f'{file.name} processed')
    result.to_csv(outpath/'result.csv')
    with open(outpath/'ssim.txt', 'w') as f:
        f.write(f'Overall ssim is {result.ssim.mean()}')
        f.close()



if __name__ == "__main__":
    datapath = Path.cwd()/'data/lol_dataset/eval15'
    modelpath = Path.cwd()/'models/noise_exp_3'
    evaluate_dataset(datapath, modelpath)