import os
import json
import shutil
from pytorch_lightning import Trainer, seed_everything
from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

seed_everything(7)
years = {'2014': 986, '2016': 1147, '2019': 1199}

def main():
    ckp_path = "/home/khai/Desktop/github/CNN-GNN-HMER/chuyende_tamer_temp/KetQua/4_Coord_Aware_GAT_1L_4H/checkpoints/best_model.ckpt"
    output_dir = "/home/khai/Desktop/github/CNN-GNN-HMER/chuyende_tamer_temp/KetQua/4_Coord_Aware_GAT_1L_4H/evaluation_results"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading model from checkpoint: {ckp_path}")
    model = LitTAMER.load_from_checkpoint(ckp_path)
    
    trainer = Trainer(logger=False, gpus=1)
    
    for test_year, test_num in years.items():
        print(f"\n==========================================")
        print(f"Evaluating CROHME {test_year}...")
        print(f"==========================================")
        
        dm = HMEDatamodule(
            folder="data/crohme",
            test_folder=test_year,
            max_size=320000,
            scale_to_limit=True,
        )
        
        metrics = trainer.test(model, datamodule=dm)[0]
        
        # Move generated files
        if os.path.exists("errors.json"):
            shutil.move("errors.json", os.path.join(output_dir, f"errors_{test_year}.json"))
        if os.path.exists("predictions.json"):
            shutil.move("predictions.json", os.path.join(output_dir, f"pred_{test_year}.json"))
            
        # Compute exact ExpRate and tolerated rates
        errors_path = os.path.join(output_dir, f"errors_{test_year}.json")
        if os.path.exists(errors_path):
            with open(errors_path, 'r') as jf:
                data = json.load(jf)
                exprate = test_num - len(data)
                exprate_1 = 0
                exprate_2 = 0
                for _, ele in data.items():
                    if ele.get('dist', 99) <= 1:
                        exprate_1 += 1
                    if ele.get('dist', 99) <= 2:
                        exprate_2 += 1
                exprate_1 = (exprate_1 + exprate) / test_num
                exprate_2 = (exprate_2 + exprate) / test_num
                exprate = exprate / test_num
                
                txt_path = os.path.join(output_dir, f"{test_year}.txt")
                with open(txt_path, 'w') as wf:
                    wf.write(f'ExpRate:  {exprate:.6f}\n')
                    wf.write(f'ExpRate<=1:  {exprate_1:.6f}\n')
                    wf.write(f'ExpRate<=2:  {exprate_2:.6f}\n')
                
                print(f"Results for CROHME {test_year}:")
                print(f"  ExpRate: {exprate:.4f}")
                print(f"  ExpRate<=1: {exprate_1:.4f}")
                print(f"  ExpRate<=2: {exprate_2:.4f}")
        else:
            print(f"Warning: errors.json not found for year {test_year}")

if __name__ == "__main__":
    main()
