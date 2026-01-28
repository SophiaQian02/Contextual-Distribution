import subprocess
import re
import itertools
import sys
import os
import time





hyperparameters = {


    'virtual_node': [True],
    'residual': [True],
    'se_type': ['entropy'],
    


    'num_layer': [4, 8, 16],
    'hidden_size': [128, 256],
    

    'K': [8, 12],
    



    'norm_type': ['Batch', 'Layer'],
    'lr': [0.001, 0.0005],
    'dropout': [0.0, 0.1],


    'model_name': ['KPGIN'],
    'batch_size': [64],
    'num_epochs': [200],
    'gpu_ids': ['0']
}
# ===========================================

def run_experiment(params, log_file):
    cmd = ["python", "-u", "train_Petti.py"] # -u ensuring unbuffered output
    

    for key, value in params.items():

        if key == 'gpu_ids':
            os.environ['CUDA_VISIBLE_DEVICES'] = value
            continue
            

        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")

        elif key == 'dropout':
             cmd.extend(["--drop_prob", str(value)])
        else:
            cmd.extend([f"--{key}", str(value)])
            

    save_dir = f"./tuning_logs/{int(time.time())}"
    cmd.extend(["--save_dir", save_dir])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:


        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        

        output = result.stdout + result.stderr
        


        match = re.search(r"Best Test AP: (0\.\d+)", output)
        val_match = re.search(r"Best Validation AP: (0\.\d+)", output)
        
        test_score = float(match.group(1)) if match else 0.0
        val_score = float(val_match.group(1)) if val_match else 0.0
        

        with open(log_file, 'a') as f:
            f.write(f"Params: {params}\n")
            f.write(f"Validation AP: {val_score:.5f}, Test AP: {test_score:.5f}\n")
            f.write("-" * 50 + "\n")
            
        return test_score
            
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error code {e.returncode}")

        print("Last 20 lines of output:")
        print('\n'.join((e.stdout + e.stderr).splitlines()[-20:]))
        return -1.0

def main():
    if not os.path.exists("tuning_logs"):
        os.makedirs("tuning_logs")
    
    log_file = "tuning_summary.txt"
    with open(log_file, 'w') as f:
        f.write("Hyperparameter Tuning Results\n============================\n")

    keys = hyperparameters.keys()
    values = hyperparameters.values()
    combinations = list(itertools.product(*values))
    
    print(f"Total experiments to run: {len(combinations)}")
    
    best_score = 0
    best_params = None
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n================ Experiment {i+1}/{len(combinations)} ================")
        print(f"Params: {params}")
        
        score = run_experiment(params, log_file)
        
        if score == -1.0:
            print("Experiment FAILED.")
        else:
            print(f"Experiment SUCCESS. Test AP: {score:.5f}")
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"*** NEW BEST SCORE ***")

    print("\n" + "="*50)
    print("Tuning Completed")
    print("="*50)
    print(f"Best Test AP: {best_score}")
    print("Best Parameters:")
    if best_params:
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    

    with open(log_file, 'a') as f:
        f.write("\n\nFINAL BEST RESULT\n")
        f.write(f"Test AP: {best_score}\n")
        f.write(f"Params: {best_params}\n")

if __name__ == "__main__":
    main()
