# Multiple-Sclerosis-Detection

# Setup
- Install all the dependencies
  - First, install the torch and cudatoolkit using the below script
    - For Windows/Linux
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
     
     - For Mac
     
     ```# CUDA is not available on MacOS, please use default package```
    ```bash
      pip3 install torch torchvision torchaudio
    ```
     
  - Then, Install the other requirements, using the below script
  ```bash
  pip install requriements.txt
  ```

# Code Execution
- Execute the script for training the auto-encoder
```bash
python train_auto_encoder.py
```
- Execute the below script for training the classifier
```bash
python train_classifier.py
```
- Execute the below script for evaluating the classifier
```bash
python eval_classifier.py
```
  

