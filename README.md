
# ProjectAlpha is an open souce development as a precursor to ProjectBravo which intends to leverage AI to grant the ability to those whom have a disability which has either lost the ability to or has never been able to play games.

# Chatroom https://discord.gg/8S5h3Un4

# Requirements

# CUDA 12.8 - https://developer.download.nvidia.com/compute/cuda/12.8.1/network_installers/cuda_12.8.1_windows_network.exe

# PYTORCH - pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# pip install -r requirements.txt via cmd from the project folder (*note This was a pip freeze everything there is not required I will go through it when I get time)

# For bettercam - open settings on your computer type graphics settings in the search bar and open it. scroll down to add app select your python installation location and add python.exe, next to the right where it says auto select this and put low graphics settings.


# Functionality

# Run program - The program may be deployed bu running gui.py or by running app.py

# Settings - settings may be edited via the config.ini or from the GUI (*reccomended) than select the save button at the bottom

# Export model - To export a model to engine select select the export tab input the name of your model e.g. fn320.py change valid settings e.g. model size 320 and select the export model tab on the bottom. The gui window will not function during export and will seem as if the program crashed during this phase. It takes time be patient.

# Changing model - to change the model (neccessary after exporting to engine) select the AI tab and input your model name (must me in models folder) e.g. fn320.engine change valid settings e.g. model size 320 and hit save button

# Neural Network -  Training the neural network model to enhance mouse functionality. Select the Neural Network tab on top. Change the setting mouse_mlp setting to true.  change settings if desired (leave as default first time and change settings one at a time to understand how they affect it) than select the train MLP button located at the bottom. this takes several minuites and the progress may be observed in the cmd window.

# documentation for model training and exporting comming soon.





