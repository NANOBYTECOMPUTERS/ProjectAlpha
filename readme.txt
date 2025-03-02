Can be started from gui.py
Do not forget to export your model to engine or it will not load just click the export button.
I have not looked at the requirements my pip freeze is included.
be sure you export to engine and set your mouse to what you are using.
cuda 12.8
cudnn 9.7
python 3.11.9
and the nightly tpytorch cuda12.8
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
NOT TESTED ON AMD I use nvidia
 I will work on an installer with the minimum requirements later.
this is working kind of fantastic