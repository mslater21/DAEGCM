# DAEGCM
Modified DAEGC using a variety of different similarity measures.
Model is based off the paper Attributed Graph Clustering: [A Deep Attentional Embedding Approach](https://www.ijcai.org/Proceedings/2019/0509.pdf).

Much of the code was adapted from the GitHub repo [DAEGC](https://github.com/Tiger101010/DAEGC) by Tiger101010.

This code requires the following dependencies:
- torch
- torchvision
- torchaudio
- torch-scatter
- torch-sparse
- torch-cluster
- torch-spline-conv
- torch-geometric
- munkres
- scikit-learn
- numpy (installed with torch and scikit-learn)
- matplotlib
- IPython

# How to run
## Local Machine
The following commands will install the needed dependencies on a Windows machine:
### Terminal Commands
  pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  
  pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
  
  pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
  
  pip3 install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
  
  pip3 install torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
  
  pip3 install torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
  
  pip3 install munkres
  
  pip3 install -U scikit-learn
  
  pip3 install matplotlib

### Choose settings in main.py
Once dependencies are installed, run from main.py after selecting desired settings.

## Google Colab
Alternatively, the code can be executed at this [Google Colab notebook](https://colab.research.google.com/drive/1kitNm2RNXJskZ__9l1JmRARK1hRIYvx_#scrollTo=tpyywrQkoLnK).
