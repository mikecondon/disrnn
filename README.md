# Disentangled RNNs
This code implements disentangled RNNs on mice behaviour in a bandit task. The papers used are below.
I have made modifications to `disentangled_rnns/library/disrnn.py` and to `disentangled_rnns/library/rnn_utils.py`. I have made `switching_utils.py` and `switching.ipynb` from scratch.

To install the required dependencies, clone this repo and cd into it. Then with `uv sync`, all required dependencies should be installed!

    Beron, C. C., Neufeld, S. Q., Linderman, S. W., & Sabatini, B. L. (2022). Mice exhibit stochastic and efficient action switching during probabilistic decision making. Proceedings of the National Academy of Sciences, 119(15), e2113961119. https://doi.org/10.1073/pnas.2113961119
    Miller, K. J., Eckstein, M., Botvinick, M. M., & Kurth-Nelson, Z. (2023). Cognitive Model Discovery via Disentangled RNNs. Neuroscience. https://doi.org/10.1101/2023.06.23.546250
