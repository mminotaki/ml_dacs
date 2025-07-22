<p align="center">
  <img src="./media/workflow_ml.png" alt="Workflow animation" width="1000"/>
</p>

<p align="center">
  <a href="https://github.com/mminotaki/ML-DACs" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/GitHub-ML--DACs-181717?style=flat&logo=github&logoColor=white" alt="GitHub Repo" />
  </a>
  <a href="https://doi.org/10.26434/chemrxiv-2025-pjdpq" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2025--pjdpq-blue?style=flat" alt="DOI" />
  </a>
  <a href="https://www.gnu.org/licenses/gpl-3.0.en.html" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/License-GPLv3-blue?style=flat&logo=gnu&logoColor=white" alt="License GPLv3" />
  </a>
  <a href="https://www.python.org" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python&logoColor=white" alt="Python Version" />
  </a>
</p>

---

## 📖 Overview

**Machine Learning Framework for Stability Evaluation of Dual-Atom Catalysts** or **ML-DACs** is a Python package designed to evaluate the stability of dual-atom catalysts (DACs) against aggregation into nanoparticles using a machine learning approach based on Random Forest Regression. This framework automates the workflow from data preprocessing to model training and evaluation, enabling researchers to efficiently predict catalyst stability and accelerate catalyst design.

---

## 🚀 Features

- 🔬 **Dual-Atom Catalyst Analysis**: Built specifically for handling DACs or any Single-Atom Catalysts datasets.
- 🧠 **Machine Learning Pipeline**: Preprocessing, training, evaluation, and feature selection all in one.
- 🌳 **Random Forest Regression**: Flexible error evaluation using K-Fold and Dual-Metal LOGOCV.
- 📊 **Insightful Visualizations**: Parity plots, error distributions, and model performance trends.

---

## ⚙️ Installation

To install ML-DACs, clone this repository and install the dependencies:

```bash
git clone https://github.com/mminotaki/ML-DACs.git
cd ML-DACs
pip install -r requirements.txt
```
---

## 📂 Dataset Access

The dataset is available in [Zenodo](https://doi.org/10.5281/zenodo.1234567) repository in pickle format. 

---

## 🤝 Contributing

We welcome contributions and suggestions!
If you find bugs, want to add new features, or improve documentation:

Fork the repo

Create your feature branch: git checkout -b feature/your-feature

Commit your changes

Open a pull request

---

## 📖 Citation

If you use **ML-DACs** in your work, please cite our preprint:

> Maria G. Minotaki and Núria López. **Transfer learning with domain knowledge adaptation for stability evaluation of dual-atom catalysts on nitrogen-doped carbon.**  *ChemRxiv* (2025). https://doi.org/10.26434/chemrxiv-2025-pjdpq


```bibtex
@article{mldacs,
  author       = {Minotaki, Maria G. and López, Núria},
  title        = {Transfer learning with domain knowledge adaptation for stability evaluation of dual-atom catalysts on nitrogen-doped carbon},
  journal      = {submitted},
  year         = {2025},
  doi          = {},
}
```
---
## ⚖️ License

**ML-DACs** is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html) – see the [LICENSE](LICENSE) file for details.

---

## 📚 References

- Minotaki, M.G. & López, N. *Transfer learning with domain knowledge adaptation for stability evaluation of dual-atom catalysts on nitrogen-doped carbon.* [ChemRxiv (2025)](https://doi.org/10.26434/chemrxiv-2025-pjdpq)<br><br>

- Minotaki, M.G., Geiger, J., Ruiz-Ferrando, A., Sabadell-Rendón, A., López, N. *A generalized model for estimating adsorption energies of single atoms on doped carbon materials.* *J. Mater. Chem. A*, **12**, 11049–11061 (2024). [https://doi.org/10.1039/D3TA05898K](https://doi.org/10.1039/D3TA05898K)
