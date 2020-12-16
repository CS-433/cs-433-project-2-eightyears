# Machine learning

## Project 2: Machine Learning for Science

In this project, we use [AtLoc](https://github.com/BingCS/AtLoc) to predict positions and orientations for datasets provided by the TOPO lab.

### Step-by-step

- [x] Try to run the code on GitHub
- [x] Plug the TOPO dataset into AtLoc
- [x] Get first results
- [x] Investigate features for best predictions
- [ ] Optimize results

### Submission

* **Code**: batch scripts running Python with `torch` ([Scripts](./scripts)) and Python code to adapt AtLoc ([Dataloader](AtLoc-master/data/dataloaders.py), as well as [Run, Train and Eval](AtLoc-master/))
* **Report**: maximum 4 page PDF ([Overleaf Report Draft](https://www.overleaf.com/5419823158fvrtbssxbvwf))

We don't know exactly how the grading will happen... Apparently, it is split evenly between the technical and the coding components.

### Running Code on IZAR

To facilitate reproducibility, here are a few resources to recreate the same environment as we used:

* **Virtual Environment**: [this tutorial](./venvs.md) explains how to set up the `atloc` virtualenv
* **Data**: [this script](scripts/prepare_air) is meant to copy the real and synthetic images of the comballaz dataset with air scene to the right locations (to be run as follows: `scripts/prepare_air`)
* **Train**: [this script](scripts/train_air) trains the model on the comballaz dataset with air scene and puts the resuling weights to `AtLoc-master/logs` (to be run on GPUs *via* `sbatch scripts/train_air --wait`)
* **Test**: [this script](scripts/eval_air) evaluates the previously established model and outputs saliency maps (to be run the same way as `train_air`, with `sbatch scripts/eval_air --wait`)

The `prepare_air` script has not been tested (yet), mainly because I had already run those copy/unzip commands before and didn't need to repeat them.

