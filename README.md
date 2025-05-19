# Mithril ML Models Comparison

Hi! This is my project for comparing different machine learning models on the MNIST digit dataset. I created this project specifically to learn and experiment with the Mithril library - a flexible ML framework that can work with different computation backends. I'm a 3rd-year Computer Engineering student at TED University, but this is a personal project driven by my interest in ML frameworks.

## What It Does

- **Different Models**: I've built three models - Logistic Regression, MLP, and CNN
- **Mithril Integration**: Uses the Mithril library to create models that can run on different backends
- **Multiple Backends**: It can use different computation engines (NumPy, and others if you install them)
- **Live Results**: See how each model performs right away
- **Easy-to-use Interface**: Just draw or pick a digit, and see what the models predict
- **Performance Comparison**: See which model is fastest and most accurate

## Screenshots

### Model Predictions Comparison
![Model Predictions](Gradio%20Interface/Model%20Predictions%20Comparison.png)

### Real-time Performance Charts
![Performance Charts](Gradio%20Interface/Example%20of%20Real-time%20Performance%20Charts.png)

### Comparison Table
![Comparison Table](Gradio%20Interface/Comparison%20Table.png)

### Main Interface
![Gradio Interface](Gradio%20Interface/Screenshot%20of%20Gradio%20Interface.png)

## What You Need

- Python 3.8 or newer
- The packages in `requirements.txt` (including Mithril)

## How to Run

Just run the main program:
```bash
python mithril_models_comparison.py
```

A web page will open where you can:
1. Pick or draw a digit
2. Click "Compare Models" to see how each model does
3. Check which model is best

## Project Files

- `mithril_models_comparison.py`: The main program with all the models and UI
- `requirements.txt`: List of packages you need
- `README.md`: This file
- `.gitignore`: Tells Git which files to ignore
- `Gradio Interface/`: Folder with screenshots

## Making Changes

You can customize things by:
- Changing the training settings in the `train_params` dictionary
- Adding new types of models
- Trying different computation backends
- Experimenting with more Mithril features

## About Me

This project was created by Muammer Eren, a 3rd-year Computer Engineering student at TED University. I built this project to explore the capabilities of the Mithril library and compare different machine learning approaches. The main goal was to work hands-on with Mithril and understand how it lets you write ML models once and run them on different computation backends (NumPy, JAX, PyTorch, etc.). 
