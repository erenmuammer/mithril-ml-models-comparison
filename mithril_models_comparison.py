import numpy as np
import tensorflow as tf
import mithril as ml
import gradio as gr
import matplotlib.pyplot as plt
import time
from PIL import Image
from functools import partial

# Load MNIST dataset
print("Loading MNIST data...")
_xdata = tf.keras.datasets.mnist.load_data()
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = _xdata

# Normalize data
x_train = (x_train_raw.astype(np.float32) / 255.0)
x_test = (x_test_raw.astype(np.float32) / 255.0)
x_train_flatten = x_train.reshape(-1, 784)
x_test_flatten = x_test.reshape(-1, 784)

# One-hot encoding
y_train = np.eye(10)[y_train_raw].astype(np.float32)
y_test = np.eye(10)[y_test_raw].astype(np.float32)

# Available backends
bkends = {"NumPy Backend": ml.NumpyBackend()}

# Check which backends are available in the current installation
try:
    jx_bkend = ml.JaxBackend()
    bkends["JAX Backend"] = jx_bkend
    print("JAX Backend available.")
except Exception as e:
    print(f"JAX Backend unavailable: {e}")

try:
    trch_bkend = ml.TorchBackend()
    bkends["PyTorch Backend"] = trch_bkend
    print("PyTorch Backend available.")
except Exception as e:
    print(f"PyTorch Backend unavailable: {e}")

try:
    tf_bkend = ml.TensorFlowBackend()
    bkends["TensorFlow Backend"] = tf_bkend
    print("TensorFlow Backend available.")
except Exception as e:
    print(f"TensorFlow Backend unavailable: {e}")

# Base model class
class _NumPyModel:
    """Base model class using NumPy (abstract class)"""
    
    def __init__(self, model_name="Base Model"):
        self._r = np.random.RandomState(42)
        self.trained = False
        self._nm = model_name
    
    def train(self, X, y, epochs=5, batch_size=64, learning_rate=0.1):
        """Train the model - to be implemented by subclasses"""
        raise NotImplementedError("Must be implemented by subclasses")
    
    def predict(self, X):
        """Make predictions - to be implemented by subclasses"""
        raise NotImplementedError("Must be implemented by subclasses")
    
    def evaluate(self, X, y):
        """Evaluate on test set"""
        if not self.trained:
            return 0
            
        probs = self.predict(X)
        pred_labels = np.argmax(probs, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.sum(pred_labels == true_labels) / len(X)
        return accuracy


class LogisticRegression(_NumPyModel):
    """Simple logistic regression model using NumPy"""
    
    def __init__(self):
        super().__init__(model_name="Logistic Regression")
        # Xavier/Glorot initialization
        self.W = self._r.normal(0, 0.01, (784, 10)).astype(np.float32)
        self.b = np.zeros(10, dtype=np.float32)
    
    def train(self, X, y, epochs=5, batch_size=64, learning_rate=0.1):
        """Train the model"""
        start_time = time.time()
        
        n_samples = len(X)
        n_batches = n_samples // batch_size
        
        history = []
        
        # Learning rate decay
        init_lr = learning_rate
        
        for epoch in range(epochs):
            # Decay learning rate
            learning_rate = init_lr / (1 + 0.1 * epoch)
            
            t_loss = 0
            correct = 0
            
            # Shuffle data
            indices = self._r.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_x = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                logits = np.dot(batch_x, self.W) + self.b
                # Numerical stability for softmax
                logits = logits - np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                
                # Cross-entropy loss with numerical stability
                epsilon = 1e-10
                batch_loss = -np.mean(np.sum(batch_y * np.log(probs + epsilon), axis=1))
                t_loss += batch_loss
                
                # Calculate accuracy
                pred_labels = np.argmax(probs, axis=1)
                true_labels = np.argmax(batch_y, axis=1)
                correct += np.sum(pred_labels == true_labels)
                
                # Backpropagation
                d_logits = (probs - batch_y) / batch_size
                d_W = np.dot(batch_x.T, d_logits)
                d_b = np.sum(d_logits, axis=0)
                
                # Update parameters
                self.W -= learning_rate * d_W
                self.b -= learning_rate * d_b
            
            avg_loss = t_loss / n_batches
            accuracy = correct / (n_batches * batch_size)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Training accuracy: {accuracy:.4f}")
            history.append((avg_loss, accuracy))
        
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f} seconds")
        
        self.trained = True
        return history, train_time
    
    def predict(self, X):
        """Make predictions with numerical stability"""
        logits = np.dot(X, self.W) + self.b
        # Numerical stability for softmax
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs


class MLP(_NumPyModel):
    """Multi-Layer Perceptron implemented with NumPy"""
    
    def __init__(self):
        super().__init__(model_name="Multi-Layer Perceptron")
        # Two-layer MLP: 784 -> 128 -> 10
        # First layer (784 -> 128)
        self.W1 = self._r.normal(0, np.sqrt(2.0/784), (784, 128)).astype(np.float32)
        self.b1 = np.zeros(128, dtype=np.float32)
        # Second layer (128 -> 10)
        self.W2 = self._r.normal(0, np.sqrt(2.0/128), (128, 10)).astype(np.float32)
        self.b2 = np.zeros(10, dtype=np.float32)
    
    def _activation(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _activation_derivative(self, x):
        """ReLU derivative"""
        return (x > 0).astype(np.float32)
    
    def train(self, X, y, epochs=5, batch_size=64, learning_rate=0.01):
        """Train the MLP model"""
        start_time = time.time()
        
        n_samples = len(X)
        n_batches = n_samples // batch_size
        
        history = []
        
        # Learning rate decay
        init_lr = learning_rate
        
        for epoch in range(epochs):
            # Decay learning rate
            learning_rate = init_lr / (1 + 0.05 * epoch)
            
            t_loss = 0
            correct = 0
            
            # Shuffle data
            indices = self._r.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_x = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                # First layer
                z1 = np.dot(batch_x, self.W1) + self.b1
                a1 = self._activation(z1)
                # Output layer
                z2 = np.dot(a1, self.W2) + self.b2
                # Softmax
                z2 = z2 - np.max(z2, axis=1, keepdims=True)
                exp_z2 = np.exp(z2)
                probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
                
                # Cross-entropy loss
                epsilon = 1e-10
                batch_loss = -np.mean(np.sum(batch_y * np.log(probs + epsilon), axis=1))
                t_loss += batch_loss
                
                # Calculate accuracy
                pred_labels = np.argmax(probs, axis=1)
                true_labels = np.argmax(batch_y, axis=1)
                correct += np.sum(pred_labels == true_labels)
                
                # Backpropagation
                # Output layer error
                d_z2 = (probs - batch_y) / batch_size
                d_W2 = np.dot(a1.T, d_z2)
                d_b2 = np.sum(d_z2, axis=0)
                
                # Hidden layer error
                d_a1 = np.dot(d_z2, self.W2.T)
                d_z1 = d_a1 * self._activation_derivative(z1)
                d_W1 = np.dot(batch_x.T, d_z1)
                d_b1 = np.sum(d_z1, axis=0)
                
                # Update parameters
                self.W2 -= learning_rate * d_W2
                self.b2 -= learning_rate * d_b2
                self.W1 -= learning_rate * d_W1
                self.b1 -= learning_rate * d_b1
            
            avg_loss = t_loss / n_batches
            accuracy = correct / (n_batches * batch_size)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Training accuracy: {accuracy:.4f}")
            history.append((avg_loss, accuracy))
        
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f} seconds")
        
        self.trained = True
        return history, train_time
    
    def predict(self, X):
        """Make predictions"""
        # First layer
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._activation(z1)
        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        # Softmax
        z2 = z2 - np.max(z2, axis=1, keepdims=True)
        exp_z2 = np.exp(z2)
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        return probs


class SimpleCNN(_NumPyModel):
    """Simplified CNN-like network using NumPy"""
    
    def __init__(self):
        super().__init__(model_name="Simple CNN")
        # Simpler network with CNN-like architecture: 784 -> 256 -> 10
        # Input layer (784 -> 256)
        self.W1 = self._r.normal(0, np.sqrt(2.0/784), (784, 256)).astype(np.float32)
        self.b1 = np.zeros(256, dtype=np.float32)
        
        # Output layer (256 -> 10)
        self.W2 = self._r.normal(0, np.sqrt(2.0/256), (256, 10)).astype(np.float32)
        self.b2 = np.zeros(10, dtype=np.float32)
    
    def _activation(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def train(self, X, y, epochs=3, batch_size=32, learning_rate=0.005):
        """Train simplified CNN model"""
        start_time = time.time()
        
        # Flatten and normalize images if needed
        if len(X.shape) == 4:  # Image format (batch, c, h, w)
            X = X.reshape(-1, 784)
        elif len(X.shape) == 3:  # Image format (batch, h, w)
            X = X.reshape(-1, 784)
        
        n_samples = len(X)
        n_batches = n_samples // batch_size
        
        history = []
        
        for epoch in range(epochs):
            t_loss = 0
            correct = 0
            
            # Shuffle data
            indices = self._r.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_x = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                # Layer 1: Input -> Hidden
                z1 = np.dot(batch_x, self.W1) + self.b1
                a1 = self._activation(z1)
                
                # Layer 2: Hidden -> Output
                z2 = np.dot(a1, self.W2) + self.b2
                
                # Softmax
                z2 = z2 - np.max(z2, axis=1, keepdims=True)
                exp_z2 = np.exp(z2)
                probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
                
                # Calculate loss
                epsilon = 1e-10
                batch_loss = -np.mean(np.sum(batch_y * np.log(probs + epsilon), axis=1))
                t_loss += batch_loss
                
                # Calculate accuracy
                pred_labels = np.argmax(probs, axis=1)
                true_labels = np.argmax(batch_y, axis=1)
                correct += np.sum(pred_labels == true_labels)
                
                # Backpropagation
                # Output layer error
                dz2 = (probs - batch_y) / batch_size
                dW2 = np.dot(a1.T, dz2)
                db2 = np.sum(dz2, axis=0)
                
                # Hidden layer error
                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * (z1 > 0)  # ReLU derivative
                dW1 = np.dot(batch_x.T, dz1)
                db1 = np.sum(dz1, axis=0)
                
                # Update parameters
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
                
                if (i+1) % 10 == 0:
                    print(f"  Epoch {epoch+1}, Batch {i+1}/{n_batches}, Loss: {batch_loss:.4f}")
            
            avg_loss = t_loss / n_batches
            accuracy = correct / (n_batches * batch_size)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Training accuracy: {accuracy:.4f}")
            history.append((avg_loss, accuracy))
        
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f} seconds")
        
        self.trained = True
        return history, train_time
    
    def predict(self, X):
        """Make predictions"""
        # Flatten images if needed
        if len(X.shape) == 4:  # (batch, c, h, w)
            X = X.reshape(-1, 784)
        elif len(X.shape) == 3:  # (batch, h, w)
            X = X.reshape(-1, 784)
        
        # Forward pass
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._activation(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        
        # Softmax
        z2 = z2 - np.max(z2, axis=1, keepdims=True)
        exp_z2 = np.exp(z2)
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        
        return probs


class _MithrilAdapter(_NumPyModel):
    """Mithril adapter for different model types"""
    
    def __init__(self, backend_name, backend, model_type="logistic"):
        """
        Args:
            backend_name: Backend name (NumPy, JAX, PyTorch, etc.)
            backend: Mithril backend object
            model_type: Model type ("logistic", "mlp", "cnn")
        """
        _model_names = {
            "logistic": f"{backend_name} Logistic Regression",
            "mlp": f"{backend_name} MLP",
            "cnn": f"{backend_name} CNN"
        }
        super().__init__(model_name=_model_names.get(model_type, f"{backend_name} Model"))
        self._backend_name = backend_name
        self._backend = backend
        self._model_type = model_type
        self._params = None
        self._pm = None
        
        # Fallback model
        if model_type == "logistic":
            self._fallback = LogisticRegression()
        elif model_type == "mlp":
            self._fallback = MLP()
        elif model_type == "cnn":
            self._fallback = SimpleCNN()
        else:
            self._fallback = LogisticRegression()
    
    def train(self, X, y, epochs=3, batch_size=32, learning_rate=0.1):
        """Train Mithril model"""
        start_time = time.time()
        
        try:
            # Create Mithril model based on selected type
            from mithril.models import Model, Linear, Softmax, Mean, MLP, LogisticRegression
            from mithril.models.primitives import CrossEntropy, Relu
            from mithril.models.train_model import TrainModel
            
            # Prepare model parameters
            if self._model_type == "logistic":
                # Logistic regression model
                model = Model()
                linear = Linear(dimension=10)
                softmax = Softmax()
                
                model += linear(input="input", w="w", b="b")
                model += softmax(input=linear.output, output="output")
                
            elif self._model_type == "mlp":
                # MLP model
                model = Model()
                
                # First layer: 784 -> 128
                linear1 = Linear(dimension=128)
                relu = Relu()
                
                # Second layer: 128 -> 10
                linear2 = Linear(dimension=10)
                softmax = Softmax()
                
                # Connect components
                model += linear1(input="input", w="w1", b="b1")
                model += relu(input=linear1.output, output="hidden")
                model += linear2(input="hidden", w="w2", b="b2")
                model += softmax(input=linear2.output, output="output")
            
            elif self._model_type == "cnn":
                # CNN model - falls back to standard implementation due to complexity
                print(f"CNN model creation with {self._backend_name} failed, using fallback...")
                return self._fallback.train(X, y, epochs, batch_size, learning_rate)
            
            else:
                # Unknown model type, use fallback
                print(f"Unknown model type: {self._model_type}, using fallback...")
                return self._fallback.train(X, y, epochs, batch_size, learning_rate)
            
            # Create training model
            train_model = TrainModel(model)
            
            # Add loss function
            train_model.add_loss(
                CrossEntropy(),
                input="output",
                target="target",
                reduce_steps=[Mean()]
            )
            
            # Compile model
            input_shape = (batch_size, 784)  # All models use flattened input
            
            self._pm = ml.compile(
                model=train_model,
                backend=self._backend,
                shapes={
                    "input": input_shape,
                    "target": (batch_size, 10)
                }
            )
            
            # Initialize parameters
            if self._model_type == "logistic":
                self._params = {
                    "w": self._backend.numpy_to_tensor(
                        self._r.normal(0, 0.01, (784, 10)).astype(np.float32)
                    ),
                    "b": self._backend.numpy_to_tensor(
                        np.zeros(10, dtype=np.float32)
                    )
                }
            elif self._model_type == "mlp":
                self._params = {
                    "w1": self._backend.numpy_to_tensor(
                        self._r.normal(0, np.sqrt(2.0/784), (784, 128)).astype(np.float32)
                    ),
                    "b1": self._backend.numpy_to_tensor(
                        np.zeros(128, dtype=np.float32)
                    ),
                    "w2": self._backend.numpy_to_tensor(
                        self._r.normal(0, np.sqrt(2.0/128), (128, 10)).astype(np.float32)
                    ),
                    "b2": self._backend.numpy_to_tensor(
                        np.zeros(10, dtype=np.float32)
                    )
                }
            
            # Training
            n_samples = len(X)
            n_batches = n_samples // batch_size
            
            history = []
            
            for epoch in range(epochs):
                total_loss = 0
                
                # Shuffle data
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    batch_x = X_shuffled[start_idx:end_idx]
                    batch_y = y_shuffled[start_idx:end_idx]
                    
                    # Forward pass and gradient calculation
                    outputs, gradients = self._pm.evaluate(
                        self._params,
                        inputs={
                            "input": self._backend.numpy_to_tensor(batch_x),
                            "target": self._backend.numpy_to_tensor(batch_y)
                        },
                        output_gradients=True
                    )
                    
                    # Update parameters with SGD
                    for param_name, grad_value in gradients.items():
                        if param_name in self._params:
                            # Convert tensor to numpy, update, then convert back to tensor
                            grad_numpy = self._backend.tensor_to_numpy(grad_value)
                            param_numpy = self._backend.tensor_to_numpy(self._params[param_name])
                            param_numpy -= learning_rate * grad_numpy
                            self._params[param_name] = self._backend.numpy_to_tensor(param_numpy)
                    
                    # Accumulate batch loss
                    batch_loss = self._backend.tensor_to_numpy(outputs["final_cost"])
                    total_loss += batch_loss
                    
                    if i % 10 == 0:
                        print(f"  Epoch {epoch+1}, Batch {i}/{n_batches}, Loss: {batch_loss:.4f}")
                
                avg_loss = total_loss / n_batches
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
                history.append((avg_loss, 0))
            
            train_time = time.time() - start_time
            print(f"Training time: {train_time:.2f} seconds")
            
            self.trained = True
            return history, train_time
        
        except Exception as e:
            print(f"{self._backend_name} {self._model_type} training error: {e}")
            print("Switching to fallback model...")
            # Use fallback model
            history, time_taken = self._fallback.train(X, y, epochs, batch_size, learning_rate)
            self.trained = True  # Fallback model trained
            return history, time_taken
    
    def predict(self, X):
        """Make predictions with Mithril model"""
        if not self.trained:
            return np.zeros((len(X), 10))
        
        try:
            if self._pm is None or self._params is None:
                # Use fallback model
                return self._fallback.predict(X)
                
            # Make prediction
            outputs = self._pm.evaluate(
                self._params,
                inputs={"input": self._backend.numpy_to_tensor(X)}
            )
            
            # Convert output to NumPy array
            output_numpy = self._backend.tensor_to_numpy(outputs["output"])
            return output_numpy
        except Exception as e:
            print(f"{self._backend_name} {self._model_type} prediction error: {e}")
            # Use fallback model
            return self._fallback.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate on test set"""
        if not self.trained:
            return 0
            
        try:
            if self._pm is None or self._params is None:
                # Use fallback model
                return self._fallback.evaluate(X, y)
                
            # Evaluate in small batches
            batch_size = 100
            n_samples = len(X)
            n_batches = n_samples // batch_size
            
            correct = 0
            total = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_x = X[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]
                
                probs = self.predict(batch_x)
                
                pred_labels = np.argmax(probs, axis=1)
                true_labels = np.argmax(batch_y, axis=1)
                correct += np.sum(pred_labels == true_labels)
                total += len(batch_x)
            
            accuracy = correct / total
            return accuracy
        except Exception as e:
            print(f"{self._backend_name} {self._model_type} evaluation error: {e}")
            # Use fallback model
            return self._fallback.evaluate(X, y)


# Trained models
mdls = {
    "Logistic Regression": LogisticRegression(),
    "MLP": MLP(),
    "Simple CNN": SimpleCNN()
}

# Temporarily enable only NumPy Backend - slow loading problem
mdls["NumPy Backend Logistic Regression"] = _MithrilAdapter("NumPy Backend", bkends["NumPy Backend"], "logistic")

# Training parameters - shorter training time
train_params = {
    "Logistic Regression": {"epochs": 3, "batch_size": 64, "learning_rate": 0.1, "subset_size": 5000},
    "MLP": {"epochs": 3, "batch_size": 64, "learning_rate": 0.01, "subset_size": 5000},
    "Simple CNN": {"epochs": 2, "batch_size": 32, "learning_rate": 0.005, "subset_size": 3000}
}

# Train models
print("Training models...")
results = {}

# Train each model with its own parameters
for name, model in mdls.items():
    # Determine model type
    if "Logistic" in name:
        model_type = "Logistic Regression"
    elif "MLP" in name:
        model_type = "MLP"
    elif "CNN" in name:
        model_type = "Simple CNN"
    else:
        model_type = "Logistic Regression"  # Default
    
    params = train_params[model_type]
    subset_size = params["subset_size"]
    
    print(f"\nTraining model {name}...")
    
    # All models use flattened input
    X_train = x_train_flatten[:subset_size]
    
    history, time_taken = model.train(
        X_train, 
        y_train[:subset_size],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"]
    )
    
    # Evaluate on test data
    X_test = x_test_flatten[:1000]
    
    accuracy = model.evaluate(X_test, y_test[:1000])
    results[name] = {
        "history": history,
        "time": time_taken,
        "accuracy": accuracy
    }
    print(f"{name} test accuracy: {accuracy:.4f}")


# Image processing and prediction
def _proc_img(img, m_name):
    """Process image and predict using the specified model"""
    if img is None or m_name not in mdls:
        return None, None
    
    # Process image
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA
            img = img[:, :, 0]  # Take only first channel
        elif img.shape[2] == 3:  # RGB
            img = np.mean(img, axis=2).astype(np.uint8)  # Convert to grayscale
    
    # Resize to 28x28
    if img.shape != (28, 28):
        p_img = Image.fromarray(img.astype('uint8'))
        p_img = p_img.resize((28, 28), Image.BILINEAR)
        img = np.array(p_img)
    
    # Normalize to match training data format
    if np.mean(img) > 127:
        img = 255 - img  # Invert if background is light
    
    # Save processed image for visualization
    p_img = img.copy()
    
    # Normalize to 0-1 range
    img_norm = p_img.astype(np.float32) / 255.0
    
    # Create flattened shape (common for all models)
    img_flat = img_norm.reshape(1, 784)
    
    # Predict with selected model
    mdl = mdls[m_name]
    probs = mdl.predict(img_flat)[0]
    
    # Create results
    fig = _vis_pred(p_img, probs, m_name)
    
    # Return classification results
    return {str(i): float(probs[i]) for i in range(10)}, fig

def _vis_pred(img, probs, m_name):
    """Visualize prediction results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Input Image")
    ax1.axis('off')
    
    # Show probabilities
    pred_digit = np.argmax(probs)
    conf = probs[pred_digit]
    
    # Color palette
    clrs = ['#3498db'] * 10
    clrs[pred_digit] = '#2ecc71'  # Green for predicted digit
    
    bars = ax2.bar(range(10), probs, color=clrs)
    
    ax2.set_xticks(range(10))
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Probability')
    ax2.set_title(f"Prediction: {pred_digit} (Confidence: {conf:.2f})\nModel: {m_name}")
    
    # Add numeric values above bars
    for i, v in enumerate(probs):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=8, 
                 color='darkgreen' if i == pred_digit else 'darkblue')
    
    plt.tight_layout()
    return fig

def _comp_mdls(img):
    """Compare all models on the same input and generate real-time performance metrics"""
    if img is None:
        return None, None, None
    
    # Process image
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA
            img = img[:, :, 0]  # Take only first channel
        elif img.shape[2] == 3:  # RGB
            img = np.mean(img, axis=2).astype(np.uint8)  # Convert to grayscale
    
    # Resize to 28x28
    if img.shape != (28, 28):
        p_img = Image.fromarray(img.astype('uint8'))
        p_img = p_img.resize((28, 28), Image.BILINEAR)
        img = np.array(p_img)
    
    # Normalize to match training data format
    if np.mean(img) > 127:
        img = 255 - img  # Invert if background is light
    
    # Save processed image for visualization
    p_img = img.copy()
    
    # Normalize to 0-1 range
    img_norm = p_img.astype(np.float32) / 255.0
    
    # Create flattened shape (for all models)
    img_flat = img_norm.reshape(1, 784)
    
    # Real-time results for the current image
    real_time_results = {}
    
    # Collect predictions from all models
    for name, model in mdls.items():
        start_time = time.time()
        probs = model.predict(img_flat)[0]
        pred_time = time.time() - start_time
        
        pred_digit = np.argmax(probs)
        confidence = probs[pred_digit]
        
        real_time_results[name] = {
            "prediction": pred_digit,
            "confidence": confidence,
            "time": pred_time
        }
    
    # Select base models for comparison chart
    comp_mdls_list = ["Logistic Regression", "MLP", "Simple CNN"]
    
    # Create 1x3 grid for prediction visualization
    fig, axes = plt.subplots(1, len(comp_mdls_list), figsize=(15, 5))
    
    # Input image and get prediction from each model
    for i, m_name in enumerate(comp_mdls_list):
        probs = mdls[m_name].predict(img_flat)[0]
        pred_digit = np.argmax(probs)
        conf = probs[pred_digit]
        
        # Color palette
        clrs = ['#3498db'] * 10
        clrs[pred_digit] = '#2ecc71'  # Green for predicted digit
        
        # Subvisualization
        axes[i].set_title(f"{m_name}\nPrediction: {pred_digit} (Confidence: {conf:.2f})")
        bars = axes[i].bar(range(10), probs, color=clrs)
        axes[i].set_ylim(0, 1)
        axes[i].set_xticks(range(10))
        
        # Add numeric values above bars
        for j, v in enumerate(probs):
            axes[i].text(j, v + 0.02, f"{v:.2f}" if j == pred_digit else "", 
                    ha='center', fontsize=8, color='darkgreen')
    
    plt.suptitle(f"Model Predictions Comparison", fontsize=16)
    plt.tight_layout()
    
    # Create performance comparison visualization based on real-time results
    perf_fig = _create_realtime_perf_chart(real_time_results)
    
    # Create table with real-time results
    comp_table = _create_realtime_table(real_time_results)
    
    return fig, perf_fig, comp_table

def _create_realtime_perf_chart(real_time_results):
    """Create performance chart based on real-time prediction results"""
    # Group model types
    m_groups = {
        "Logistic Regression": [],
        "MLP": [],
        "CNN": []
    }
    
    for name in real_time_results.keys():
        if "Logistic" in name:
            m_groups["Logistic Regression"].append(name)
        elif "MLP" in name:
            m_groups["MLP"].append(name)
        elif "CNN" in name or "Simple CNN" in name:
            m_groups["CNN"].append(name)
    
    # Create 3 rows (for each model type) and 2 columns (confidence and time)
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    for i, (m_type, m_names) in enumerate(m_groups.items()):
        if not m_names:
            continue
            
        # Get confidence and times for all backends of this model type
        confidence_vals = [real_time_results[name]["confidence"] * 100 for name in m_names]
        times = [real_time_results[name]["time"] * 1000 for name in m_names]  # Convert to milliseconds
        
        # Confidence graph
        ax1 = fig.add_subplot(gs[i, 0])
        
        # Color palette
        clrs = plt.cm.viridis(np.linspace(0, 0.8, len(m_names)))
        
        bars1 = ax1.bar(m_names, confidence_vals, color=clrs)
        ax1.set_ylabel("Confidence (%)")
        ax1.set_title(f"{m_type} Models Confidence Comparison", fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate X labels
        ax1.set_xticklabels(m_names, rotation=45, ha='right')
        
        # Add numeric values above bars
        for j, v in enumerate(confidence_vals):
            ax1.text(j, v + 1, f"{v:.1f}%", ha='center', fontsize=9)
        
        # Prediction time graph
        ax2 = fig.add_subplot(gs[i, 1])
        bars2 = ax2.bar(m_names, times, color=clrs)
        ax2.set_ylabel("Time (milliseconds)")
        ax2.set_title(f"{m_type} Models Prediction Time Comparison", fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate X labels
        ax2.set_xticklabels(m_names, rotation=45, ha='right')
        
        # Add numeric values above bars
        for j, v in enumerate(times):
            ax2.text(j, v + 0.5, f"{v:.2f}ms", ha='center', fontsize=9)
    
    plt.suptitle("Real-time Model Performance Comparison", fontsize=16, y=0.98)
    plt.tight_layout()
    return fig

def _create_realtime_table(real_time_results):
    """Create comparison table with real-time data"""
    # Collect all model results
    comp_data = []
    
    # Calculate relative values
    all_conf = [real_time_results[name]["confidence"] * 100 for name in real_time_results.keys()]
    all_times = [real_time_results[name]["time"] * 1000 for name in real_time_results.keys()]  # milliseconds
    max_conf = max(all_conf) if all_conf else 0
    min_time = min([t for t in all_times if t > 0]) if all_times else 0
    
    # Get global test accuracy from original results for reference
    global_accs = {name: results[name]["accuracy"] * 100 if name in results else 0 for name in real_time_results.keys()}
    
    for name in real_time_results.keys():
        # Determine model type
        if "Logistic" in name:
            m_type = "Logistic Regression"
        elif "MLP" in name:
            m_type = "MLP"
        elif "CNN" in name:
            m_type = "CNN"
        else:
            m_type = "Other"
        
        # Determine backend
        if "NumPy" in name:
            backend = "NumPy"
        elif "JAX" in name:
            backend = "JAX"
        elif "PyTorch" in name:
            backend = "PyTorch"
        elif "TensorFlow" in name:
            backend = "TensorFlow"
        else:
            backend = "NumPy"
        
        # Get real-time data
        prediction = real_time_results[name]["prediction"]
        conf = real_time_results[name]["confidence"] * 100
        pred_time = real_time_results[name]["time"] * 1000  # Convert to milliseconds
        
        # Calculate relative values
        rel_conf = (conf / max_conf * 100) if max_conf > 0 else 0
        rel_speed = (min_time / pred_time * 100) if pred_time > 0 else 0
        
        # Get test accuracy from global results
        test_acc = global_accs.get(name, 0)
        
        comp_data.append([
            name,
            m_type,
            backend,
            str(int(prediction)),
            f"{conf:.2f}",
            f"{pred_time:.2f}",
            f"{test_acc:.2f}",
            f"{rel_conf:.1f}"
        ])
    
    # Sort by highest confidence
    comp_data.sort(key=lambda x: float(x[4]), reverse=True)
    return comp_data

def _create_app():
    # Gradio interface - modified to have only Model Comparison tab with all features
    with gr.Blocks(title="MNIST Model Comparison") as demo:
        gr.Markdown("# MNIST Digit Recognition - Model Comparison")
        
        # Create example images list for reuse
        ex_imgs = []
        for i in range(10):
            idx = np.where(y_test_raw == i)[0][0]
            img = x_test_raw[idx]
            ex_imgs.append(img)
        
        with gr.Tab("Model Comparison"):
            gr.Markdown("## Models Comparison")
            
            with gr.Row():
                with gr.Column():
                    # Select image for comparison
                    comp_img = gr.Image(
                        type="numpy", 
                        label="Select Digit for Comparison", 
                        image_mode="L", 
                        sources=["clipboard"],
                        interactive=True,
                        height=256,
                        width=256
                    )
                    
                    # Example inputs
                    gr.Examples(
                        examples=[[img] for img in ex_imgs],
                        inputs=[comp_img],
                        label="Test Examples",
                        examples_per_page=5
                    )
                    
                    comp_btn = gr.Button("Compare Models", variant="primary")
                
                with gr.Column():
                    comp_plot = gr.Plot(label="Model Predictions Comparison")
            
            gr.Markdown("## Real-time Performance Metrics")
            
            with gr.Row():
                perf_plot = gr.Plot(label="Real-time Performance Charts")
                
            # Real-time comparison table
            with gr.Row():
                comp_table = gr.DataFrame(
                    headers=["Model Name", "Model Type", "Backend", "Prediction", "Confidence (%)", 
                            "Prediction Time (ms)", "Test Accuracy (%)", "Relative Confidence (%)"],
                    label="Real-time Model Comparison"
                )
            
            # Best Model info will be displayed based on real-time results
            with gr.Row():
                best_model_info = gr.Markdown("### Click 'Compare Models' to see results")
                
            # Predictions comparison now also updates performance charts and table
            comp_btn.click(
                _comp_mdls,
                inputs=[comp_img],
                outputs=[comp_plot, perf_plot, comp_table]
            )
    
    return demo

# Launch the app
if __name__ == "__main__":
    _app = _create_app()
    _app.launch() 