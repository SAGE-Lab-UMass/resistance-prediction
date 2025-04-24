import numpy as np
from sklearn.metrics import mean_squared_error
from utility_files.data_preprocessing import seed_everything

class FusedRidge:
    def __init__(self, initial_coef, optimizer='vanilla', alpha=1.0, lambda_fuse=1.0, scale_param=10.0,
                 learning_rate=0.01, max_iter=1000, clip_value=1.0, decay_type=None, beta=0.1, momentum=0.9,
                 early_stopping=False, tolerance=1e-10, n_iter_no_change=50, dimension_threshold=100):

        seed_everything(42)
        optimizers = {
            'vanilla': self._vanilla_sgd,
            'enhanced': self._enhanced_sgd,
            'momentum': self._momentum_sgd,
            'nesterov': self._nesterov_sgd
        }
        if optimizer not in optimizers.keys():
            raise ValueError(f"{optimizer} is not recognized. Use one of the {list(optimizers.keys())}")
        else:
            self.optimizer = optimizers[optimizer]

        self.alpha = alpha
        self.lambda_fuse = lambda_fuse
        self.scale_param = scale_param
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.clip_value = clip_value
        self.decay_type = decay_type
        self.beta = beta
        self.momentum = momentum
        self.early_stopping = early_stopping
        self.tolerance = tolerance
        self.n_iter_no_change = n_iter_no_change
        self.dimension_threshold = dimension_threshold  # Threshold for closed-form vs. iterative
        self.distance_matrix = None
        self.coef_ = initial_coef
        self.iteration = 0
        self.best_loss = np.inf
        self.iter_no_change = 0
        self.verbose = True
        self.converged = False

        self.history = {'loss': [], 'coefficients': []}
        self.loss_function_calls = 0
        self.velocity = np.zeros_like(initial_coef)  # For momentum and Nesterov

    def set_distance_matrix(self, distance_matrix):
        self.distance_matrix = distance_matrix

    def _update_learning_rate(self, iteration):
        if self.decay_type == 'harmonic':
            return self.learning_rate / (1 + self.beta * iteration)
        elif self.decay_type == 'sqrt':
            return self.learning_rate / np.sqrt(1 + self.beta * iteration)
        else:
            return self.learning_rate

    #closed form solution
    def fit(self, X, y):
        n_features = X.shape[1]
        if n_features <= self.dimension_threshold:
            self._fit_closed_form(X, y)
        else:
            self._fit_iterative(X, y)




    def _fit_closed_form(self, X, y):
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not set. Use set_distance_matrix() to set it.")
        
        # Construct fusion penalty matrix L
        L = self._compute_fusion_matrix()
        
        # Compute closed-form solution
        A = X.T @ X + self.alpha * np.eye(X.shape[1]) + self.lambda_fuse * L
        b = X.T @ y
        self.coef_ = np.linalg.solve(A, b)
        if self.verbose:
            print("Closed-form solution applied.")

    def _fit_iterative(self, X, y,verbose=False):
        #original fit method
        self.verbose = verbose
        print("Iterative optimization applied.")
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not set. Use set_distance_matrix() to set it.")

        self.X, self.y = X, y
        self.history = {'loss': [], 'coefficients': []}
        self.iteration = 0

        self.optimizer()

        if not self.converged:
            self.converged = False  # If max_iter reached, mark as not converged

        return self
    def _compute_fusion_matrix(self):
        # Construct L from the distance matrix
        n = self.distance_matrix.shape[0]
        L = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    L[i, i] += self.distance_matrix[i, j]
                    L[i, j] -= self.distance_matrix[i, j]
        return L



    def _vanilla_sgd(self):
        for iteration in range(self.max_iter):
            subgrad = self._compute_subgradient(self.coef_, self.X, self.y)
            self.coef_ -= self.learning_rate * subgrad
            current_loss = self._custom_loss(self.coef_, self.X, self.y)
            self.history['loss'].append(current_loss)
            self.history['coefficients'].append(self.coef_.copy())

            if self.early_stopping and self._check_early_stopping(current_loss):
                print(f"Converged after {iteration} iterations with loss: {current_loss}")
                self.converged = True  # Converged successfully
                self.final_loss = current_loss 
                return

            self.iteration += 1
        if self.iteration == self.max_iter:
            self.converged = False  # If max_iter reached without early stopping
            self.final_loss = current_loss  
    def _enhanced_sgd(self):
        for iteration in range(self.max_iter):
            current_learning_rate = self._update_learning_rate(iteration)
            subgrad = self._compute_subgradient(self.coef_, self.X, self.y)
            subgrad = np.clip(subgrad, -self.clip_value, self.clip_value)
            self.coef_ -= current_learning_rate * subgrad
            current_loss = self._custom_loss(self.coef_, self.X, self.y)
            self.history['loss'].append(current_loss)
            self.history['coefficients'].append(self.coef_.copy())

            if self.early_stopping and self._check_early_stopping(current_loss):
                print(f"Converged after {iteration} iterations with loss: {current_loss}")
                self.final_loss = current_loss 
                return  # Stop early if converged

            self.iteration += 1

        # If we exit the loop without early stopping
        print(f"Stopped after reaching max_iter ({self.max_iter}). Final loss: {current_loss}")
        self.converged = False
        self.final_loss = current_loss 

    def _momentum_sgd(self):
        for iteration in range(self.max_iter):
            current_learning_rate = self._update_learning_rate(iteration)
            subgrad = self._compute_subgradient(self.coef_, self.X, self.y)
            subgrad = np.clip(subgrad, -self.clip_value, self.clip_value)
            self.velocity = self.momentum * self.velocity + current_learning_rate * subgrad
            self.coef_ -= self.velocity
            current_loss = self._custom_loss(self.coef_, self.X, self.y)
            self.history['loss'].append(current_loss)
            self.history['coefficients'].append(self.coef_.copy())

            if self.early_stopping and self._check_early_stopping(current_loss):
                print(f"Converged after {iteration} iterations with loss: {current_loss}")
                self.final_loss = current_loss 
                return

            self.iteration += 1

        if self.iteration == self.max_iter:
            self.converged = False  # If max_iter reached without early stopping
            self.final_loss = current_loss 

    def _nesterov_sgd(self):
        for iteration in range(self.max_iter):
            current_learning_rate = self._update_learning_rate(iteration)
            lookahead_coef = self.coef_ - self.momentum * self.velocity
            subgrad = self._compute_subgradient(lookahead_coef, self.X, self.y)
            subgrad = np.clip(subgrad, -self.clip_value, self.clip_value)
            self.velocity = self.momentum * self.velocity + current_learning_rate * subgrad
            self.coef_ -= self.velocity
            current_loss = self._custom_loss(self.coef_, self.X, self.y)
            self.history['loss'].append(current_loss)
            self.history['coefficients'].append(self.coef_.copy())

            if self.early_stopping and self._check_early_stopping(current_loss):
                print(f"Converged after {iteration} iterations with loss: {current_loss}")
                self.final_loss = current_loss 
                return

            self.iteration += 1

        if self.iteration == self.max_iter:
            self.converged = False  # If max_iter reached without early stopping
            self.final_loss = current_loss 

    def _check_early_stopping(self, current_loss):
        if current_loss < self.best_loss - self.tolerance:
            self.best_loss = current_loss
            self.iter_no_change = 0
        else:
            self.iter_no_change += 1
            if self.iter_no_change >= self.n_iter_no_change:
                print("Early stopping: stopping training")
                self.converged = True  # Set convergence to True when early stopping occurs
                return True
        return False

    def predict(self, X):
        return np.dot(X, self.coef_)

    def _custom_loss(self, coef, X, y):
        self.loss_function_calls += 1
        y_pred = np.dot(X, coef)
        squared_loss = 0.5 * np.sum((y - y_pred) ** 2)
        l2_penalty = self.alpha * np.sum(coef ** 2)
        fusion_penalty = self._fusion_penalty(coef)
        return squared_loss + l2_penalty + fusion_penalty

    def _compute_subgradient(self, coef, X, y):
        y_pred = np.dot(X, coef)
        residual = y_pred - y
        subgrad_squared_loss = np.dot(X.T, residual)
        subgrad_l2 = 2 * self.alpha * coef  # Subgradient of L2 penalty
        subgrad_fusion = self._fusion_penalty_subgradient(coef)
        subgrad = subgrad_squared_loss + subgrad_l2 + subgrad_fusion
        return subgrad

    def _fusion_penalty(self, coef):
        coef_diff = (coef[:, None] - coef) ** 2
        thresholded_distances = np.full(self.distance_matrix.shape, np.inf)
        indices_i, indices_j = np.triu_indices_from(self.distance_matrix, k=1)
        thresholded_distances[indices_i, indices_j] = self.distance_matrix[indices_i, indices_j]
        weights = np.exp(-thresholded_distances / self.scale_param)
        weights[thresholded_distances == np.inf] = 0
        fusion_term = np.sum(weights * coef_diff)
        return self.lambda_fuse * fusion_term

    def _fusion_penalty_subgradient(self, coef):
        coef_diff = (coef[:, None] - coef) ** 2
        thresholded_distances = np.full(self.distance_matrix.shape, np.inf)
        indices_i, indices_j = np.triu_indices_from(self.distance_matrix, k=1)
        thresholded_distances[indices_i, indices_j] = self.distance_matrix[indices_i, indices_j]
        weights = np.exp(-thresholded_distances / self.scale_param)
        weights[thresholded_distances == np.inf] = 0
        n = len(coef)
        subgradient = np.zeros(n)
        for i in range(n):
            subgradient[i] = 2 * np.sum(weights[i, :] * (coef[i] - coef))
        return self.lambda_fuse * subgradient

    def calculate_mse(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)