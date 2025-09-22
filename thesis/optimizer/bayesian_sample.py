# # import numpy as np
# # import matplotlib.pyplot as plt
# # from matplotlib.animation import FuncAnimation
# # from scipy.stats import norm
# # from scipy.optimize import minimize
# # import warnings
# #
# # warnings.filterwarnings('ignore')
# #
# # # Set random seed for reproducibility
# # np.random.seed(42)
# #
# #
# # class BayesianOptimizationAnimator:
# #     def __init__(self):
# #         # Define the same complex power loss function
# #         self.true_func = self._complex_power_loss_function
# #
# #         # Search space
# #         self.bounds = (15, 35)
# #         self.X_plot = np.linspace(self.bounds[0], self.bounds[1], 300)
# #         self.y_plot_true = self.true_func(self.X_plot)
# #
# #         # Find all local minima for visualization
# #         self.local_minima = self._find_local_minima()
# #         print(f"Found {len(self.local_minima)} minima: {[(m['x'], m['value'], m['type']) for m in self.local_minima]}")
# #
# #         # Ensure we have at least one minimum
# #         if len(self.local_minima) == 0:
# #             x_opt = 25.0
# #             self.local_minima = [{
# #                 'x': x_opt,
# #                 'value': self.true_func(x_opt),
# #                 'type': 'global'
# #             }]
# #
# #         self.global_minimum = self.local_minima[0]
# #         # Set global optimum to zero
# #         self.global_minimum['value'] = 0.0
# #
# #         # BO Parameters - Optimized for healthy exploration
# #         self.max_iterations = 25
# #         self.noise_level = 0.1
# #         self.length_scale = 2.5  # Balanced for exploration/exploitation
# #         self.signal_variance = 100.0
# #         self.exploration_factor = 2.0  # Higher for more exploration
# #
# #         # Adaptive parameters for healthy exploration
# #         self.uncertainty_target = 15.0  # Target uncertainty level
# #         self.min_distance_between_points = 1.0  # Prevent clustering
# #         self.exploration_boost_iterations = [5, 10, 15]  # Boost exploration at these iterations
# #
# #         # Data storage
# #         self.iteration = 0
# #         self.X_observed = []
# #         self.y_observed = []
# #         self.acquisition_history = []
# #         self.uncertainty_history = []
# #         self.exploration_metrics = []
# #         self.next_points = []
# #         self.gp_means = []
# #         self.gp_stds = []
# #
# #         # Initialize with strategic starting points for better coverage
# #         initial_points = self._get_strategic_initial_points()
# #         for x in initial_points:
# #             y = self.true_func(x) + np.random.normal(0, self.noise_level)
# #             self.X_observed.append(x)
# #             self.y_observed.append(y)
# #
# #         self._update_gp_and_metrics()
# #
# #     def _complex_power_loss_function(self, x):
# #         """Complex power loss function with multiple local minima (in Watts)"""
# #         x = np.asarray(x)
# #
# #         # Base quadratic with global minimum at x=25 (scaled for power loss in Watts)
# #         base = 150 + 0.8 * (x - 25) ** 2
# #
# #         # Add local minima that are HIGHER than the global minimum
# #         local_bumps = (25 * np.exp(-3.0 * (x - 17) ** 2) +
# #                        30 * np.exp(-4.0 * (x - 20.5) ** 2) +
# #                        35 * np.exp(-3.5 * (x - 32) ** 2))
# #
# #         # Add some oscillations (thermal and mechanical variations)
# #         oscillations = 5 * np.sin(1.2 * np.pi * x / 4) + 3 * np.cos(0.6 * np.pi * x / 3)
# #
# #         # Subtract local bumps to create local minima
# #         result = base - local_bumps + oscillations
# #
# #         return result
# #
# #     def _find_local_minima(self):
# #         """Find local minima in the function for visualization"""
# #         x_fine = np.linspace(self.bounds[0], self.bounds[1], 2000)
# #         y_fine = self.true_func(x_fine)
# #
# #         local_minima = []
# #         window_size = 20
# #
# #         for i in range(window_size, len(y_fine) - window_size):
# #             local_window = y_fine[i - window_size:i + window_size + 1]
# #             if y_fine[i] == np.min(local_window):
# #                 surrounding_mean = np.mean(y_fine[i - window_size:i + window_size + 1])
# #                 if y_fine[i] < surrounding_mean - 2.0:
# #                     # Avoid duplicates
# #                     too_close = False
# #                     for existing in local_minima:
# #                         if abs(x_fine[i] - existing['x']) < 1.0:
# #                             too_close = True
# #                             break
# #
# #                     if not too_close:
# #                         local_minima.append({
# #                             'x': x_fine[i],
# #                             'value': y_fine[i],
# #                             'type': 'local'
# #                         })
# #
# #         # Add manual minima if none found
# #         if len(local_minima) == 0:
# #             known_minima = [17.0, 20.5, 25.0, 32.0]
# #             for x in known_minima:
# #                 local_minima.append({
# #                     'x': x,
# #                     'value': self.true_func(x),
# #                     'type': 'local'
# #                 })
# #
# #         # Sort by value and mark global minimum
# #         local_minima.sort(key=lambda x: x['value'])
# #         if local_minima:
# #             local_minima[0]['type'] = 'global'
# #             for i in range(1, len(local_minima)):
# #                 local_minima[i]['type'] = 'local'
# #
# #         return local_minima
# #
# #     def _get_strategic_initial_points(self):
# #         """Get strategically placed initial points for better exploration"""
# #         # Start with points that provide good coverage but avoid clustering
# #         initial_points = [
# #             16.5,  # Left region
# #             22.0,  # Center-left
# #             28.0,  # Center-right
# #             33.0  # Right region
# #         ]
# #         return initial_points
# #
# #     def _rbf_kernel(self, X1, X2):
# #         """RBF (Gaussian) kernel for GP"""
# #         X1 = np.array(X1).reshape(-1, 1)
# #         X2 = np.array(X2).reshape(-1, 1)
# #
# #         sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
# #         return self.signal_variance * np.exp(-0.5 * sqdist / (self.length_scale ** 2))
# #
# #     def _gp_predict(self, X_test):
# #         """Gaussian Process prediction with healthy uncertainty"""
# #         if len(self.X_observed) == 0:
# #             mean = np.full(len(X_test), 150.0)  # Prior mean
# #             std = np.full(len(X_test), np.sqrt(self.signal_variance))
# #             return mean, std
# #
# #         X_obs = np.array(self.X_observed)
# #         y_obs = np.array(self.y_observed)
# #
# #         # Compute kernel matrices
# #         K = self._rbf_kernel(X_obs, X_obs)
# #         K += self.noise_level ** 2 * np.eye(len(X_obs))  # Add noise
# #
# #         K_s = self._rbf_kernel(X_obs, X_test)
# #         K_ss = self._rbf_kernel(X_test, X_test)
# #
# #         # Compute inverse
# #         try:
# #             K_inv = np.linalg.inv(K)
# #         except np.linalg.LinAlgError:
# #             K_inv = np.linalg.pinv(K)
# #
# #         # Predict mean and variance
# #         mu = K_s.T.dot(K_inv).dot(y_obs)
# #         cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
# #         std = np.sqrt(np.maximum(np.diag(cov), 1e-6))  # Ensure positive
# #
# #         return mu, std
# #
# #     def _expected_improvement(self, X_test):
# #         """Expected Improvement acquisition function with exploration boost"""
# #         mu, sigma = self._gp_predict(X_test)
# #
# #         if len(self.y_observed) == 0:
# #             return sigma  # Pure exploration initially
# #
# #         f_best = np.min(self.y_observed)
# #
# #         # Add exploration boost at certain iterations
# #         exploration_boost = 1.0
# #         if self.iteration in self.exploration_boost_iterations:
# #             exploration_boost = 2.5
# #
# #         xi = self.exploration_factor * exploration_boost
# #
# #         with np.errstate(divide='warn'):
# #             imp = f_best - mu - xi
# #             Z = imp / sigma
# #             ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
# #             ei[sigma == 0.0] = 0.0
# #
# #         return ei
# #
# #     def _upper_confidence_bound(self, X_test, beta=2.0):
# #         """Upper Confidence Bound acquisition function"""
# #         mu, sigma = self._gp_predict(X_test)
# #         return mu - beta * sigma  # Minus because we're minimizing
# #
# #     def _calculate_exploration_metrics(self):
# #         """Calculate metrics for healthy exploration"""
# #         if len(self.X_observed) < 2:
# #             return {
# #                 'coverage': 0.0,
# #                 'avg_uncertainty': 100.0,
# #                 'min_distance': 20.0,
# #                 'exploration_efficiency': 0.0
# #             }
# #
# #         # Coverage: how well we span the search space
# #         x_range = self.bounds[1] - self.bounds[0]
# #         observed_range = max(self.X_observed) - min(self.X_observed)
# #         coverage = observed_range / x_range
# #
# #         # Average uncertainty in unexplored regions
# #         X_test = np.linspace(self.bounds[0], self.bounds[1], 100)
# #         _, uncertainty = self._gp_predict(X_test)
# #         avg_uncertainty = np.mean(uncertainty)
# #
# #         # Minimum distance between observed points (diversity measure)
# #         distances = []
# #         for i in range(len(self.X_observed)):
# #             for j in range(i + 1, len(self.X_observed)):
# #                 distances.append(abs(self.X_observed[i] - self.X_observed[j]))
# #         min_distance = min(distances) if distances else 20.0
# #
# #         # Exploration efficiency (coverage vs iterations)
# #         exploration_efficiency = coverage / max(1, len(self.X_observed))
# #
# #         return {
# #             'coverage': coverage,
# #             'avg_uncertainty': avg_uncertainty,
# #             'min_distance': min_distance,
# #             'exploration_efficiency': exploration_efficiency
# #         }
# #
# #     def _update_gp_and_metrics(self):
# #         """Update GP model and exploration metrics"""
# #         # Update GP predictions
# #         mu, sigma = self._gp_predict(self.X_plot)
# #         self.gp_means.append(mu.copy())
# #         self.gp_stds.append(sigma.copy())
# #
# #         # Calculate exploration metrics
# #         metrics = self._calculate_exploration_metrics()
# #         self.exploration_metrics.append(metrics)
# #
# #         # Store uncertainty for tracking
# #         self.uncertainty_history.append(metrics['avg_uncertainty'])
# #
# #     def _select_next_point(self):
# #         """Select next point with healthy exploration strategy"""
# #         if self.iteration >= self.max_iterations:
# #             return None
# #
# #         # Use Expected Improvement with exploration considerations
# #         X_candidates = np.linspace(self.bounds[0], self.bounds[1], 1000)
# #
# #         # Calculate acquisition function
# #         acquisition = self._expected_improvement(X_candidates)
# #
# #         # Prevent clustering by penalizing points too close to existing ones
# #         for x_obs in self.X_observed:
# #             distances = np.abs(X_candidates - x_obs)
# #             penalty = np.exp(-distances / self.min_distance_between_points)
# #             acquisition *= (1 - 0.8 * penalty)  # Reduce acquisition near existing points
# #
# #         # Select point with highest acquisition value
# #         best_idx = np.argmax(acquisition)
# #         next_x = X_candidates[best_idx]
# #
# #         # Store acquisition function for visualization
# #         self.acquisition_history.append({
# #             'x': X_candidates.copy(),
# #             'values': acquisition.copy(),
# #             'next_point': next_x
# #         })
# #
# #         return next_x
# #
# #     def _iterate(self):
# #         """Perform one iteration of Bayesian Optimization"""
# #         if self.iteration >= self.max_iterations:
# #             return
# #
# #         # Select next point
# #         next_x = self._select_next_point()
# #         if next_x is None:
# #             return
# #
# #         # Evaluate function (with noise)
# #         next_y = self.true_func(next_x) + np.random.normal(0, self.noise_level)
# #
# #         # Add to observations
# #         self.X_observed.append(next_x)
# #         self.y_observed.append(next_y)
# #         self.next_points.append(next_x)
# #
# #         self.iteration += 1
# #         self._update_gp_and_metrics()
# #
# #     def update_data(self, frame):
# #         """Update data for each frame"""
# #         if frame > 0 and self.iteration < self.max_iterations:
# #             self._iterate()
# #
# #     def animate_bo_process(self, frame):
# #         """Animation function for BO process"""
# #         plt.clf()
# #
# #         self.update_data(frame)
# #
# #         if len(self.gp_means) > 0:
# #             # Plot true function
# #             plt.plot(self.X_plot, self.y_plot_true, 'k-', alpha=0.8, label='True Function', linewidth=3)
# #
# #             # Plot GP mean and uncertainty
# #             current_idx = min(frame, len(self.gp_means) - 1)
# #             mu = self.gp_means[current_idx]
# #             sigma = self.gp_stds[current_idx]
# #
# #             plt.plot(self.X_plot, mu, 'b-', alpha=0.8, label='GP Mean', linewidth=2)
# #             plt.fill_between(self.X_plot, mu - 2 * sigma, mu + 2 * sigma,
# #                              alpha=0.3, color='blue', label='95% Confidence')
# #
# #             # Plot observations up to current frame
# #             current_frame_obs = min(frame + 4, len(self.X_observed))  # +4 for initial points
# #             if current_frame_obs > 0:
# #                 X_obs_current = self.X_observed[:current_frame_obs]
# #                 y_obs_current = self.y_observed[:current_frame_obs]
# #
# #                 # Color code observations
# #                 colors = ['red' if i < 4 else 'blue' for i in range(len(X_obs_current))]
# #                 sizes = [120 if i < 4 else 100 for i in range(len(X_obs_current))]
# #
# #                 plt.scatter(X_obs_current, y_obs_current, c=colors, s=sizes,
# #                             zorder=10, edgecolors='black', linewidth=1,
# #                             label='Observations' if frame == 0 else '')
# #
# #             # Show local minima after some iterations
# #             if frame > 2:
# #                 for i, minima in enumerate(self.local_minima):
# #                     if minima['type'] == 'local':
# #                         plt.scatter(minima['x'], minima['value'], c='orange', s=100,
# #                                     marker='o', zorder=15,
# #                                     label=f"Local Min" if i == 1 else "",
# #                                     edgecolors='black')
# #
# #             # Show next point if available
# #             if frame > 0 and frame <= len(self.next_points):
# #                 next_idx = min(frame - 1, len(self.next_points) - 1)
# #                 if next_idx >= 0 and next_idx < len(self.next_points):
# #                     next_x = self.next_points[next_idx]
# #                     next_y = self.true_func(next_x)
# #                     plt.scatter(next_x, next_y, c='green', s=150, marker='*',
# #                                 zorder=20, label='Next Sample',
# #                                 edgecolors='black', linewidth=2)
# #
# #             plt.xlabel('Piston Diameter (mm)', fontsize=12)
# #             plt.ylabel('Power Loss (Watts)', fontsize=12)
# #
# #             # Dynamic title
# #             if frame == 0:
# #                 title = f'Iteration {frame}: Strategic Initial Sampling'
# #             elif frame < 5:
# #                 title = f'Iteration {frame}: Building GP Model with Exploration'
# #             else:
# #                 title = f'Iteration {frame}: Balancing Exploration & Exploitation'
# #
# #             plt.title(title, fontsize=14, fontweight='bold')
# #             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# #             plt.grid(True, alpha=0.3)
# #             plt.tight_layout()
# #
# #     def animate_acquisition(self, frame):
# #         """Animation function for acquisition function"""
# #         plt.clf()
# #
# #         self.update_data(frame)
# #
# #         if len(self.acquisition_history) > 0 and frame > 0:
# #             # Plot true function (top subplot)
# #             plt.subplot(2, 1, 1)
# #             plt.plot(self.X_plot, self.y_plot_true, 'k-', alpha=0.6, linewidth=2)
# #
# #             # Plot observations
# #             current_obs = min(frame + 4, len(self.X_observed))
# #             if current_obs > 0:
# #                 X_obs = self.X_observed[:current_obs]
# #                 y_obs = self.y_observed[:current_obs]
# #                 plt.scatter(X_obs, y_obs, c='red', s=80, zorder=10, alpha=0.8)
# #
# #             # Plot GP mean
# #             if len(self.gp_means) > 0:
# #                 current_idx = min(frame, len(self.gp_means) - 1)
# #                 mu = self.gp_means[current_idx]
# #                 plt.plot(self.X_plot, mu, 'b-', alpha=0.8, linewidth=2, label='GP Mean')
# #
# #             plt.ylabel('Power Loss (Watts)', fontsize=11)
# #             plt.title(f'Iteration {frame}: GP Model and Observations', fontsize=12)
# #             plt.legend()
# #             plt.grid(True, alpha=0.3)
# #
# #             # Plot acquisition function (bottom subplot)
# #             plt.subplot(2, 1, 2)
# #             acq_idx = min(frame - 1, len(self.acquisition_history) - 1)
# #             if acq_idx >= 0:
# #                 acq_data = self.acquisition_history[acq_idx]
# #                 plt.plot(acq_data['x'], acq_data['values'], 'g-', linewidth=2,
# #                          label='Expected Improvement')
# #
# #                 # Highlight next point
# #                 next_x = acq_data['next_point']
# #                 next_acq = np.interp(next_x, acq_data['x'], acq_data['values'])
# #                 plt.scatter(next_x, next_acq, c='green', s=150, marker='*',
# #                             zorder=20, label='Selected Next Point')
# #
# #                 # Show high uncertainty regions
# #                 if len(self.gp_stds) > current_idx:
# #                     sigma = self.gp_stds[current_idx]
# #                     high_uncertainty = sigma > np.percentile(sigma, 75)
# #                     plt.fill_between(self.X_plot, 0, np.max(acq_data['values']) * 0.1,
# #                                      where=high_uncertainty, alpha=0.2, color='yellow',
# #                                      label='High Uncertainty Regions')
# #
# #             plt.xlabel('Piston Diameter (mm)', fontsize=11)
# #             plt.ylabel('Expected Improvement', fontsize=11)
# #             plt.title('Acquisition Function: Guiding Exploration', fontsize=12)
# #             plt.legend()
# #             plt.grid(True, alpha=0.3)
# #
# #             plt.tight_layout()
# #
# #     def animate_exploration_metrics(self, frame):
# #         """Animation function for exploration metrics"""
# #         plt.clf()
# #
# #         self.update_data(frame)
# #
# #         if len(self.exploration_metrics) > 1:
# #             iterations = range(len(self.exploration_metrics))
# #
# #             # Extract metrics
# #             coverage = [m['coverage'] for m in self.exploration_metrics]
# #             uncertainty = [m['avg_uncertainty'] for m in self.exploration_metrics]
# #             min_distance = [m['min_distance'] for m in self.exploration_metrics]
# #
# #             # Plot exploration metrics
# #             plt.subplot(2, 1, 1)
# #             plt.plot(iterations, coverage, 'b-o', linewidth=2, markersize=4, label='Search Space Coverage')
# #             plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Excellent Coverage (80%)')
# #             plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Good Coverage (60%)')
# #             plt.xlabel('Iteration', fontsize=11)
# #             plt.ylabel('Coverage Ratio', fontsize=11)
# #             plt.title(f'Iteration {frame}: Search Space Coverage - Avoiding Local Traps', fontsize=12)
# #             plt.legend()
# #             plt.grid(True, alpha=0.3)
# #             plt.ylim(0, 1.1)
# #
# #             # Add status indicator
# #             if len(coverage) > 0:
# #                 current_coverage = coverage[-1]
# #                 if current_coverage >= 0.8:
# #                     status = "Excellent - Comprehensive exploration"
# #                     color = "green"
# #                 elif current_coverage >= 0.6:
# #                     status = "Good - Adequate space coverage"
# #                     color = "orange"
# #                 else:
# #                     status = "Limited - Risk of missing global optimum"
# #                     color = "red"
# #
# #                 plt.text(0.02, 0.85, f"Status: {status}\nCoverage: {current_coverage:.1%}",
# #                          transform=plt.gca().transAxes,
# #                          bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
# #                          verticalalignment='top', fontsize=10)
# #
# #             # Plot uncertainty and diversity
# #             plt.subplot(2, 1, 2)
# #             plt.plot(iterations, uncertainty, 'r-s', linewidth=2, markersize=4,
# #                      label='Average Uncertainty', alpha=0.8)
# #             plt.plot(iterations, min_distance, 'g-^', linewidth=2, markersize=4,
# #                      label='Min Distance Between Points', alpha=0.8)
# #
# #             # Threshold lines
# #             plt.axhline(y=self.uncertainty_target, color='red', linestyle=':', alpha=0.7,
# #                         label=f'Uncertainty Target ({self.uncertainty_target:.1f})')
# #             plt.axhline(y=self.min_distance_between_points, color='green', linestyle=':', alpha=0.7,
# #                         label=f'Min Distance Threshold ({self.min_distance_between_points:.1f})')
# #
# #             plt.xlabel('Iteration', fontsize=11)
# #             plt.ylabel('Metric Value', fontsize=11)
# #             plt.title('Exploration Quality Metrics', fontsize=12)
# #             plt.legend()
# #             plt.grid(True, alpha=0.3)
# #
# #             plt.tight_layout()
# #
# #
# # def create_three_separate_bo_gifs():
# #     """Create three separate GIF animations for Bayesian Optimization"""
# #
# #     print("🎯 Creating three separate BO GIF animations...")
# #
# #     # Create three separate animator instances
# #     animator1 = BayesianOptimizationAnimator()  # For BO process
# #     animator2 = BayesianOptimizationAnimator()  # For acquisition function
# #     animator3 = BayesianOptimizationAnimator()  # For exploration metrics
# #
# #     # Set the same random seed for consistency
# #     np.random.seed(42)
# #
# #     # Create BO Process Animation
# #     print("🎯 Creating BO Process animation...")
# #     fig1 = plt.figure(figsize=(12, 8))
# #     anim1 = FuncAnimation(fig1, animator1.animate_bo_process, frames=27,
# #                           interval=1500, repeat=True, blit=False)
# #     anim1.save('bo_process_evolution.gif', writer='pillow', fps=0.67)
# #     plt.close(fig1)
# #     print("✅ BO Process animation saved as 'bo_process_evolution.gif'")
# #
# #     # Create Acquisition Function Animation
# #     print("📈 Creating Acquisition Function animation...")
# #     fig2 = plt.figure(figsize=(12, 8))
# #     anim2 = FuncAnimation(fig2, animator2.animate_acquisition, frames=27,
# #                           interval=1500, repeat=True, blit=False)
# #     anim2.save('bo_acquisition_function.gif', writer='pillow', fps=0.67)
# #     plt.close(fig2)
# #     print("✅ Acquisition Function animation saved as 'bo_acquisition_function.gif'")
# #
# #     # Create Exploration Metrics Animation
# #     print("📊 Creating Exploration Metrics animation...")
# #     fig3 = plt.figure(figsize=(12, 8))
# #     anim3 = FuncAnimation(fig3, animator3.animate_exploration_metrics, frames=27,
# #                           interval=1500, repeat=True, blit=False)
# #     anim3.save('bo_exploration_metrics.gif', writer='pillow', fps=0.67)
# #     plt.close(fig3)
# #     print("✅ Exploration Metrics animation saved as 'bo_exploration_metrics.gif'")
# #
# #     print("\n🎉 All three BO animations created successfully!")
# #     print("Files created:")
# #     print("  - bo_process_evolution.gif (GP model building and strategic sampling)")
# #     print("  - bo_acquisition_function.gif (Expected Improvement guiding exploration)")
# #     print("  - bo_exploration_metrics.gif (Healthy coverage and uncertainty management)")
# #
# #
# # def run_bo_vs_other_methods_comparison():
# #     """Compare BO with other optimization methods"""
# #
# #     def complex_power_loss(x):
# #         """Power loss function for comparison"""
# #         x = np.asarray(x)
# #         base = 150 + 0.8 * (x - 25) ** 2
# #         local_bumps = (25 * np.exp(-3.0 * (x - 17) ** 2) +
# #                        30 * np.exp(-4.0 * (x - 20.5) ** 2) +
# #                        35 * np.exp(-3.5 * (x - 32) ** 2))
# #         oscillations = 5 * np.sin(1.2 * np.pi * x / 4) + 3 * np.cos(0.6 * np.pi * x / 3)
# #         return base - local_bumps + oscillations
# #
# #     print("\n" + "=" * 60)
# #     print("COMPARISON: Bayesian Optimization vs Other Methods")
# #     print("=" * 60)
# #
# #     # Random Search
# #     def random_search(num_evaluations=25):
# #         best_x = None
# #         best_y = float('inf')
# #
# #         for _ in range(num_evaluations):
# #             x = np.random.uniform(15, 35)
# #             y = complex_power_loss(x)
# #
# #             if y < best_y:
# #                 best_x = x
# #                 best_y = y
# #
# #         return best_x, best_y, num_evaluations
# #
# #     # Grid Search
# #     def grid_search(num_points=25):
# #         x_grid = np.linspace(15, 35, num_points)
# #         y_grid = [complex_power_loss(x) for x in x_grid]
# #
# #         best_idx = np.argmin(y_grid)
# #         return x_grid[best_idx], y_grid[best_idx], num_points
# #
# #     # Test different methods
# #     print("\n1. Random Search:")
# #     rs_x, rs_y, rs_evals = random_search()
# #     print(f"   Best: x={rs_x:.2f}mm, Power Loss={rs_y:.1f}W, Evaluations={rs_evals}")
# #
# #     print("\n2. Grid Search:")
# #     gs_x, gs_y, gs_evals = grid_search()
# #     print(f"   Best: x={gs_x:.2f}mm, Power Loss={gs_y:.1f}W, Evaluations={gs_evals}")
# #
# #     print("\n3. Bayesian Optimization:")
# #     bo = BayesianOptimizationAnimator()
# #
# #     # Run BO for comparison
# #     total_evaluations = len(bo.X_observed)  # Initial points
# #     for iteration in range(21):  # 21 more iterations
# #         bo._iterate()
# #         total_evaluations += 1
# #
# #     best_idx = np.argmin(bo.y_observed)
# #     final_best_x = bo.X_observed[best_idx]
# #     final_best_y = bo.y_observed[best_idx]
# #
# #     final_metrics = bo.exploration_metrics[-1]
# #
# #     print(f"   Best: x={final_best_x:.2f}mm, Power Loss={final_best_y:.1f}W")
# #     print(f"   Total Evaluations: {total_evaluations}")
# #     print(f"   Search Space Coverage: {final_metrics['coverage']:.1%}")
# #     print(f"   Final Uncertainty: {final_metrics['avg_uncertainty']:.1f}")
# #
# #     print("\n" + "=" * 60)
# #     print("SUMMARY:")
# #     print("=" * 60)
# #
# #     print(f"Random Search:       {rs_y:.1f}W ({rs_evals} evaluations)")
# #     print(f"Grid Search:         {gs_y:.1f}W ({gs_evals} evaluations)")
# #     print(f"Bayesian Optimization: {final_best_y:.1f}W ({total_evaluations} evaluations)")
# #
# #     print("\nBO Advantages:")
# #     print("- Model-based: builds understanding of the function")
# #     print("- Strategic sampling: balances exploration and exploitation")
# #     print("- Uncertainty quantification: knows where it's uncertain")
# #     print("- Efficient: finds good solutions with fewer evaluations")
# #     print("- Adaptive: learns from each observation to guide future sampling")
# #     print("- Global perspective: maintains coverage to avoid local minima")
# #
# #
# # if __name__ == "__main__":
# #     print("🎯 Bayesian Optimization: Three Separate GIF Animations")
# #     print("=" * 60)
# #     print("This demo creates three separate GIF files showing different aspects of BO:")
# #     print("1. BO Process (GP model building and strategic sampling)")
# #     print("2. Acquisition Function (Expected Improvement guiding exploration)")
# #     print("3. Exploration Metrics (healthy coverage and uncertainty management)")
# #
# #     # Run comparison first
# #     run_bo_vs_other_methods_comparison()
# #
# #     print("\n" + "=" * 60)
# #     print("Creating three separate GIF animations...")
# #     print("=" * 60)
# #
# #     # Create three separate GIFs
# #     create_three_separate_bo_gifs()
#
# import numpy as np
# import matplotlib.pyplot as plt
# from skopt import Optimizer
# import imageio
# import os
#
#
# # Objective function: nonlinear for visible GP adjustment
# def objective(x):
#     return np.sin(2 * x[0]) + 0.1 * x[0]
#
#
# # 1D bounds
# bounds = [(0.0, 10.0)]
#
# # Initialize optimizer
# opt = Optimizer(dimensions=bounds, base_estimator="GP", acq_func="EI", random_state=42)
#
# # Grid and true function
# x_grid = np.linspace(0, 10, 500).reshape(-1, 1)
# true_y = np.sin(2 * x_grid) + 0.1 * x_grid
#
# # Folder for frames
# frame_folder = "frames"
# os.makedirs(frame_folder, exist_ok=True)
# filenames = []
#
# # Number of iterations
# n_iter = 15
#
# for i in range(n_iter):
#     # Ask for next point
#     next_x = opt.ask()
#     y = objective(next_x)
#     opt.tell(next_x, y)
#
#     # GP prediction
#     mu, std = opt.base_estimator_.predict(x_grid, return_std=True)
#
#     # Plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(x_grid, true_y, 'k--', label="True Function")
#     plt.plot(x_grid, mu, 'b-', label="GP Mean Prediction")
#     plt.fill_between(x_grid.ravel(), mu - std, mu + std, color='blue', alpha=0.2, label="GP Uncertainty")
#
#     sampled = np.array(opt.Xi)
#     sampled_y = np.array(opt.yi)
#     plt.scatter(sampled[:, 0], sampled_y, c='red', s=40, label="Sampled Points")
#
#     # Highlight best point in last iteration
#     if i == n_iter - 1:
#         best_index = np.argmin(sampled_y)
#         best_x = sampled[best_index, 0]
#         best_y = sampled_y[best_index]
#         plt.scatter([best_x], [best_y], c='green', s=100, edgecolors='black', label="Best Found", zorder=5)
#         plt.annotate(f"Best: x={best_x:.2f}\nf(x)={best_y:.2f}",
#                      xy=(best_x, best_y),
#                      xytext=(best_x + 0.5, best_y + 0.5),
#                      arrowprops=dict(facecolor='green', shrink=0.05),
#                      fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="green"))
#
#     plt.title(f"Bayesian Optimization - Iteration {i + 1}")
#     plt.xlabel("x")
#     plt.ylabel("f(x)")
#     plt.ylim(-1.5, 2.5)
#     plt.grid(True)
#     plt.legend(loc="upper left")
#
#     # Save to frames folder
#     filename = os.path.join(frame_folder, f"frame_{i}.png")
#     plt.savefig(filename)
#     filenames.append(filename)
#     plt.close()
#
# # Create GIF
# gif_path = "gp_adjusting_with_optimum.gif"
# with imageio.get_writer(gif_path, mode="I", duration=1.0) as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)
#
# print(f"GIF saved as '{gif_path}'")
# print(f"Frames saved in folder: '{frame_folder}'")
#
# # Optional: Uncomment to delete frames after GIF is created
# # for filename in filenames:
# #     os.remove(filename)
# # os.rmdir(frame_folder)
#
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import imageio

# Initial data points (D_piston and power loss)
X_all = np.array([[19.1], [19.3], [19.4]])
y_all = np.array([15, 20, 16])

# Prediction grid
X_grid = np.linspace(19.0, 19.5, 200).reshape(-1, 1)

# Improved kernel: smoother + low noise
kernel = RBF(length_scale=0.1) + WhiteKernel(noise_level=0.01)

# Store plots
filenames = []


# True (hidden) power loss function
def true_function(x):
    return 10 + 50 * (x - 19.25) ** 2


# Bayesian loop: 20 iterations
for i in range(20):
    # Fit GP
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=0)
    gp.fit(X_all, y_all)

    # Predict
    y_pred, sigma = gp.predict(X_grid, return_std=True)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(X_grid, y_pred, 'b-', label='GP mean')
    plt.fill_between(X_grid.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma,
                     alpha=0.2, color='blue', label='95% CI')
    plt.scatter(X_all, y_all, c='red', s=50, label='Observed')
    plt.title(f'Bayesian GP fit - iteration {i + 1}')
    plt.xlabel('D_piston (mm)')
    plt.ylabel('Power loss (kW)')
    plt.legend()
    plt.grid(True)

    # Save frame
    filename = f'gp_frame_{i}.png'
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

    # Acquisition: pick point with best trade-off (mean - uncertainty)
    acquisition = y_pred - 1.0 * sigma
    x_next = X_grid[np.argmin(acquisition)]

    # Simulate true function
    y_next = true_function(x_next)

    # Add to dataset
    X_all = np.vstack([X_all, x_next.reshape(1, -1)])
    y_all = np.append(y_all, y_next)

# Create GIF
with imageio.get_writer('bayesian_gp_fit.gif', mode='I', duration=1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF saved as bayesian_gp_fit.gif")
