# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import matplotlib.patches as patches
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # Set random seed for reproducibility
# np.random.seed(42)
#
# # PowerPoint-style color scheme
# PPT_COLORS = {
#     'primary': '#2E86AB',  # Blue
#     'secondary': '#A23B72',  # Purple
#     'accent': '#F18F01',  # Orange
#     'success': '#C73E1D',  # Red
#     'background': '#F5F5F5',  # Light gray
#     'text': '#2C3E50',  # Dark blue-gray
#     'grid': '#E8E8E8'  # Light grid
# }
#
#
# class GeneticAlgorithmPresentation:
#     def __init__(self):
#         # Define the same complex power loss function
#         self.true_func = self._complex_power_loss_function
#
#         # Search space
#         self.bounds = (15, 35)
#         self.X_plot = np.linspace(self.bounds[0], self.bounds[1], 300)
#         self.y_plot_true = self.true_func(self.X_plot)
#
#         # Find all local minima for visualization
#         self.local_minima = self._find_local_minima()
#         self.global_minimum = self.local_minima[0] if self.local_minima else {'x': 25.0, 'value': 0.0, 'type': 'global'}
#         self.global_minimum['value'] = 0.0
#
#         # GA Parameters - Optimized for good performance
#         self.population_size = 30
#         self.max_generations = 25
#         self.mutation_rate = 0.25
#         self.crossover_rate = 0.7
#         self.elite_size = 3
#         self.tournament_size = 3
#
#         # Data storage
#         self.generation = 0
#         self.population_history = []
#         self.fitness_history = []
#         self.best_individual_history = []
#         self.best_fitness_history = []
#         self.average_fitness_history = []
#         self.diversity_history = []
#         self.operator_stats = {'crossover': 0, 'mutation': 0, 'selection': 0}
#
#         # Diversity maintenance parameters
#         self.diversity_target = 3.0
#         self.diversity_pressure = True
#
#         # Initialize population
#         self.population = self._initialize_population()
#         self.fitness = self._evaluate_population(self.population)
#         self._update_statistics()
#
#         # Presentation slides data
#         self.slides = self._create_slide_content()
#         self.current_slide = 0
#
#     def _complex_power_loss_function(self, x):
#         """Complex power loss function with multiple local minima"""
#         x = np.asarray(x)
#         base = 150 + 0.8 * (x - 25) ** 2
#         local_bumps = (25 * np.exp(-3.0 * (x - 17) ** 2) +
#                        30 * np.exp(-4.0 * (x - 20.5) ** 2) +
#                        35 * np.exp(-3.5 * (x - 32) ** 2))
#         oscillations = 5 * np.sin(1.2 * np.pi * x / 4) + 3 * np.cos(0.6 * np.pi * x / 3)
#         return base - local_bumps + oscillations
#
#     def _find_local_minima(self):
#         """Find local minima in the function"""
#         x_fine = np.linspace(self.bounds[0], self.bounds[1], 2000)
#         y_fine = self.true_func(x_fine)
#
#         local_minima = []
#         window_size = 20
#
#         for i in range(window_size, len(y_fine) - window_size):
#             local_window = y_fine[i - window_size:i + window_size + 1]
#             if y_fine[i] == np.min(local_window):
#                 surrounding_mean = np.mean(y_fine[i - window_size:i + window_size + 1])
#                 if y_fine[i] < surrounding_mean - 2.0:
#                     too_close = False
#                     for existing in local_minima:
#                         if abs(x_fine[i] - existing['x']) < 1.0:
#                             too_close = True
#                             break
#                     if not too_close:
#                         local_minima.append({
#                             'x': x_fine[i],
#                             'value': y_fine[i],
#                             'type': 'local'
#                         })
#
#         if len(local_minima) == 0:
#             known_minima = [17.0, 20.5, 25.0, 32.0]
#             for x in known_minima:
#                 local_minima.append({
#                     'x': x,
#                     'value': self.true_func(x),
#                     'type': 'local'
#                 })
#
#         local_minima.sort(key=lambda x: x['value'])
#         if local_minima:
#             local_minima[0]['type'] = 'global'
#             for i in range(1, len(local_minima)):
#                 local_minima[i]['type'] = 'local'
#
#         return local_minima
#
#     def _initialize_population(self):
#         """Initialize random population within bounds"""
#         return np.random.uniform(self.bounds[0], self.bounds[1], self.population_size)
#
#     def _evaluate_population(self, population):
#         """Evaluate fitness (negative power loss for minimization)"""
#         return -self.true_func(population)
#
#     def _tournament_selection(self, population, fitness, tournament_size=None):
#         """Tournament selection"""
#         if tournament_size is None:
#             tournament_size = self.tournament_size
#
#         selected = []
#         for _ in range(len(population)):
#             tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
#             tournament_fitness = fitness[tournament_indices]
#             winner_idx = tournament_indices[np.argmax(tournament_fitness)]
#             selected.append(population[winner_idx])
#
#         return np.array(selected)
#
#     def _crossover(self, parent1, parent2):
#         """Single-point crossover with blend"""
#         if np.random.random() < self.crossover_rate:
#             alpha = 0.5
#             child1 = alpha * parent1 + (1 - alpha) * parent2
#             child2 = (1 - alpha) * parent1 + alpha * parent2
#             child1 = np.clip(child1, self.bounds[0], self.bounds[1])
#             child2 = np.clip(child2, self.bounds[0], self.bounds[1])
#             self.operator_stats['crossover'] += 2
#             return child1, child2
#         else:
#             return parent1, parent2
#
#     def _mutate(self, individual):
#         """Adaptive Gaussian mutation"""
#         current_diversity = self._calculate_diversity(self.population) if hasattr(self, 'population') else 3.0
#
#         if self.diversity_pressure and current_diversity < self.diversity_target:
#             adaptive_mutation_rate = min(0.4, self.mutation_rate * 2.0)
#         else:
#             adaptive_mutation_rate = self.mutation_rate
#
#         if np.random.random() < adaptive_mutation_rate:
#             if current_diversity < self.diversity_target:
#                 mutation_strength = (self.bounds[1] - self.bounds[0]) * 0.15
#             else:
#                 mutation_strength = (self.bounds[1] - self.bounds[0]) * 0.08
#
#             mutated = individual + np.random.normal(0, mutation_strength)
#             mutated = np.clip(mutated, self.bounds[0], self.bounds[1])
#             self.operator_stats['mutation'] += 1
#             return mutated
#
#         return individual
#
#     def _calculate_diversity(self, population):
#         """Calculate population diversity"""
#         return np.std(population)
#
#     def _update_statistics(self):
#         """Update generation statistics"""
#         self.population_history.append(self.population.copy())
#         self.fitness_history.append(self.fitness.copy())
#
#         best_idx = np.argmax(self.fitness)
#         self.best_individual_history.append(self.population[best_idx])
#         self.best_fitness_history.append(-self.fitness[best_idx])
#         self.average_fitness_history.append(-np.mean(self.fitness))
#
#         diversity = self._calculate_diversity(self.population)
#         self.diversity_history.append(diversity)
#
#     def _evolve_generation(self):
#         """Evolve one generation with diversity preservation"""
#         if self.generation >= self.max_generations:
#             return
#
#         current_diversity = self._calculate_diversity(self.population)
#         selected_population = self._tournament_selection(self.population, self.fitness)
#         self.operator_stats['selection'] += len(selected_population)
#
#         elite_indices = np.argsort(self.fitness)[-self.elite_size:]
#         elite_individuals = self.population[elite_indices]
#
#         new_population = []
#         new_population.extend(elite_individuals)
#
#         if current_diversity < 2.0:
#             num_random = min(5, self.population_size - len(new_population))
#             random_individuals = np.random.uniform(self.bounds[0], self.bounds[1], num_random)
#             new_population.extend(random_individuals)
#
#         while len(new_population) < self.population_size:
#             parent1, parent2 = np.random.choice(selected_population, 2, replace=False)
#             child1, child2 = self._crossover(parent1, parent2)
#             child1 = self._mutate(child1)
#             child2 = self._mutate(child2)
#             new_population.extend([child1, child2])
#
#         self.population = np.array(new_population[:self.population_size])
#         self.fitness = self._evaluate_population(self.population)
#         self.generation += 1
#         self._update_statistics()
#
#     def _create_slide_content(self):
#         """Create presentation slide content"""
#         return [
#             {
#                 'type': 'title',
#                 'title': 'Genetic Algorithm Optimization',
#                 'subtitle': 'Escaping Local Minima Through Population Diversity',
#                 'author': 'Engineering Optimization Case Study',
#                 'duration': 3
#             },
#             {
#                 'type': 'intro',
#                 'title': 'Problem Overview',
#                 'content': [
#                     '• Complex multi-modal optimization landscape',
#                     '• Multiple local minima trap traditional methods',
#                     '• Global optimum requires intelligent exploration',
#                     '• Population-based approach maintains diversity'
#                 ],
#                 'duration': 4
#             },
#             {
#                 'type': 'evolution',
#                 'title': 'Population Evolution Process',
#                 'subtitle': 'Generation-by-generation adaptation',
#                 'duration': 15
#             },
#             {
#                 'type': 'diversity',
#                 'title': 'Diversity Maintenance Strategy',
#                 'subtitle': 'Preventing premature convergence',
#                 'duration': 15
#             },
#             {
#                 'type': 'convergence',
#                 'title': 'Convergence Analysis',
#                 'subtitle': 'Tracking optimization progress',
#                 'duration': 15
#             },
#             {
#                 'type': 'conclusion',
#                 'title': 'Key Advantages of Genetic Algorithms',
#                 'content': [
#                     '✓ Population-based: Explores multiple regions simultaneously',
#                     '✓ Genetic diversity: Maintains exploration capability',
#                     '✓ Crossover: Combines solutions from different regions',
#                     '✓ Adaptive mutation: Escapes local optima effectively',
#                     '✓ Robust performance: Less likely to get trapped'
#                 ],
#                 'duration': 4
#             }
#         ]
#
#     def _draw_ppt_header(self, title, subtitle=None):
#         """Draw PowerPoint-style header"""
#         # Header background
#         header_rect = patches.Rectangle((0, 0.85), 1, 0.15,
#                                         transform=plt.gca().transAxes,
#                                         facecolor=PPT_COLORS['primary'],
#                                         alpha=0.9, zorder=1000)
#         plt.gca().add_patch(header_rect)
#
#         # Title
#         plt.text(0.02, 0.925, title, transform=plt.gca().transAxes,
#                  fontsize=16, fontweight='bold', color='white', zorder=1001)
#
#         # Subtitle
#         if subtitle:
#             plt.text(0.02, 0.875, subtitle, transform=plt.gca().transAxes,
#                      fontsize=12, color='white', alpha=0.9, zorder=1001)
#
#     def _draw_ppt_footer(self, slide_num, total_slides):
#         """Draw PowerPoint-style footer"""
#         footer_rect = patches.Rectangle((0, 0), 1, 0.05,
#                                         transform=plt.gca().transAxes,
#                                         facecolor=PPT_COLORS['background'],
#                                         alpha=0.8, zorder=1000)
#         plt.gca().add_patch(footer_rect)
#
#         plt.text(0.98, 0.025, f"{slide_num}/{total_slides}",
#                  transform=plt.gca().transAxes,
#                  fontsize=10, ha='right', va='center',
#                  color=PPT_COLORS['text'], zorder=1001)
#
#     def animate_presentation(self, frame):
#         """Main presentation animation function"""
#         plt.clf()
#
#         # Calculate which slide and frame within slide
#         total_frames = sum(slide['duration'] for slide in self.slides)
#         frame = frame % total_frames
#
#         current_frame = 0
#         slide_index = 0
#
#         for i, slide in enumerate(self.slides):
#             if frame < current_frame + slide['duration']:
#                 slide_index = i
#                 slide_frame = frame - current_frame
#                 break
#             current_frame += slide['duration']
#
#         slide = self.slides[slide_index]
#
#         # Set background color
#         plt.gca().set_facecolor(PPT_COLORS['background'])
#
#         if slide['type'] == 'title':
#             self._animate_title_slide(slide, slide_frame)
#         elif slide['type'] == 'intro':
#             self._animate_intro_slide(slide, slide_frame)
#         elif slide['type'] == 'evolution':
#             self._animate_evolution_slide(slide, slide_frame)
#         elif slide['type'] == 'diversity':
#             self._animate_diversity_slide(slide, slide_frame)
#         elif slide['type'] == 'convergence':
#             self._animate_convergence_slide(slide, slide_frame)
#         elif slide['type'] == 'conclusion':
#             self._animate_conclusion_slide(slide, slide_frame)
#
#         self._draw_ppt_footer(slide_index + 1, len(self.slides))
#
#     def _animate_title_slide(self, slide, frame):
#         """Animate title slide"""
#         plt.axis('off')
#
#         # Main title
#         plt.text(0.5, 0.6, slide['title'], ha='center', va='center',
#                  fontsize=24, fontweight='bold', color=PPT_COLORS['primary'],
#                  transform=plt.gca().transAxes)
#
#         # Subtitle
#         plt.text(0.5, 0.45, slide['subtitle'], ha='center', va='center',
#                  fontsize=16, color=PPT_COLORS['secondary'], style='italic',
#                  transform=plt.gca().transAxes)
#
#         # Author
#         plt.text(0.5, 0.25, slide['author'], ha='center', va='center',
#                  fontsize=12, color=PPT_COLORS['text'],
#                  transform=plt.gca().transAxes)
#
#         # Animated logo/icon
#         if frame > 1:
#             circle = patches.Circle((0.5, 0.8), 0.08,
#                                     transform=plt.gca().transAxes,
#                                     facecolor=PPT_COLORS['accent'], alpha=0.7)
#             plt.gca().add_patch(circle)
#             plt.text(0.5, 0.8, 'GA', ha='center', va='center',
#                      fontsize=16, fontweight='bold', color='white',
#                      transform=plt.gca().transAxes)
#
#     def _animate_intro_slide(self, slide, frame):
#         """Animate introduction slide"""
#         plt.axis('off')
#         self._draw_ppt_header(slide['title'])
#
#         # Animate bullet points appearing one by one
#         for i, point in enumerate(slide['content']):
#             if frame > i:
#                 y_pos = 0.7 - i * 0.12
#                 plt.text(0.1, y_pos, point, fontsize=14,
#                          color=PPT_COLORS['text'], transform=plt.gca().transAxes)
#
#     def _animate_evolution_slide(self, slide, frame):
#         """Animate evolution slide with population plot"""
#         self._draw_ppt_header(slide['title'], slide['subtitle'])
#
#         # Update GA data
#         if frame > 0 and self.generation < self.max_generations:
#             self._evolve_generation()
#
#         # Create subplot for the main content
#         plt.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.15)
#
#         if len(self.population_history) > 0:
#             # Plot true function
#             plt.plot(self.X_plot, self.y_plot_true, 'k-', alpha=0.8,
#                      label='Objective Function', linewidth=2)
#
#             # Plot current population
#             current_gen = min(frame, len(self.population_history) - 1)
#             current_population = self.population_history[current_gen]
#             current_fitness = -self.fitness_history[current_gen]
#
#             # Color code population
#             colors = []
#             for individual in current_population:
#                 min_distance = float('inf')
#                 closest_type = 'global'
#                 for minima in self.local_minima:
#                     distance = abs(individual - minima['x'])
#                     if distance < min_distance:
#                         min_distance = distance
#                         closest_type = minima['type']
#
#                 if min_distance < 1.5:
#                     if closest_type == 'global':
#                         colors.append(PPT_COLORS['success'])
#                     else:
#                         colors.append(PPT_COLORS['accent'])
#                 else:
#                     colors.append(PPT_COLORS['primary'])
#
#             # Plot population
#             for x, y, color in zip(current_population, current_fitness, colors):
#                 plt.scatter(x, y, c=color, s=80, alpha=0.8,
#                             edgecolors='white', linewidth=1)
#
#             # Show local minima after first few frames
#             if frame > 2:
#                 for minima in self.local_minima:
#                     if minima['type'] == 'local':
#                         plt.scatter(minima['x'], minima['value'],
#                                     c=PPT_COLORS['accent'], s=120, marker='s',
#                                     edgecolors='black', zorder=15, alpha=0.7)
#
#             plt.xlabel('Piston Diameter (mm)', fontsize=12)
#             plt.ylabel('Power Loss (Watts)', fontsize=12)
#             plt.legend(loc='upper right')
#             plt.grid(True, alpha=0.3, color=PPT_COLORS['grid'])
#
#             # Add generation counter
#             plt.text(0.02, 0.02, f'Generation: {current_gen}',
#                      transform=plt.gca().transAxes,
#                      bbox=dict(boxstyle="round,pad=0.3",
#                                facecolor=PPT_COLORS['primary'], alpha=0.7),
#                      color='white', fontweight='bold')
#
#     def _animate_diversity_slide(self, slide, frame):
#         """Animate diversity slide"""
#         self._draw_ppt_header(slide['title'], slide['subtitle'])
#
#         # Update GA data
#         if frame > 0 and self.generation < self.max_generations:
#             self._evolve_generation()
#
#         plt.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.15)
#
#         if len(self.diversity_history) > 1:
#             generations = range(len(self.diversity_history))
#
#             plt.plot(generations, self.diversity_history, 'o-',
#                      color=PPT_COLORS['success'], linewidth=3, markersize=8,
#                      label='Population Diversity')
#
#             plt.axhline(y=3.0, color=PPT_COLORS['primary'], linestyle='--',
#                         linewidth=2, label='Target Diversity (3.0 mm)')
#             plt.axhline(y=2.0, color=PPT_COLORS['accent'], linestyle=':',
#                         linewidth=2, label='Minimum Threshold (2.0 mm)')
#
#             plt.xlabel('Generation', fontsize=12)
#             plt.ylabel('Diversity (mm)', fontsize=12)
#             plt.legend()
#             plt.grid(True, alpha=0.3, color=PPT_COLORS['grid'])
#
#             # Current status
#             if len(self.diversity_history) > 0:
#                 current_div = self.diversity_history[-1]
#                 if current_div >= 3.0:
#                     status = "Excellent Diversity"
#                     color = PPT_COLORS['success']
#                 elif current_div >= 2.0:
#                     status = "Good Diversity"
#                     color = PPT_COLORS['accent']
#                 else:
#                     status = "Low Diversity"
#                     color = PPT_COLORS['primary']
#
#                 plt.text(0.02, 0.02, f'{status}: {current_div:.2f} mm',
#                          transform=plt.gca().transAxes,
#                          bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
#                          color='white', fontweight='bold')
#
#     def _animate_convergence_slide(self, slide, frame):
#         """Animate convergence slide"""
#         self._draw_ppt_header(slide['title'], slide['subtitle'])
#
#         # Update GA data
#         if frame > 0 and self.generation < self.max_generations:
#             self._evolve_generation()
#
#         plt.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.15)
#
#         if len(self.average_fitness_history) > 0:
#             generations = range(len(self.average_fitness_history))
#
#             plt.plot(generations, self.average_fitness_history, 's-',
#                      color=PPT_COLORS['secondary'], linewidth=3, markersize=6,
#                      label='Average Fitness')
#
#             # Show local minima levels
#             for minima in self.local_minima:
#                 if minima['type'] == 'local':
#                     plt.axhline(y=minima['value'], color=PPT_COLORS['accent'],
#                                 linestyle=':', alpha=0.5)
#
#             plt.xlabel('Generation', fontsize=12)
#             plt.ylabel('Power Loss (Watts)', fontsize=12)
#             plt.legend()
#             plt.grid(True, alpha=0.3, color=PPT_COLORS['grid'])
#
#             # Performance metrics
#             if len(self.average_fitness_history) > 1:
#                 current_avg = self.average_fitness_history[-1]
#                 improvement = self.average_fitness_history[0] - current_avg
#
#                 plt.text(0.02, 0.02, f'Improvement: {improvement:.1f}W',
#                          transform=plt.gca().transAxes,
#                          bbox=dict(boxstyle="round,pad=0.3",
#                                    facecolor=PPT_COLORS['success'], alpha=0.7),
#                          color='white', fontweight='bold')
#
#     def _animate_conclusion_slide(self, slide, frame):
#         """Animate conclusion slide"""
#         plt.axis('off')
#         self._draw_ppt_header(slide['title'])
#
#         # Animate bullet points
#         for i, point in enumerate(slide['content']):
#             if frame > i:
#                 y_pos = 0.65 - i * 0.1
#                 plt.text(0.1, y_pos, point, fontsize=14,
#                          color=PPT_COLORS['text'], transform=plt.gca().transAxes)
#
#
# def create_presentation_gif():
#     """Create PowerPoint-style presentation GIF"""
#     print("🎯 Creating PowerPoint-style Genetic Algorithm Presentation...")
#
#     presenter = GeneticAlgorithmPresentation()
#
#     # Calculate total frames
#     total_duration = sum(slide['duration'] for slide in presenter.slides)
#
#     print(f"📊 Creating {total_duration} frame presentation...")
#
#     fig = plt.figure(figsize=(14, 10))
#     fig.patch.set_facecolor(PPT_COLORS['background'])
#
#     anim = FuncAnimation(fig, presenter.animate_presentation,
#                          frames=total_duration,
#                          interval=2000, repeat=True, blit=False)
#
#     anim.save('genetic_algorithm_presentation.gif', writer='pillow', fps=0.5)
#     plt.close(fig)
#
#     print("✅ Presentation saved as 'genetic_algorithm_presentation.gif'")
#     print("\n🎉 PowerPoint-style GA presentation created successfully!")
#     print("\nSlide Structure:")
#     for i, slide in enumerate(presenter.slides, 1):
#         print(f"  {i}. {slide['title']} ({slide['duration']} frames)")
#
#
# if __name__ == "__main__":
#     print("🎯 Genetic Algorithm: PowerPoint-Style Presentation")
#     print("=" * 60)
#     print("Creating a professional slideshow-style presentation that includes:")
#     print("• Title slide with branding")
#     print("• Problem overview and context")
#     print("• Live population evolution visualization")
#     print("• Diversity maintenance demonstration")
#     print("• Convergence analysis")
#     print("• Summary of key advantages")
#     print("=" * 60)
#
#     create_presentation_gif()
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


# Multi-minima function
def objective(x):
    return np.sin(3 * x) + 0.3 * np.cos(5 * x) + 0.1 * x


POP_SIZE = 30
N_GENERATIONS = 150 # Increased from 50 to 80 for more frames
MUTATION_RATE = 0.3
BOUNDS = (0.0, 10.0)

population = np.random.uniform(BOUNDS[0], BOUNDS[1], POP_SIZE)

frame_folder = "ga_local_global_frames"
os.makedirs(frame_folder, exist_ok=True)
filenames = []


def evaluate_pop(pop):
    return np.array([objective(x) for x in pop])


def roulette_wheel_selection(pop, fitness, num_parents):
    fitness_inv = np.max(fitness) - fitness + 1e-6
    probs = fitness_inv / np.sum(fitness_inv)
    selected_indices = np.random.choice(len(pop), size=num_parents, replace=True, p=probs)
    return pop[selected_indices]


def crossover(parent1, parent2):
    return (parent1 + parent2) / 2.0


def mutate(child):
    if np.random.rand() < MUTATION_RATE:
        child += np.random.normal(0, 0.5)
        child = np.clip(child, BOUNDS[0], BOUNDS[1])
    return child


x_grid = np.linspace(BOUNDS[0], BOUNDS[1], 1000)
true_y = objective(x_grid)

best_so_far = None
best_fitness_so_far = np.inf

for gen in range(N_GENERATIONS):
    fitness = evaluate_pop(population)

    min_idx = np.argmin(fitness)
    if fitness[min_idx] < best_fitness_so_far:
        best_fitness_so_far = fitness[min_idx]
        best_so_far = population[min_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(x_grid, true_y, 'k--', label="True Function")
    plt.scatter(population, fitness, color='red', s=40, label="Population")

    plt.title(f"Genetic Algorithm - Generation {gen + 1}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim(true_y.min() - 1, true_y.max() + 1)
    plt.grid(True)
    plt.legend(loc="upper left")

    filename = os.path.join(frame_folder, f"frame_{gen:03d}.png")
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

    parents = roulette_wheel_selection(population, fitness, POP_SIZE // 2)

    children = []
    while len(children) < POP_SIZE:
        p1, p2 = np.random.choice(parents, 2, replace=False)
        child = crossover(p1, p2)
        child = mutate(child)
        children.append(child)
    population = np.array(children[:POP_SIZE])

# Final frame with best solution marked
plt.figure(figsize=(10, 5))
plt.plot(x_grid, true_y, 'k--', label="True Function")
plt.scatter(population, evaluate_pop(population), color='red', s=40, label="Population")
plt.scatter([best_so_far], [best_fitness_so_far], color='green', s=150, edgecolors='black', label="Best Found",
            zorder=5)
plt.annotate(f"Best: x={best_so_far:.2f}\nf(x)={best_fitness_so_far:.2f}",
             xy=(best_so_far, best_fitness_so_far),
             xytext=(best_so_far + 0.7, best_fitness_so_far + 0.7),
             arrowprops=dict(facecolor='green', shrink=0.05),
             fontsize=12, bbox=dict(boxstyle="round", fc="white", ec="green"))
plt.title(f"Genetic Algorithm - Final Best Solution")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.ylim(true_y.min() - 1, true_y.max() + 1)
plt.grid(True)
plt.legend(loc="upper left")
final_frame = os.path.join(frame_folder, f"frame_final.png")
plt.savefig(final_frame)
plt.close()

filenames.append(final_frame)

# Create GIF with longer duration per frame (2 seconds)
gif_path = "ga_local_global_minima_slow.gif"
with imageio.get_writer(gif_path, mode='I', duration=4.0) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"GIF saved as '{gif_path}'")
print(f"Frames saved in folder: '{frame_folder}'")
print(f"Best solution found at x={best_so_far:.4f} with f(x)={best_fitness_so_far:.4f}")


