# AI-Enhanced Evolutionary Architecture for FractalGenesis

## Executive Summary

This document outlines an integrated architecture for combining evolutionary algorithms with modern AI techniques (VAEs, GANs, Diffusion Models) to create a self-improving, AI-guided fractal generation system. The architecture bridges the gap between parameter space exploration and perceptual quality assessment.

---

## Current System Analysis

### Existing Architecture Strengths

**1. Well-Structured Genome Representation** (`shared/genome.py`)
- Hierarchical gene structure: Camera, Fractal, Color, Lighting
- Normalized parameter ranges (0-1 for most values)
- Built-in diversity calculation via `calculate_diversity()`
- JSON serialization for training data export

**2. Mature Evolution Engine** (`FractalExplorer/genetic_algorithm/`)
- Tournament and rank-based selection strategies
- Elitism with configurable preservation of top individuals
- Multi-component fitness evaluation (user + diversity + novelty)
- Archive-based novelty search (in `NoveltyFitnessEvaluator`)

**3. AI Preference Learning Foundation** (`ai/preference_learner.py`)
- Random Forest classifier for preference prediction
- Feature extraction from Flam3 parameters
- Model persistence and selector plugin system
- Dataset management with JSON export

**4. Multiple Renderers**
- Mandelbulber (external, high-quality)
- Genesis Render (internal, fast)
- Flam3 (2D fractal flames)

### Current Limitations

1. **Manual Fitness**: User must manually select from 4 options per generation
2. **No Image-Based Evaluation**: Fitness is parameter-based, not perceptual
3. **Limited Exploration**: Random mutation may get stuck in local optima
4. **No Generative Models**: Cannot generate novel parameter combinations from learned distributions
5. **Discontinuous Space**: Small parameter changes can cause large visual changes

---

## Proposed AI-Enhanced Architecture

### 1. AI-Driven Parameter Generation Layer

#### 1.1 Variational Autoencoder (VAE) for Parameter Space

```
┌─────────────────────────────────────────────────────────────┐
│                 VAE Parameter Generator                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Learned latent vector z (64-128 dimensions)         │
│         ↓                                                   │
│  Decoder Network:                                           │
│    - FC(128 → 256) → ReLU                                   │
│    - FC(256 → 512) → ReLU                                   │
│    - FC(512 → GenomeParams) → Sigmoid/Tanh                  │
│         ↓                                                   │
│  Output: Valid FractalGenome parameters                     │
│                                                              │
│  Training Data: User-selected genomes (positive examples)   │
│                 + Random/diverse genomes (contrastive)      │
└─────────────────────────────────────────────────────────────┘
```

**Purpose**: Learn a smooth, continuous latent space where nearby points produce similar-looking fractals

**Key Features**:
- **Reparameterization trick** for sampling
- **KL divergence loss** to ensure smooth latent space
- **Parameter constraints** enforced via output activation functions
- **Conditioning** on renderer type (Mandelbulber vs Flam3)

#### 1.2 Conditional Diffusion Model for High-Quality Generation

```
┌─────────────────────────────────────────────────────────────┐
│              Diffusion Parameter Generator                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Training: Denoise from random → valid parameters           │
│                                                              │
│  Inference:                                                 │
│    z_T ~ N(0, I)                                            │
│    for t = T to 1:                                          │
│      z_{t-1} = denoise(z_t, t, condition)                  │
│    ↓                                                        │
│  Output: High-quality genome parameters                     │
│                                                              │
│  Conditioning:                                              │
│  - User preference embedding                                │
│  - Target aesthetic (colorful, dark, symmetric, etc.)      │
│  - Formula type constraints                                 │
└─────────────────────────────────────────────────────────────┘
```

**Purpose**: Generate novel, high-quality parameter combinations beyond simple interpolation

**Advantages**:
- Better mode coverage than VAE
- Can generate from pure noise (true creativity)
- Classifier-free guidance for controllable generation
- Handles multi-modal distributions well

#### 1.3 GAN for Adversarial Parameter Refinement

```
┌─────────────────────────────────────────────────────────────┐
│              GAN Parameter Refiner                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Generator: VAE or Diffusion decoder                        │
│  Discriminator: Distinguishes real vs generated genomes    │
│                                                              │
│  Training: Minimax game                                     │
│    - Generator tries to fool discriminator                  │
│    - Discriminator learns user preference boundaries         │
│                                                              │
│  Use Case: Refine evolved parameters                        │
│  - Take offspring from evolution                           │
│  - Pass through trained generator                           │
│  - Smooths out parameter discontinuities                    │
└─────────────────────────────────────────────────────────────┘
```

**Purpose**: Act as a "beauty filter" for evolved parameters

### 2. Fitness Evaluation System

#### 2.1 Multi-Modal Perceptual Evaluator

```
┌─────────────────────────────────────────────────────────────┐
│          Neural Aesthetic Assessment Pipeline                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1: Image Encoding                                    │
│  ┌──────────────┐    ┌──────────────┐                      │
│  │   Render     │ →  │  Pretrained  │                      │
│  │  (256×256)   │    │   CLIP/VGG   │ → Image embedding    │
│  └──────────────┘    └──────────────┘     (512-dim)        │
│                                                              │
│  Stage 2: Preference Scoring                                │
│  ┌──────────────┐    ┌──────────────┐                      │
│  │   Image      │ →  │   MLP/       │ → Aesthetic score    │
│  │  Embedding   │    │   Transformer│    (0-1)             │
│  └──────────────┘    └──────────────┘                      │
│                    ↑                                         │
│         User preference embedding (learned)                │
│                                                              │
│  Stage 3: Multi-Aspect Scoring                              │
│  - Composition quality                                      │
│  - Color harmony                                            │
│  - Complexity/interest                                      │
│  - Technical quality (noise, artifacts)                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Training Strategy**:
1. **Transfer Learning**: Initialize with CLIP or VGG features
2. **Triplet Loss**: Learn from user selections (selected > not selected)
3. **Active Learning**: Query user on uncertain predictions

#### 2.2 Neural Image Quality Assessment (NIMA)

Adaptation of Google's NIMA for fractals:

```python
class FractalQualityPredictor(nn.Module):
    """Predicts technical quality of fractal renders"""
    
    def __init__(self):
        self.base_cnn = models.mobilenet_v3_small(pretrained=True)
        self.quality_head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)  # Score distribution
        )
        self.attribute_head = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # [detail, noise, artifacts, exposure, contrast]
        )
    
    def forward(self, image):
        features = self.base_cnn.features(image)
        quality_dist = self.quality_head(features)
        attributes = torch.sigmoid(self.attribute_head(features))
        return quality_dist, attributes
```

**Use Cases**:
- Pre-filter low-quality renders before showing user
- Guide evolution away from problematic regions
- Automated quality assurance

#### 2.3 Style-Based Fitness Components

```
┌─────────────────────────────────────────────────────────────┐
│              Style-Aware Fitness System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Learned Style Embeddings:                                  │
│  - Natural/Organic vs Geometric                             │
│  - Dark/Moody vs Bright/Vibrant                             │
│  - Simple vs Complex                                        │
│  - Symmetric vs Asymmetric                                  │
│  - Smooth vs Textured                                       │
│                                                              │
│  User can specify target style vector:                      │
│  target_style = [0.2, 0.8, 0.6, 0.3, 0.7]                  │
│                                                              │
│  Fitness includes: -||style(genome) - target_style||²      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3. Automated Selection & Data Collection

#### 3.1 Active Learning Selection Strategy

```
┌─────────────────────────────────────────────────────────────┐
│           Uncertainty-Guided Selection                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  For each generation:                                       │
│                                                              │
│  1. Generate candidate population                           │
│     - 50% from evolution (crossover + mutation)            │
│     - 30% from AI generator (VAE/Diffusion)                │
│     - 20% from novelty search                              │
│                                                              │
│  2. Pre-filter with learned fitness model                   │
│     - Discard bottom 50% by predicted fitness               │
│                                                              │
│  3. Select 4 candidates for user:                          │
│     - 2 with HIGH uncertainty (disagreement between models)│
│     - 1 with HIGH predicted fitness (exploitation)         │
│     - 1 diverse outlier (exploration)                       │
│                                                              │
│  4. Record selection and update models                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Benefits**:
- Maximizes information gain per user interaction
- Balances exploration vs exploitation
- Builds training dataset efficiently

#### 3.2 Automated Evolution Mode

```
┌─────────────────────────────────────────────────────────────┐
│              Fully Automated Evolution                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  When confidence > threshold (e.g., 50 selections):        │
│                                                              │
│  Fitness = w1 * AI_Preference_Model(genome)                │
│         + w2 * Quality_Predictor(render(genome))           │
│         + w3 * Novelty_Score(genome, archive)              │
│         + w4 * Diversity_Bonus(genome, population)         │
│                                                              │
│  No user interaction required!                              │
│  - Can run overnight to generate 100s of candidates         │
│  - User reviews top results in morning                     │
│  - Feedback refines model further                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4. Latent Space Exploration

#### 4.1 Continuous Parameter Space via VAE

```
┌─────────────────────────────────────────────────────────────┐
│           Smooth Interpolation & Exploration                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  VAE Latent Space Properties:                               │
│  - Local neighborhoods = similar fractals                  │
│  - Linear interpolation = smooth morphing                  │
│  - Arithmetic works: z_new = z1 + 0.5*(z2 - z1)           │
│                                                              │
│  Operations Enabled:                                        │
│  ┌────────────────────────────────────────────────┐        │
│  │ Latent Walk: Move in random direction          │        │
│  │   z_new = z + ε * direction                    │        │
│  │                                                │        │
│  │ Interpolation: Create sequence between two     │        │
│  │   for α in [0, 0.1, 0.2, ..., 1.0]:           │        │
│  │     z_mid = (1-α)*z1 + α*z2                   │        │
│  │                                                │        │
│  │ Arithmetic: Combine properties                 │        │
│  │   z_colorful = z_bright + (z_vibrant - z_base)│        │
│  └────────────────────────────────────────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 4.2 Quality Landscape Mapping

```
┌─────────────────────────────────────────────────────────────┐
│            Fitness Landscape Visualization                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Use dimensionality reduction (t-SNE, UMAP):               │
│  - Project high-dim latent space to 2D                     │
│  - Color by predicted fitness                              │
│  - Mark user selections                                    │
│  - Show evolution trajectory                               │
│                                                              │
│  Benefits:                                                  │
│  - Visualize "good regions" in parameter space             │
│  - Identify unexplored promising areas                     │
│  - Guide mutation directions                               │
│  - Understand user preferences                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 4.3 Multi-Objective Navigation

```python
class LatentSpaceNavigator:
    """Navigate parameter space with multiple objectives"""
    
    def __init__(self, vae, preference_model, quality_model):
        self.vae = vae
        self.preference = preference_model
        self.quality = quality_model
    
    def find_pareto_front(self, objectives):
        """
        Find genomes optimal for multiple criteria:
        - Aesthetic quality
        - Rendering speed
        - Novelty
        - Diversity from current set
        """
        candidates = self.generate_candidates(1000)
        scores = {obj: obj.evaluate(candidates) for obj in objectives}
        return self.pareto_selection(candidates, scores)
    
    def guided_exploration(self, start_z, direction, steps=10):
        """Explore in a specific direction with adaptive step size"""
        trajectory = [start_z]
        current_z = start_z
        
        for _ in range(steps):
            # Evaluate quality at next position
            next_z = current_z + direction
            quality = self.evaluate_quality(next_z)
            
            # Adaptive step: smaller steps near boundaries
            step_size = 0.1 * quality
            current_z = current_z + step_size * direction
            trajectory.append(current_z)
        
        return trajectory
```

---

## Implementation Roadmap

### Phase 1: Foundation (2-3 weeks)

**Deliverables**:
1. **Dataset Pipeline**
   - Export genomes + rendered images + user selections
   - Organize into train/val/test splits
   - Data augmentation (parameter perturbations)

2. **Baseline Neural Evaluator**
   - Pretrained CLIP/VGG feature extractor
   - Simple MLP for preference prediction
   - Triplet loss training on existing data

3. **Integration Point**
   - Extend `AIGuidedFitnessEvaluator` with neural model
   - Keep fallback to user preference when uncertain

### Phase 2: Generative Models (3-4 weeks)

**Deliverables**:
1. **VAE Implementation**
   - Encoder: Genome → latent z
   - Decoder: z → Genome
   - Training on user-selected genomes

2. **Parameter Space Smoothing**
   - Validate latent interpolation produces smooth visual transitions
   - Implement latent space visualization

3. **Enhanced Mutation**
   - Add "VAE-guided mutation": mutate in latent space then decode
   - Compare exploration efficiency vs random mutation

### Phase 3: Full Integration (3-4 weeks)

**Deliverables**:
1. **Diffusion Model**
   - Train conditional diffusion on high-quality genomes
   - Implement classifier-free guidance

2. **Multi-Modal Fitness**
   - Combine parameter-based and image-based evaluation
   - Quality assessment pre-filtering

3. **Active Learning Interface**
   - Uncertainty visualization
   - Smart candidate selection
   - Automated mode when confidence high

### Phase 4: Advanced Features (4-6 weeks)

**Deliverables**:
1. **Style Transfer**
   - Extract style from reference fractals
   - Apply to generated genomes

2. **Animation-Aware Evolution**
   - Ensure temporal coherence in evolved animations
   - Learn smooth transition preferences

3. **Community Learning**
   - Aggregate preferences across users
   - Personalization layer on top of global model

---

## Data Format Specification

### Training Data Schema

```json
{
  "session_id": "uuid",
  "timestamp": "2026-02-09T10:30:00Z",
  "user_id": "optional_user_identifier",
  
  "candidates": [
    {
      "genome_id": "abc123",
      "genome": { /* Full FractalGenome dict */ },
      "renderer_type": "mandelbulber",
      "render_params": { /* Renderer-specific parameters */ },
      "image_path": "renders/gen5_candidate1.png",
      "image_hash": "sha256:...",
      
      "precomputed_features": {
        "clip_embedding": [512 floats],
        "style_embedding": [128 floats],
        "vae_latent": [64 floats],
        "predicted_fitness": 0.82,
        "predicted_quality": 0.91
      }
    }
  ],
  
  "user_selection": {
    "selected_index": 2,
    "selection_time_ms": 3500,
    "confidence": "high",
    "notes": "optional user comment"
  },
  
  "generation_context": {
    "generation_number": 5,
    "population_diversity": 0.73,
    "evolution_config": { /* Mutation rates, etc */ }
  }
}
```

### Model Checkpoint Format

```
ai_models/
├── preference_predictor/
│   ├── model.pt              # PyTorch weights
│   ├── config.yaml           # Architecture hyperparameters
│   ├── feature_stats.json    # Normalization stats
│   └── version_info.json     # Training metadata
│
├── vae_parameter_gen/
│   ├── encoder.pt
│   ├── decoder.pt
│   ├── latent_stats.npz      # μ, σ for normalization
│   └── training_history.json
│
└── diffusion_model/
    ├── unet.pt
    ├── noise_schedule.json
    └── condition_embeddings/
```

---

## Metrics for "Creative Diversity"

### 1. Parameter Space Metrics

```python
def parameter_space_diversity(genomes):
    """Measure diversity in raw parameter space"""
    features = np.array([g.to_feature_vector() for g in genomes])
    
    # Pairwise distances
    distances = pairwise_distances(features)
    
    # Metrics
    avg_distance = np.mean(distances)
    min_distance = np.min(distances[distances > 0])
    coverage = len(set(np.digitize(features[:, i], bins=10) 
                      for i in range(features.shape[1])))
    
    return {
        'mean_distance': avg_distance,
        'min_distance': min_distance,  # Avoid duplicates
        'feature_coverage': coverage,
        'effective_population': estimate_effective_population(distances)
    }
```

### 2. Perceptual Diversity Metrics

```python
def perceptual_diversity(images, clip_model):
    """Measure visual diversity using learned embeddings"""
    embeddings = clip_model.encode(images)
    
    # Cluster analysis
    kmeans = KMeans(n_clusters=5)
    clusters = kmeans.fit_predict(embeddings)
    
    # Style diversity
    style_clusters = cluster_by_style(embeddings)
    
    # Novelty vs archive
    novelty_scores = [min(distance(e, archive_emb) for archive_emb in archive) 
                     for e in embeddings]
    
    return {
        'embedding_spread': np.std(embeddings),
        'cluster_balance': entropy(np.bincount(clusters)),
        'style_coverage': len(style_clusters),
        'avg_novelty': np.mean(novelty_scores)
    }
```

### 3. Evolution Progress Metrics

```python
def evolution_health_stats(history):
    """Assess if evolution is healthy and exploring"""
    recent_diversity = [h['diversity'] for h in history[-10:]]
    
    return {
        'diversity_trend': np.polyfit(range(10), recent_diversity, 1)[0],
        'fitness_improvement_rate': calculate_improvement_rate(history),
        'exploration_efficiency': successful_mutations / total_mutations,
        'convergence_risk': 1.0 - (recent_diversity[-1] / recent_diversity[0]),
        'generation_time': avg_time_per_generation
    }
```

---

## Avoiding Local Optima

### Strategy 1: Multi-Scale Mutation

```python
class AdaptiveMutation:
    """Mutation strength adapts based on progress"""
    
    def __init__(self):
        self.strength_history = []
        self.no_improvement_count = 0
    
    def get_mutation_strength(self):
        # Increase strength when stuck
        if self.no_improvement_count > 5:
            return 0.3  # Large jumps
        elif self.no_improvement_count > 2:
            return 0.15  # Medium exploration
        else:
            return 0.05  # Fine-tuning
    
    def record_outcome(self, improved):
        if improved:
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
```

### Strategy 2: Island Model with Migration

```
┌─────────────────────────────────────────────────────────────┐
│              Island Model Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Population split into N subpopulations (islands):         │
│                                                              │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐        │
│  │Island 1│   │Island 2│   │Island 3│   │Island 4│        │
│  │High    │   │Diverse │   │Novelty │   │Quality │        │
│  │Fitness │   │Preserv.│   │Search  │   │Focus   │        │
│  └───┬────┘   └───┬────┘   └───┬────┘   └───┬────┘        │
│      │            │            │            │              │
│      └────────────┴────────────┴────────────┘              │
│                   Periodic Migration                        │
│                   (every K generations)                     │
│                                                              │
│  Benefits:                                                  │
│  - Different islands explore different niches              │
│  - Migration prevents total convergence                    │
│  - Can focus some islands on exploration, others exploitation│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Strategy 3: Fitness Sharing / Niching

```python
def fitness_sharing(population, fitness_scores, sigma=0.3):
    """
    Reduce fitness of similar individuals to maintain diversity
    """
    shared_fitness = []
    
    for i, (ind, fit) in enumerate(zip(population, fitness_scores)):
        # Count similar individuals
        niche_count = sum(
            1 for j, other in enumerate(population)
            if i != j and ind.calculate_diversity(other) < sigma
        )
        
        # Share fitness
        shared_fitness.append(fit / (1 + niche_count))
    
    return shared_fitness
```

### Strategy 4: Restart with Elites

```python
def detect_convergence(population, threshold=0.1):
    """Detect if population has converged"""
    diversity = calculate_population_diversity(population)
    return diversity < threshold

def restart_with_elites(engine, num_elites=3):
    """Keep best individuals, randomize rest"""
    elites = engine.get_best_genomes(num_elites)
    
    # New random population
    engine.initialize_population()
    
    # Inject elites
    for elite in elites:
        engine.population.individuals[0] = elite
```

---

## Bridge: Parameter Space ↔ Image Space

### The Rendering Bottleneck

**Problem**: Rendering is expensive (1-10 seconds per image)

**Solutions**:

1. **Fast Approximate Renderer** (Genesis Render)
   - Use for initial filtering
   - Full quality render only for top candidates

2. **Neural Renderer** (Neural Radiance Fields)
   - Train network to approximate Mandelbulber
   - Instant inference: z → image
   - Fine-tune on actual renders

3. **Hierarchical Evaluation**
   ```
   Stage 1: Parameter-level (instant)
     - Skip invalid parameter combinations
     - Estimate complexity
   
   Stage 2: Fast render (100ms)
     - Genesis Render at low resolution
     - Filter obvious poor quality
   
   Stage 3: Neural evaluation (10ms)
     - Pretrained quality model
     - Rank candidates
   
   Stage 4: Full render (2s)
     - Only top 10 candidates
     - Final user presentation
   ```

### Differentiable Rendering

For gradient-based optimization:

```python
class DifferentiableFractal(nn.Module):
    """Approximate fractal rendering with differentiable operations"""
    
    def __init__(self):
        self.marcher = SphereMarcher()
        self.shading = DifferentiableShading()
    
    def forward(self, params):
        # Approximate distance field
        distances = self.marcher.estimate(params)
        
        # Differentiable shading
        image = self.shading.apply(distances, params.lighting)
        
        return image
```

**Use Case**: Direct optimization of parameters to match target image

---

## Hardware & Performance Considerations

### Training Infrastructure

```yaml
Minimum Viable:
  GPU: RTX 3060 (12GB)
  RAM: 16GB
  Storage: 100GB SSD
  Training Time: ~2-4 hours per model

Recommended:
  GPU: RTX 4090 (24GB) or A100 (40GB)
  RAM: 32GB
  Storage: 500GB NVMe SSD
  Training Time: ~30-60 minutes per model
```

### Inference Optimization

```python
# ONNX export for fast CPU inference
import onnx
from onnxruntime import InferenceSession

# Export trained model
torch.onnx.export(model, dummy_input, "preference_model.onnx")

# Load optimized model
session = InferenceSession("preference_model.onnx")
# Inference: ~5ms on CPU
```

### Distributed Evolution

```python
# Ray-based distributed evolution
import ray

@ray.remote
def evaluate_genome_remote(genome):
    renderer = get_renderer()
    image = renderer.render(genome)
    score = ai_model.predict(image)
    return score

# Evaluate population in parallel
futures = [evaluate_genome_remote.remote(g) for g in population]
scores = ray.get(futures)
```

---

## Summary

This architecture transforms FractalGenesis from a user-driven evolutionary tool into an AI-augmented creative partner. Key innovations:

1. **Generative Models** learn user preferences and can generate novel, high-quality parameters
2. **Perceptual Evaluation** enables automated fitness assessment without human-in-the-loop
3. **Latent Space Navigation** provides smooth, controllable exploration
4. **Multi-Strategy Diversity** prevents premature convergence
5. **Hierarchical Rendering** balances quality and speed

The result: A system that evolves fractals 10-100x faster, with higher quality, and more diverse results than traditional approaches.

---

## Appendix: Code Integration Points

### Modifying `EvolutionEngine`

```python
class AIGuidedEvolutionEngine(EvolutionEngine):
    """Enhanced evolution with AI guidance"""
    
    def __init__(self, config, renderer_type, use_ai=True):
        super().__init__(config, renderer_type)
        
        if use_ai:
            self.vae = load_vae_model()
            self.preference_model = load_preference_model()
            self.diffusion = load_diffusion_model()
            self.active_learner = ActiveLearningSelector()
    
    def _create_offspring(self, parents):
        """Enhanced offspring creation with AI generation"""
        offspring = []
        
        # Traditional evolution (50%)
        traditional = super()._create_offspring(parents[:len(parents)//2])
        offspring.extend(traditional)
        
        # VAE-guided mutation (30%)
        for parent in parents[len(parents)//2:3*len(parents)//4]:
            z = self.vae.encode(parent)
            z_mutated = z + torch.randn_like(z) * 0.1
            child = self.vae.decode(z_mutated)
            offspring.append(child)
        
        # Diffusion generation (20%)
        for _ in range(len(parents) // 5):
            child = self.diffusion.sample(condition="high_quality")
            offspring.append(child)
        
        return offspring
```

### Extending `FractalGenome`

```python
class AIEnhancedGenome(FractalGenome):
    """Genome with AI-specific features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_representation = None
        self.predicted_fitness = None
        self.style_embedding = None
        self.uncertainty_score = None
    
    def to_feature_vector(self):
        """Flatten genome to vector for ML models"""
        return np.concatenate([
            np.array(self.camera.position),
            np.array(self.camera.target),
            [self.camera.fov],
            [self.fractal.power, self.fractal.bailout, float(self.fractal.iterations)],
            np.array(self.color.base_color),
            [self.color.roughness, self.color.metallic],
            np.array(self.lighting.main_light_direction),
            [self.lighting.main_light_intensity]
        ])
    
    @classmethod
    def from_latent(cls, z, vae_decoder):
        """Create genome from VAE latent code"""
        params = vae_decoder.decode(z)
        return cls.from_parameter_vector(params)
```

---

*End of Architecture Document*
