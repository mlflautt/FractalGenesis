# Development Workflow & Version Control Strategy

## Branch Strategy

### Main Branches
- **`main`** - Production-ready, stable releases
- **`dev`** - Integration branch for features (current)
- **`release/vX.X`** - Release preparation branches

### Feature Branches (Naming Convention)
- `feature/enhanced-animation-system`
- `feature/deap-evolution`
- `feature/cellular-automata`
- `feature/ai-preference-learning`
- `feature/qt6-migration`
- `feature/audio-reactive`

### Hotfix Branches
- `hotfix/critical-bug-description`

## Workflow

### Starting New Feature
```bash
# From dev branch
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name

# Work on feature...
git add .
git commit -m "descriptive message"
git push origin feature/your-feature-name

# Create PR to dev when ready
```

### Version Tagging
- `v0.1.0` - Initial stable release
- `v0.2.0` - Formula library added
- `v0.3.0` - Animation system
- `v0.4.0` - DEAP evolution
- `v0.5.0` - Cellular automata
- `v0.6.0` - AI integration
- `v1.0.0` - Full release

## Implementation Priority

### Phase 1: Core Animation Enhancement (Weeks 1-2)
- [x] Basic keyframe system (exists)
- [ ] Procedural animation generators
- [ ] Animation presets/templates
- [ ] Batch rendering queue
- [ ] Animation library browser

### Phase 2: DEAP Evolution (Weeks 3-4)
- [ ] Integrate DEAP library
- [ ] Novelty search implementation
- [ ] Phylogenetic tree visualization
- [ ] Population analytics

### Phase 3: Cellular Automata (Weeks 5-6)
- [ ] 3D CA grid implementation
- [ ] CA-driven parameter modulation
- [ ] CA-fractal hybrid modes
- [ ] Predefined CA patterns

### Phase 4: AI Foundation (Weeks 7-8)
- [ ] Data collection pipeline
- [ ] VAE training on fractal dataset
- [ ] CLIP aesthetic scorer
- [ ] Preference learning system

### Phase 5: Integration (Weeks 9-10)
- [ ] Synergistic features
- [ ] Performance optimization
- [ ] UI/UX refinement

## Commit Message Guidelines

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Formatting
- `refactor` - Code restructuring
- `test` - Tests
- `chore` - Maintenance

Examples:
- `feat(animation): add procedural orbit generator`
- `feat(evolution): integrate DEAP with novelty search`
- `fix(rendering): correct distance estimation for negative powers`
- `docs(readme): update installation instructions`

## Reverting Changes

### Revert Last Commit
```bash
git revert HEAD
```

### Revert Specific Commit
```bash
git revert <commit-hash>
```

### Reset to Previous State (DANGEROUS)
```bash
# Soft reset (keep changes)
git reset --soft HEAD~1

# Hard reset (destroy changes)
git reset --hard HEAD~1
```

### Restore Specific File
```bash
git checkout HEAD -- path/to/file
```

## Backup Strategy

Before major changes:
1. Create branch: `git checkout -b backup/pre-feature-name`
2. Push to remote: `git push origin backup/pre-feature-name`
3. Return to dev: `git checkout dev`
4. Start feature work

## Feature Implementation Log

| Feature | Branch | Status | Start Date | End Date |
|---------|--------|--------|------------|----------|
| Enhanced Animation | feature/enhanced-animation | Planned | - | - |
| DEAP Evolution | feature/deap-evolution | Planned | - | - |
| Cellular Automata | feature/cellular-automata | Planned | - | - |
| AI Integration | feature/ai-learning | Planned | - | - |
| Qt6 Migration | feature/qt6-migration | Planned | - | - |

## Testing Checklist Before Merge

- [ ] All existing tests pass
- [ ] New features have basic tests
- [ ] No regressions in rendering
- [ ] UI is responsive
- [ ] Documentation updated
- [ ] Performance acceptable
