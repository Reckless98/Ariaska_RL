# 🔧 ARIASKA_RL System Fix Plan

## Current Issues Analysis

### 1. Memory Router Parameter Incompatibility
- **Issue**: Enhanced system uses `MemoryRouter(buffer_size=50000, enable_sqlite=True, db_path=..., enable_compression=True, priority_decay=0.99)`
- **Reality**: Actual signature is `MemoryRouter(buffer_size, alpha, beta_start, beta_frames, persistence_path, enable_sqlite)`
- **Fix**: Align parameters with actual MemoryRouter implementation

### 2. Agent Device Parameter Error
- **Issue**: Enhanced system passes `device="cuda"` to agent constructors
- **Reality**: Agent `__init__` methods don't accept `device` parameter
- **Fix**: Remove device parameter from agent initialization or handle GPU internally

### 3. Missing Module Import
- **Issue**: `from core.analytics.stats_monitor import StatsMonitor` fails
- **Reality**: Module is at `core.utils.stats_monitor`
- **Fix**: Correct import path

### 4. GPU Detection Problems
- **Issue**: System shows "CPU Only" despite GPU availability
- **Reality**: Need proper PyTorch CUDA detection and device management
- **Fix**: Implement robust GPU detection and device assignment

### 5. UI Freezing and Poor Visibility
- **Issue**: Dashboard freezes with unclear command/output display
- **Reality**: Rich Live display needs better error handling and non-blocking updates
- **Fix**: Improve dashboard refresh logic and error resilience

## Fix Strategy

### Phase 1: Parameter Compatibility (Critical)
1. Fix MemoryRouter initialization parameters
2. Remove device parameter from agent initialization
3. Fix import paths for stats_monitor

### Phase 2: GPU Detection & Device Management
1. Implement proper CUDA detection
2. Add device management within agents
3. Ensure PyTorch tensors use correct device

### Phase 3: UI Improvements
1. Fix dashboard freezing issues
2. Improve command/output visibility
3. Add better error handling for UI components

### Phase 4: System Integration
1. Test complete training pipeline
2. Validate all agent interactions
3. Ensure memory router functionality

## Expected Outcomes
- ✅ System starts without parameter errors
- ✅ Proper GPU detection and utilization
- ✅ Clear, non-freezing UI with visible commands/outputs
- ✅ All agents initialize and function correctly
- ✅ Memory router operates with correct parameters
- ✅ Training proceeds without critical errors
