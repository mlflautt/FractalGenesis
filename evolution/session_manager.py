#!/usr/bin/env python3
"""
Evolution Session Manager
=========================

Model: minimax-m2.5 (opencode)
Created: 2026-02-16
Version: 1.0

Manages user selection sessions for fractal evolution.
- User makes selections (picks favorites)
- Sessions stored for later AI training
- Can reset/load/save sessions
- Supports both still and animation parameters

Usage:
    from evolution.session_manager import SessionManager, SelectionSession
    
    manager = SessionManager()
    session = manager.create_session(name="my_experiment")
    
    # Record a selection
    session.add_selection(fractal_params, rank=1)
    
    # Save for later AI training
    manager.save_session(session)
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StillParams:
    """Parameters for a still fractal image."""
    fractal_type: str = "mandelbulb"
    power: float = 8.0
    iterations: int = 100
    bailout: float = 2.0
    julia_c: tuple = (0.0, 0.0, 0.0)
    scale: float = -1.5
    min_r: float = 0.5
    ifs_scale: float = 2.0
    ifs_folds: int = 3
    lambda_val: float = 1.0
    
    camera_pos: tuple = (0.0, 0.0, -3.0)
    target: tuple = (0.0, 0.0, 0.0)
    fov: float = 45.0
    
    color_palette: str = "warm"
    color_intensity: float = 1.0
    coloring_mode: str = "orbit_trap"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StillParams":
        return cls(**data)


@dataclass
class AnimationParams:
    """Parameters for a fractal animation."""
    start_params: Dict = field(default_factory=dict)
    end_params: Dict = field(default_factory=dict)
    num_frames: int = 30
    fps: int = 10
    anim_type: str = "power"  # power, camera_orbit, color, custom
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AnimationParams":
        return cls(**data)


@dataclass
class FractalSelection:
    """A single selection made by user."""
    selection_id: str
    timestamp: str
    selection_type: str  # "still" or "animation"
    rank: int  # 1 = best, 2 = second, etc.
    still_params: Optional[Dict] = None
    animation_params: Optional[Dict] = None
    image_path: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FractalSelection":
        return cls(**data)


@dataclass 
class SelectionSession:
    """A complete user selection session."""
    session_id: str
    name: str
    created_at: str
    updated_at: str
    population_size: int = 8
    generation: int = 0
    renderer: str = "python_3d"
    selections: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_selection(self, params: Dict, selection_type: str, rank: int, 
                      image_path: str = None, notes: str = ""):
        """Add a selection to this session."""
        selection = FractalSelection(
            selection_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            selection_type=selection_type,
            rank=rank,
            still_params=params if selection_type == "still" else None,
            animation_params=params if selection_type == "animation" else None,
            image_path=image_path,
            notes=notes
        )
        self.selections.append(selection.to_dict())
        self.updated_at = datetime.now().isoformat()
        return selection
    
    def get_selections_by_rank(self, min_rank: int = 1, max_rank: int = 3) -> List[Dict]:
        """Get selections filtered by rank."""
        return [s for s in self.selections if min_rank <= s.get('rank', 0) <= max_rank]
    
    def get_total_selections(self) -> int:
        return len(self.selections)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SelectionSession":
        return cls(**data)


class SessionManager:
    """Manages multiple selection sessions."""
    
    def __init__(self, sessions_dir: str = "output/evolution_sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[SelectionSession] = None
        self.sessions: Dict[str, SelectionSession] = {}
    
    def create_session(self, name: str, population_size: int = 8, 
                      renderer: str = "python_3d") -> SelectionSession:
        """Create a new selection session."""
        session = SelectionSession(
            session_id=str(uuid.uuid4())[:8],
            name=name,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            population_size=population_size,
            renderer=renderer
        )
        self.current_session = session
        self.sessions[session.session_id] = session
        logger.info(f"Created session: {name} ({session.session_id})")
        return session
    
    def load_session(self, session_id: str) -> Optional[SelectionSession]:
        """Load a session from disk."""
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                data = json.load(f)
                session = SelectionSession.from_dict(data)
                self.sessions[session_id] = session
                self.current_session = session
                return session
        return None
    
    def save_session(self, session: SelectionSession = None) -> bool:
        """Save session to disk."""
        s = session or self.current_session
        if not s:
            return False
        
        session_file = self.sessions_dir / f"{s.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(s.to_dict(), f, indent=2)
        
        # Also save as pickle for ML training
        pickle_file = self.sessions_dir / f"{s.session_id}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(s, f)
        
        logger.info(f"Saved session: {s.name} ({s.session_id})")
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            
        session_file = self.sessions_dir / f"{session_id}.json"
        pickle_file = self.sessions_dir / f"{session_id}.pkl"
        
        if session_file.exists():
            session_file.unlink()
        if pickle_file.exists():
            pickle_file.unlink()
            
        if self.current_session and self.current_session.session_id == session_id:
            self.current_session = None
            
        return True
    
    def list_sessions(self) -> List[Dict]:
        """List all saved sessions."""
        sessions = []
        for f in self.sessions_dir.glob("*.json"):
            with open(f, 'r') as fp:
                data = json.load(fp)
                sessions.append({
                    "session_id": data["session_id"],
                    "name": data["name"],
                    "created_at": data["created_at"],
                    "selections": len(data.get("selections", [])),
                    "generation": data.get("generation", 0)
                })
        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)
    
    def get_current_session(self) -> Optional[SelectionSession]:
        return self.current_session
    
    def set_current_session(self, session: SelectionSession):
        self.current_session = session
        self.sessions[session.session_id] = session
    
    def export_for_training(self, session_ids: List[str] = None) -> Dict:
        """Export sessions data for training AI model."""
        if session_ids is None:
            session_ids = [s.session_id for s in self.sessions.values()]
        
        training_data = {
            "exported_at": datetime.now().isoformat(),
            "sessions": [],
            "total_selections": 0
        }
        
        for sid in session_ids:
            if sid in self.sessions:
                session = self.sessions[sid]
                training_data["sessions"].append(session.to_dict())
                training_data["total_selections"] += session.get_total_selections()
        
        return training_data
    
    def reset_session(self, session_id: str = None) -> bool:
        """Reset a session (clear selections but keep config)."""
        s = self.sessions.get(session_id) if session_id else self.current_session
        if s:
            s.selections = []
            s.generation = 0
            s.updated_at = datetime.now().isoformat()
            self.save_session(s)
            return True
        return False


# Quick session for testing
def quick_session():
    """Create a quick test session."""
    manager = SessionManager()
    session = manager.create_session("test_session")
    return manager


if __name__ == "__main__":
    # Demo
    manager = quick_session()
    print("Sessions directory:", manager.sessions_dir)
    print("Current session:", manager.current_session.name)