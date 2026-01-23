"""
Face Database Manager

Handles:
- Storage and retrieval of face embeddings
- Face matching against database
- Adding new faces/embeddings
"""

import os
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime

import config
from face_processor import calculate_face_distance


@dataclass
class PersonRecord:
    """Record for a registered person"""
    name: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    registered_at: datetime = field(default_factory=datetime.now)
    last_seen: Optional[datetime] = None
    access_count: int = 0


@dataclass
class MatchResult:
    """Result of a database search"""
    found: bool
    person_name: Optional[str]
    distance: float
    second_best_distance: Optional[float]
    margin: Optional[float]
    confidence: float
    message: str


class FaceDatabase:
    """
    Manages the database of registered faces.
    
    Stores face embeddings associated with person names.
    Supports multiple embeddings per person for better matching.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the face database.
        
        Args:
            db_path: Path to the database file
        """
        self.db_path = db_path or config.DATABASE_PATH
        self.persons: Dict[str, PersonRecord] = {}
        self._load_database()
    
    def _load_database(self):
        """Load the database from disk if it exists."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        self.persons = data
                        print(f"Loaded {len(self.persons)} persons from database")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.persons = {}
        else:
            print("No existing database found, starting fresh")
    
    def _save_database(self):
        """Save the database to disk."""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.persons, f)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def register_person(self, name: str, embedding: np.ndarray) -> bool:
        """
        Register a new person or add embedding to existing person.
        
        Args:
            name: Person's name
            embedding: Face embedding vector
            
        Returns:
            True if successful
        """
        if name in self.persons:
            person = self.persons[name]
            # Limit embeddings per person
            if len(person.embeddings) >= config.MAX_EMBEDDINGS_PER_PERSON:
                # Remove oldest embedding
                person.embeddings.pop(0)
            person.embeddings.append(embedding)
            person.last_seen = datetime.now()
        else:
            self.persons[name] = PersonRecord(
                name=name,
                embeddings=[embedding],
                registered_at=datetime.now()
            )
        
        self._save_database()
        return True
    
    def remove_person(self, name: str) -> bool:
        """
        Remove a person from the database.
        
        Args:
            name: Person's name
            
        Returns:
            True if person was removed
        """
        if name in self.persons:
            del self.persons[name]
            self._save_database()
            return True
        return False
    
    def get_all_persons(self) -> List[str]:
        """Get list of all registered person names."""
        return list(self.persons.keys())
    
    def get_person_info(self, name: str) -> Optional[PersonRecord]:
        """Get information about a registered person."""
        return self.persons.get(name)
    
    def search(self, embedding: np.ndarray) -> MatchResult:
        """
        Search the database for a matching face.
        
        Args:
            embedding: Face embedding vector to search for
            
        Returns:
            MatchResult with search outcome
        """
        if not self.persons:
            return MatchResult(
                found=False,
                person_name=None,
                distance=float('inf'),
                second_best_distance=None,
                margin=None,
                confidence=0.0,
                message="Database is empty"
            )
        
        # Calculate distances to all embeddings
        all_distances = []
        
        for name, person in self.persons.items():
            for stored_embedding in person.embeddings:
                distance = calculate_face_distance(embedding, stored_embedding)
                all_distances.append((name, distance))
        
        # Sort by distance
        all_distances.sort(key=lambda x: x[1])
        
        best_name, best_distance = all_distances[0]
        
        # Get second best from different person
        second_best_distance = None
        for name, distance in all_distances[1:]:
            if name != best_name:
                second_best_distance = distance
                break
        
        # Calculate margin
        margin = None
        if second_best_distance is not None:
            margin = second_best_distance - best_distance
        
        # Check if match meets threshold
        if best_distance > config.MATCH_THRESHOLD:
            return MatchResult(
                found=False,
                person_name=best_name,
                distance=best_distance,
                second_best_distance=second_best_distance,
                margin=margin,
                confidence=max(0, 1 - best_distance),
                message=f"No match (distance {best_distance:.3f} > threshold {config.MATCH_THRESHOLD})"
            )
        
        # Check margin if we have multiple people
        if margin is not None and margin < config.MATCH_MARGIN:
            return MatchResult(
                found=False,
                person_name=best_name,
                distance=best_distance,
                second_best_distance=second_best_distance,
                margin=margin,
                confidence=max(0, 1 - best_distance),
                message=f"Match ambiguous (margin {margin:.3f} < required {config.MATCH_MARGIN})"
            )
        
        # Update last seen
        self.persons[best_name].last_seen = datetime.now()
        self.persons[best_name].access_count += 1
        self._save_database()
        
        # Calculate confidence (inverse of distance, clamped)
        confidence = max(0, min(1, 1 - (best_distance / config.MATCH_THRESHOLD)))
        
        return MatchResult(
            found=True,
            person_name=best_name,
            distance=best_distance,
            second_best_distance=second_best_distance,
            margin=margin,
            confidence=confidence,
            message=f"Match found: {best_name}"
        )
    
    def add_embedding_to_matched_person(self, name: str, embedding: np.ndarray) -> bool:
        """
        Add a new embedding to an existing matched person.
        Helps improve future matching accuracy.
        
        Args:
            name: Person's name
            embedding: New embedding to add
            
        Returns:
            True if successful
        """
        if name not in self.persons:
            return False
        
        person = self.persons[name]
        
        # Check if embedding is significantly different from existing ones
        # to avoid adding near-duplicates
        for existing in person.embeddings:
            if calculate_face_distance(embedding, existing) < 0.1:
                return False  # Too similar, don't add
        
        if len(person.embeddings) >= config.MAX_EMBEDDINGS_PER_PERSON:
            person.embeddings.pop(0)
        
        person.embeddings.append(embedding)
        self._save_database()
        return True
    
    def get_database_stats(self) -> dict:
        """Get statistics about the database."""
        total_embeddings = sum(
            len(person.embeddings) for person in self.persons.values()
        )
        return {
            'total_persons': len(self.persons),
            'total_embeddings': total_embeddings,
            'avg_embeddings_per_person': total_embeddings / len(self.persons) if self.persons else 0
        }
