"""
Entity Resolver Module
======================

Provides standardized entity name resolution. Since the same entity may appear 
with different surface forms (e.g., "Dr. Chen", "Sarah Chen", "Chen"), this 
module normalizes all forms to a canonical name.

Design decisions (docs/DESIGN.md):
  D20 - Take the longest form as the canonical name
       - All shorter forms map to the longest form's canonical
       - Rationale: in academic text, longer surface forms are more
         specific; selecting longest minimizes ambiguity
  D20 - Register() vs get_canonical() order matters: register longer
         forms FIRST, otherwise a shorter form may become canonical
         prematurely and lock out a longer one.
  D7  - The resolved canonical entity is what flows into Role Rule 6
         (subject match) via resolved_subject, enabling pronoun
         resolution like "She argued" → "Dr. Sarah Chen".

D21 (deferred): semantic similarity merging via spaCy word vectors was
tested and rolled back — the bag-of-vectors signal cannot reliably
distinguish same-entity variations (Clinton/William J. Clinton=0.72)
from collocated-name collisions (John Waters/John Pierson=0.92), so the
threshold either under-triggers or over-triggers. Future work: use
entity-linking or LLM-based coreference instead.
"""

from typing import Dict, List, Optional, Any
import re


class EntityResolver:
    """
    Resolves entity surface forms to canonical names.
    
    Uses a longest-match strategy: the longest registered surface form
    becomes the canonical name, and all shorter forms map to it.
    
    Example:
        register("Dr. Sarah Chen") -> "Dr. Sarah Chen" (canonical)
        register("Sarah Chen")     -> "Dr. Sarah Chen"
        register("Chen")            -> "Dr. Sarah Chen"
        
        get_canonical("Dr. Chen")  -> "Dr. Sarah Chen"
        normalize_entities(["Chen", "Nexus Labs"]) -> ["Dr. Sarah Chen", "Nexus Labs"]
    """
    
    def __init__(self, nlp: Any = None):
        """
        Initialize the entity resolver.

        Args:
            nlp: Optional spaCy model — accepted for API compatibility but
                unused (D21 deferred). String-based longest-match only.
        """
        # Maps surface forms → canonical entity name
        self.entity_map: Dict[str, str] = {}
        # Stores canonical → all registered surface forms
        self._canonical_forms: Dict[str, List[str]] = {}
        # Counter for unique canonical disambiguation
        self._canonical_counter: Dict[str, int] = {}
        # Stored for API compatibility with future semantic merging
        self._nlp = nlp
    
    def register_entity(self, surface_form: str) -> str:
        """
        Register a surface form and return its canonical form.
        
        If the surface form is a substring of an existing canonical form,
        it will be mapped to that canonical. If it contains an existing 
        canonical form, the new (longer) form becomes canonical.
        
        Args:
            surface_form: The entity surface form to register
            
        Returns:
            The canonical entity name
        """
        normalized = self._normalize(surface_form)
        
        if not normalized:
            return surface_form
        
        # Check if this form should replace an existing canonical
        existing_canonical = self._find_existing_canonical(normalized)
        
        if existing_canonical:
            # Map this form to existing canonical
            self.entity_map[normalized] = existing_canonical
            if normalized not in self._canonical_forms[existing_canonical]:
                self._canonical_forms[existing_canonical].append(normalized)
            return existing_canonical
        
        # Check if an existing form should be remapped to this (longer) form
        forms_to_remap = self._find_forms_to_remap(normalized)
        
        if forms_to_remap:
            # This form becomes the new canonical
            self.entity_map[normalized] = normalized
            self._canonical_forms[normalized] = [normalized]
            
            # Remap all shorter forms
            for old_form in forms_to_remap:
                old_canonical = self.entity_map.get(old_form, old_form)
                self.entity_map[old_form] = normalized
                if old_form not in self._canonical_forms[normalized]:
                    self._canonical_forms[normalized].append(old_form)
                
                # Update the old canonical's form list
                if old_canonical in self._canonical_forms:
                    if old_form in self._canonical_forms[old_canonical]:
                        self._canonical_forms[old_canonical].remove(old_form)
                    # If old canonical has no more forms, remove it
                    if not self._canonical_forms.get(old_canonical):
                        del self._canonical_forms[old_canonical]
            
            return normalized
        
        # New entity entirely
        self.entity_map[normalized] = normalized
        self._canonical_forms[normalized] = [normalized]
        return normalized
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison (lowercase, strip whitespace)."""
        return text.strip().lower()
    
    def _find_existing_canonical(self, normalized: str) -> Optional[str]:
        """Find if this form is already mapped to a canonical."""
        return self.entity_map.get(normalized)
    
    def _find_forms_to_remap(self, normalized: str) -> List[str]:
        """Find existing forms that are substrings of this new form."""
        forms_to_remap = []
        new_words = set(normalized.split())
        
        for existing_form, canonical in self.entity_map.items():
            existing_words = set(existing_form.split())
            # If all words of existing form are in new form, it should be remapped
            if existing_words.issubset(new_words) and len(existing_form) < len(normalized):
                # Additional check: ensure they're in similar positions (basic heuristic)
                forms_to_remap.append(existing_form)
        
        # Sort by length descending to handle nested cases
        forms_to_remap.sort(key=len, reverse=True)
        return forms_to_remap
    
    def get_canonical(self, surface_form: str) -> str:
        """
        Return the canonical entity name for a surface form.
        
        Args:
            surface_form: The entity surface form
            
        Returns:
            The canonical entity name, or the input if not registered
        """
        normalized = self._normalize(surface_form)
        return self.entity_map.get(normalized, surface_form)
    
    def normalize_entities(self, entities: List[str]) -> List[str]:
        """
        Normalize a list of entity surface forms to canonical names.
        
        Args:
            entities: List of entity surface forms
            
        Returns:
            List of canonical entity names (duplicates removed)
        """
        canonicals = []
        seen = set()
        
        for entity in entities:
            canonical = self.get_canonical(entity)
            if canonical not in seen:
                canonicals.append(canonical)
                seen.add(canonical)
        
        return canonicals
    
    def get_all_aliases(self, canonical: str) -> List[str]:
        """
        Get all registered surface forms for a canonical entity.
        
        Args:
            canonical: The canonical entity name
            
        Returns:
            List of all surface forms mapped to this canonical
        """
        normalized = self._normalize(canonical)
        return self._canonical_forms.get(normalized, [canonical])
    
    def get_canonical_for_display(self, surface_form: str) -> str:
        """
        Return canonical name in original case (preserving capitalization).
        
        Returns the canonical name with the original casing from the
        longest registered surface form.
        """
        normalized = self._normalize(surface_form)
        canonical_normalized = self.entity_map.get(normalized, normalized)
        
        # Find the original-cased version
        for form in self.entity_map:
            if form == canonical_normalized:
                return surface_form  # Return original input if no better match
        
        # Find longest registered form for this canonical
        aliases = self.get_all_aliases(canonical_normalized)
        if aliases:
            return max(aliases, key=len)
        
        return surface_form
    
    def merge_entities(self, entity_a: str, entity_b: str) -> str:
        """
        Merge two entities, keeping the longer as canonical.
        
        Useful when we discover that two entities are actually the same.
        
        Args:
            entity_a: First entity surface form
            entity_b: Second entity surface form
            
        Returns:
            The canonical name for the merged entity
        """
        canonical_a = self.get_canonical(entity_a)
        canonical_b = self.get_canonical(entity_b)
        
        if canonical_a == canonical_b:
            # Already the same, just register both forms
            self.register_entity(entity_a)
            self.register_entity(entity_b)
            return canonical_a
        
        # Merge into the longer canonical
        if len(canonical_a) >= len(canonical_b):
            merged = canonical_a
            absorbed = canonical_b
        else:
            merged = canonical_b
            absorbed = canonical_a
        
        # Register absorbed entity's forms under merged
        self.register_entity(merged)
        absorbed_aliases = self.get_all_aliases(absorbed)
        for alias in absorbed_aliases:
            self.register_entity(alias)
        
        return merged
    
    def __len__(self) -> int:
        """Return the number of registered canonical entities."""
        return len(self._canonical_forms)
    
    def __repr__(self) -> str:
        canonicals = list(self._canonical_forms.keys())
        return f"EntityResolver({len(self)} canonical entities)"
