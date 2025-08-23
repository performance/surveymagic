# src/data_processing/trie_builder.py

from typing import List, Dict, Optional, Any
from collections import defaultdict
import json

from src.data_processing.parser import ConversationTurn
from src.llm_utils.embedding_utils import get_embedding, cosine_similarity
from config.project_config import project_config

class TrieNode:
    """A node in the conversational Trie."""
    def __init__(self, text: str, speaker: Optional[str] = None):
        self.text = text
        self.speaker = speaker
        self.children: List['TrieNode'] = []
        self.participant_ids: List[str] = []
        self.is_leaf = True
        self._embedding: Optional[List[float]] = None

    @property
    def embedding(self) -> List[float]:
        """Lazy-loads and caches the embedding for the node's text."""
        if self._embedding is None:
            self._embedding = get_embedding(self.text)
        return self._embedding

    def find_similar_child(self, text_to_match: str, embedding_to_match: List[float]) -> Optional['TrieNode']:
        """
        Finds a child node that is semantically similar to the given text.
        """
        best_match = None
        highest_similarity = project_config.similarity_threshold_trie_coalescing

        for child in self.children:
            similarity = cosine_similarity(embedding_to_match, child.embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = child
        
        return best_match

    def to_dict(self) -> Dict[str, Any]:
        """Converts the node and its children to a dictionary for serialization."""
        return {
            "text": self.text,
            "speaker": self.speaker,
            "participant_ids": self.participant_ids,
            "children": [child.to_dict() for child in self.children]
        }


class Trie:
    """A conversational Trie data structure."""
    def __init__(self, question_name: str):
        self.root = TrieNode(text=question_name, speaker="root")
        self.question_name = question_name

    def insert(self, conversation: List[ConversationTurn], participant_id: str):
        """
        Inserts a full conversation path into the trie for a given participant.
        """
        if not conversation:
            return

        current_node = self.root
        
        for turn in conversation:
            turn_text = turn['text']
            turn_speaker = turn['speaker']
            
            turn_embedding = get_embedding(turn_text)
            
            matching_child = current_node.find_similar_child(turn_text, turn_embedding)

            if matching_child:
                current_node = matching_child
            else:
                new_node = TrieNode(text=turn_text, speaker=turn_speaker)
                current_node.children.append(new_node)
                current_node.is_leaf = False
                current_node = new_node
        
        # After the last turn, add the participant ID to the final node in the path
        if participant_id not in current_node.participant_ids:
            current_node.participant_ids.append(participant_id)
    
    def get_all_user_responses(self) -> Dict[str, List[str]]:
        """
        Traverses the trie and collects all user responses, grouped by participant ID.
        Returns a dictionary mapping participant_id to a list of their response texts.
        """
        responses = defaultdict(list)
        
        def _traverse(node: TrieNode, path_participants: List[str]):
            # If this node has participant IDs, it's a terminal node for them.
            # We need to trace back the user responses for this path.
            # A simpler approach is to flatten responses per participant first, then analyze.
            # This function is more for structural analysis if needed.
            # For our pipeline, we'll rely on the flattened responses.
            pass # Implementation for path reconstruction can be complex.

        # The actual response collection will happen before trie insertion for simplicity
        # This function can be built out if deeper structural analysis is needed.
        return dict(responses)


    def save_to_json(self, file_path: str):
        """Saves the trie structure to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.root.to_dict(), f, indent=2)