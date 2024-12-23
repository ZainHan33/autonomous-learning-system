import torch
import torch.nn as nn
from typing import Dict, List
from datetime import datetime

class SurfaceSystem(nn.Module):
    """Surface system for conscious, deliberate processing"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.processor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(0.2)  # Add dropout for regularization
        )
        self.conscious_memory = []
        self.interaction_history = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.processor(x)
    
    def store_interaction(self, interaction: Dict):
        """Store and process new interactions"""
        self.interaction_history.append(interaction)
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]
        
        # Process interaction for conscious memory
        processed = {
            'timestamp': datetime.now(),
            'type': interaction.get('type', 'unknown'),
            'content': interaction.get('content', ''),
            'importance': interaction.get('importance', 0.5)
        }
        self.conscious_memory.append(processed)
        
        # Keep only important recent memories
        self.conscious_memory = sorted(
            self.conscious_memory,
            key=lambda x: (x['importance'], x['timestamp']),
            reverse=True
        )[:100]  # Keep top 100 memories
        
    def get_conscious_state(self) -> Dict:
        """Get current conscious state"""
        if not self.conscious_memory:
            return {'state': 'inactive'}
            
        recent_memories = self.conscious_memory[:5]
        return {
            'state': 'active',
            'current_focus': recent_memories[0],
            'recent_context': recent_memories[1:],
            'interaction_count': len(self.interaction_history)
        }